"""Generate, run, and score controlled long-document summarization tasks.

The core task uses real long-document corpora as background text and injects
known compliance-incident facts. Hosted LLMs summarize the incident facts, and
the scorer compares JSON outputs against objective ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency is installed by setup.
    tiktoken = None


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
OUTPUTS = RESULTS / "model_outputs"
EVALS = RESULTS / "evaluations"
LOGS = ROOT / "logs"

SEED = 42
LENGTH_BINS = [2_000, 6_000, 12_000, 24_000]
FIXED_FACT_COUNT = 10
SCALED_FACT_COUNTS = {2_000: 4, 6_000: 8, 12_000: 16, 24_000: 32}

UNITS = [
    "Aurora Records Office",
    "Beacon Finance Desk",
    "Cedar Logistics Cell",
    "Delta Safety Board",
    "Elm Training Unit",
    "Falcon Procurement Team",
    "Granite Review Panel",
    "Harbor Systems Group",
    "Iris Compliance Team",
    "Juniper Field Office",
    "Keystone Audit Cell",
    "Lumen Archives Unit",
]

ACTIONS = [
    "reconciled",
    "quarantined",
    "redirected",
    "validated",
    "escalated",
    "archived",
    "inspected",
    "delayed",
    "merged",
    "isolated",
    "recoded",
    "withheld",
]

OBJECTS = [
    "supplier invoices",
    "sensor logs",
    "safety forms",
    "access badges",
    "calibration files",
    "meeting transcripts",
    "expense waivers",
    "shipment manifests",
    "training records",
    "license renewals",
    "inventory tags",
    "inspection photos",
]

MONTHS = [
    "January 2025",
    "February 2025",
    "March 2025",
    "April 2025",
    "May 2025",
    "June 2025",
    "July 2025",
    "August 2025",
    "September 2025",
    "October 2025",
    "November 2025",
    "December 2025",
]

SEVERITIES = ["low", "moderate", "high", "critical"]

FIELD_NAMES = ["unit", "action", "object", "count", "month", "severity"]


def get_encoding():
    """Return a tokenizer suitable for current OpenAI long-context models."""

    if tiktoken is None:
        return None
    for name in ["o200k_base", "cl100k_base"]:
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            continue
    return None


ENCODING = get_encoding()


def token_count(text: str) -> int:
    if ENCODING is None:
        return max(1, len(text.split()))
    return len(ENCODING.encode(text))


def trim_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if ENCODING is None:
        return " ".join(text.split()[:max_tokens])
    ids = ENCODING.encode(text)
    return ENCODING.decode(ids[:max_tokens])


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def ensure_dirs() -> None:
    for path in [RESULTS, OUTPUTS, EVALS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def log_environment(models: list[str]) -> None:
    """Save reproducibility metadata without recording secret values."""

    env_keys = sorted(
        name
        for name in os.environ
        if any(k in name.upper() for k in ["OPENAI", "OPENROUTER", "ANTHROPIC", "GEMINI"])
    )
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - hardware dependent.
        gpu = f"GPU query failed: {exc}"

    model_list: list[str] = []
    if os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI()
            model_list = sorted(m.id for m in client.models.list().data)
        except Exception as exc:
            model_list = [f"OpenAI model list failed: {exc}"]

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "seed": SEED,
        "requested_models": models,
        "available_env_keys": env_keys,
        "gpu": gpu,
        "openai_model_matches": {
            "gpt-4.1": [m for m in model_list if "gpt-4.1" in m],
            "gpt-5": [m for m in model_list if "gpt-5" in m],
        },
    }
    (RESULTS / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_backgrounds() -> list[dict[str, str]]:
    """Load natural long-document backgrounds from the downloaded corpora."""

    backgrounds: list[dict[str, str]] = []
    qmsum = load_from_disk(str(DATASETS / "qmsum_cleaned"))["test"]
    gov = load_from_disk(str(DATASETS / "govreport_summarization"))["test"]

    for idx, row in enumerate(qmsum.select(range(min(180, len(qmsum))))):
        text = clean_text(row["input"])
        if token_count(text) > 1_500:
            backgrounds.append({"source": "QMSum", "source_id": str(row["id"]), "text": text})
    for idx, row in enumerate(gov.select(range(min(180, len(gov))))):
        text = clean_text(row["report"])
        if token_count(text) > 1_500:
            backgrounds.append({"source": "GovReport", "source_id": f"gov-test-{idx}", "text": text})
    if not backgrounds:
        raise RuntimeError("No usable background documents found.")
    return backgrounds


def fact_sentence(fact: dict[str, Any], template_id: int) -> str:
    templates = [
        (
            "Compliance incident note {record_id}: in {month}, the {unit} "
            "{action} {count} {object}; reviewers classified severity as {severity}."
        ),
        (
            "The {unit} filed compliance incident {record_id} after it {action} "
            "{count} {object} in {month}, with {severity} severity."
        ),
        (
            "For compliance incident {record_id}, {month} records show the {unit} "
            "{action} {count} {object}; severity: {severity}."
        ),
        (
            "{month} compliance incident {record_id} states that the {unit} "
            "{action} {count} {object}. Severity was {severity}."
        ),
    ]
    return templates[template_id % len(templates)].format(**fact)


def generate_facts(rng: random.Random, task_index: int, count: int) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for i in range(count):
        record_id = f"IR-{task_index:03d}-{i + 1:02d}"
        if record_id in used_ids:
            raise AssertionError("Duplicate generated record id")
        used_ids.add(record_id)
        facts.append(
            {
                "record_id": record_id,
                "unit": rng.choice(UNITS),
                "action": rng.choice(ACTIONS),
                "object": rng.choice(OBJECTS),
                "count": rng.randint(7, 987),
                "month": rng.choice(MONTHS),
                "severity": rng.choice(SEVERITIES),
            }
        )
    return facts


def assemble_background(rng: random.Random, backgrounds: list[dict[str, str]], target_tokens: int) -> tuple[str, list[str]]:
    chunks: list[str] = []
    sources: list[str] = []
    remaining = target_tokens
    attempts = 0
    while remaining > 0 and attempts < 100:
        attempts += 1
        item = rng.choice(backgrounds)
        text = item["text"]
        ids = ENCODING.encode(text) if ENCODING is not None else text.split()
        total = len(ids)
        if total < 500:
            continue
        take = min(remaining, rng.randint(min(700, remaining), max(min(remaining, 3_000), min(700, remaining))))
        if ENCODING is not None:
            start = 0 if total <= take else rng.randint(0, total - take)
            chunk = ENCODING.decode(ids[start : start + take])
        else:
            start = 0 if total <= take else rng.randint(0, total - take)
            chunk = " ".join(ids[start : start + take])
        chunks.append(clean_text(chunk))
        sources.append(f"{item['source']}:{item['source_id']}")
        remaining = target_tokens - token_count("\n\n".join(chunks))
    return "\n\n".join(chunks), sources


def interleave_facts(background: str, fact_sentences: list[str]) -> str:
    """Scatter fact sentences through background at roughly even intervals."""

    if ENCODING is not None:
        ids = ENCODING.encode(background)
        cuts = [round(i * len(ids) / (len(fact_sentences) + 1)) for i in range(len(fact_sentences) + 2)]
        parts = [ENCODING.decode(ids[cuts[i] : cuts[i + 1]]) for i in range(len(cuts) - 1)]
    else:
        words = background.split()
        cuts = [round(i * len(words) / (len(fact_sentences) + 1)) for i in range(len(fact_sentences) + 2)]
        parts = [" ".join(words[cuts[i] : cuts[i + 1]]) for i in range(len(cuts) - 1)]

    merged: list[str] = []
    for idx, part in enumerate(parts):
        if part.strip():
            merged.append(clean_text(part))
        if idx < len(fact_sentences):
            merged.append(fact_sentences[idx])
    return "\n\n".join(merged)


def generate_tasks(replicates: int) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    backgrounds = load_backgrounds()
    tasks: list[dict[str, Any]] = []
    task_index = 0

    for family in ["fixed_load", "scaled_load"]:
        for target_tokens in LENGTH_BINS:
            for replicate in range(replicates):
                task_index += 1
                fact_count = FIXED_FACT_COUNT if family == "fixed_load" else SCALED_FACT_COUNTS[target_tokens]
                facts = generate_facts(rng, task_index, fact_count)
                fact_sentences = [fact_sentence(f, i) for i, f in enumerate(facts)]
                fact_token_budget = token_count("\n\n".join(fact_sentences))
                background_budget = max(500, target_tokens - fact_token_budget - 80)
                background, source_ids = assemble_background(rng, backgrounds, background_budget)
                document = interleave_facts(background, fact_sentences)
                actual_tokens = token_count(document)
                tasks.append(
                    {
                        "task_id": f"{family}-{target_tokens}-{replicate}",
                        "task_index": task_index,
                        "family": family,
                        "target_tokens": target_tokens,
                        "actual_document_tokens": actual_tokens,
                        "fact_count": fact_count,
                        "replicate": replicate,
                        "source_ids": source_ids,
                        "facts": facts,
                        "document": document,
                    }
                )

    path = RESULTS / "tasks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    sample_path = RESULTS / "task_summary.json"
    summary = [
        {
            "task_id": t["task_id"],
            "family": t["family"],
            "target_tokens": t["target_tokens"],
            "actual_document_tokens": t["actual_document_tokens"],
            "fact_count": t["fact_count"],
            "replicate": t["replicate"],
            "source_count": len(t["source_ids"]),
        }
        for t in tasks
    ]
    sample_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return tasks


def load_tasks() -> list[dict[str, Any]]:
    path = RESULTS / "tasks.jsonl"
    if not path.exists():
        raise FileNotFoundError("results/tasks.jsonl not found; run with --generate first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a careful long-document analyst. Extract only compliance incident "
        "facts explicitly stated in the source. Do not infer missing facts."
    )
    user = f"""
Read the source document and summarize every compliance incident note.

Return only a valid JSON object with this exact shape:
{{
  "records": [
    {{
      "record_id": "IR-000-00",
      "unit": "unit name",
      "action": "single past-tense action",
      "object": "object phrase",
      "count": 0,
      "month": "Month YYYY",
      "severity": "low|moderate|high|critical"
    }}
  ]
}}

Rules:
- Include every compliance incident note in the source document.
- Preserve record_id, unit, action, object, count, month, and severity exactly as stated.
- Do not include ordinary background facts that are not compliance incident notes.
- If there are no compliance incident notes, return {{"records": []}}.
- Return JSON only.

SOURCE DOCUMENT:
<<<
{task["document"]}
>>>
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
def call_chat(model: str, messages: list[dict[str, str]], max_completion_tokens: int) -> Any:
    """Call OpenAI chat completions with conservative fallbacks."""

    client = OpenAI()
    base = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
    }
    attempts = []
    if not model.startswith("gpt-5"):
        attempts.append({**base, "temperature": 0})
    attempts.append(base)
    attempts.append({k: v for k, v in base.items() if k != "response_format"})

    last_exc: Exception | None = None
    for params in attempts:
        try:
            return client.chat.completions.create(**params)
        except Exception as exc:
            last_exc = exc
            message = str(exc)
            if any(term in message.lower() for term in ["unsupported", "unknown parameter", "response_format", "temperature"]):
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenAI call failed without exception")


def output_path(task_id: str, model: str) -> Path:
    safe_model = model.replace("/", "__").replace(":", "_")
    return OUTPUTS / f"{safe_model}__{task_id}.json"


def run_model_calls(tasks: list[dict[str, Any]], models: list[str], force: bool = False) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for real model calls.")

    for model in models:
        for task in tqdm(tasks, desc=f"API calls {model}"):
            path = output_path(task["task_id"], model)
            if path.exists() and not force:
                continue
            messages = build_messages(task)
            started = time.time()
            try:
                completion = call_chat(model, messages, max_completion_tokens=5_000)
                elapsed = time.time() - started
                content = completion.choices[0].message.content or ""
                payload = {
                    "task_id": task["task_id"],
                    "model": model,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "actual_document_tokens": task["actual_document_tokens"],
                    "fact_count": task["fact_count"],
                    "family": task["family"],
                    "target_tokens": task["target_tokens"],
                    "response_content": content,
                    "raw_response": completion.model_dump(mode="json"),
                }
            except Exception as exc:
                elapsed = time.time() - started
                payload = {
                    "task_id": task["task_id"],
                    "model": model,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "actual_document_tokens": task["actual_document_tokens"],
                    "fact_count": task["fact_count"],
                    "family": task["family"],
                    "target_tokens": task["target_tokens"],
                    "error": repr(exc),
                    "response_content": "",
                }
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = text.strip()
    if not text:
        return None, "empty"
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None, None if isinstance(obj, dict) else "json_not_object"
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None, "no_open_brace"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None, None if isinstance(obj, dict) else "json_not_object"
                    except json.JSONDecodeError as exc:
                        return None, f"json_decode_error:{exc}"
    return None, "unbalanced_braces"


def norm_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_phrase(value: Any, strip_leading_number: bool = False) -> str:
    """Normalize short factual fields while ignoring harmless packaging tokens."""

    text = norm_text(value)
    if strip_leading_number:
        text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text


MONTH_LOOKUP = {
    "jan": "january",
    "january": "january",
    "feb": "february",
    "february": "february",
    "mar": "march",
    "march": "march",
    "apr": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "june": "june",
    "jul": "july",
    "july": "july",
    "aug": "august",
    "august": "august",
    "sep": "september",
    "sept": "september",
    "september": "september",
    "oct": "october",
    "october": "october",
    "nov": "november",
    "november": "november",
    "dec": "december",
    "december": "december",
}


def norm_month(value: Any) -> str:
    text = norm_text(value)
    parts = text.split()
    if not parts:
        return ""
    month = MONTH_LOOKUP.get(parts[0], parts[0])
    year = next((p for p in parts if re.fullmatch(r"20\d{2}", p)), "")
    return f"{month} {year}".strip()


def parse_count(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None


def canonical_record(raw: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "record_id": ["record_id", "recordId", "id", "incident_id", "incidentId"],
        "unit": ["unit", "team", "office"],
        "action": ["action", "verb"],
        "object": ["object", "item", "records", "affected_object"],
        "count": ["count", "number", "quantity"],
        "month": ["month", "date"],
        "severity": ["severity", "level"],
    }
    out: dict[str, Any] = {}
    for target, keys in key_map.items():
        for key in keys:
            if key in raw:
                out[target] = raw[key]
                break
        else:
            out[target] = ""
    out["record_id"] = str(out["record_id"]).strip()
    out["count"] = parse_count(out["count"])
    out["month"] = norm_month(out["month"])
    out["unit"] = norm_phrase(out["unit"])
    out["action"] = norm_phrase(out["action"])
    out["object"] = norm_phrase(out["object"], strip_leading_number=True)
    out["severity"] = norm_phrase(out["severity"])
    return out


def canonical_truth(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": fact["record_id"],
        "unit": norm_phrase(fact["unit"]),
        "action": norm_phrase(fact["action"]),
        "object": norm_phrase(fact["object"], strip_leading_number=True),
        "count": int(fact["count"]),
        "month": norm_month(fact["month"]),
        "severity": norm_phrase(fact["severity"]),
    }


def score_outputs(tasks: list[dict[str, Any]], models: list[str]) -> None:
    run_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    task_by_id = {t["task_id"]: t for t in tasks}
    for model in models:
        for task in tasks:
            path = output_path(task["task_id"], model)
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            content = payload.get("response_content", "")
            parsed, parse_error = extract_json_object(content)
            records_raw = []
            if parsed is not None:
                records_raw = parsed.get("records", [])
                if not isinstance(records_raw, list):
                    parse_error = "records_not_list"
                    records_raw = []
            predictions = [canonical_record(r) for r in records_raw if isinstance(r, dict)]
            truth = {f["record_id"]: canonical_truth(f) for f in task_by_id[task["task_id"]]["facts"]}
            pred_counts = Counter(p["record_id"] for p in predictions)
            first_pred: dict[str, dict[str, Any]] = {}
            for pred in predictions:
                if pred["record_id"] not in first_pred:
                    first_pred[pred["record_id"]] = pred

            matched_ids = sorted(set(truth) & set(first_pred))
            missing_ids = sorted(set(truth) - set(first_pred))
            hallucinated_ids = sorted(pid for pid in set(first_pred) - set(truth) if pid)
            blank_id_count = sum(1 for p in predictions if not p["record_id"])
            duplicate_count = sum(max(0, count - 1) for count in pred_counts.values())

            for pred_index, pred in enumerate(predictions):
                pred_rows.append(
                    {
                        "model": model,
                        "task_id": task["task_id"],
                        "family": task["family"],
                        "target_tokens": task["target_tokens"],
                        "actual_document_tokens": task["actual_document_tokens"],
                        "fact_count": task["fact_count"],
                        "replicate": task["replicate"],
                        "prediction_index": pred_index,
                        "record_id": pred["record_id"],
                        "hallucinated": int(pred["record_id"] not in truth),
                        "duplicate": int(pred_counts[pred["record_id"]] > 1 and pred_index != next(i for i, p in enumerate(predictions) if p["record_id"] == pred["record_id"])),
                    }
                )

            for record_id, expected in truth.items():
                pred = first_pred.get(record_id)
                row = {
                    "model": model,
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "target_tokens": task["target_tokens"],
                    "actual_document_tokens": task["actual_document_tokens"],
                    "fact_count": task["fact_count"],
                    "replicate": task["replicate"],
                    "record_id": record_id,
                    "omission": int(pred is None),
                }
                if pred is None:
                    for field in FIELD_NAMES:
                        row[f"{field}_error"] = None
                    row["any_field_error"] = None
                    row["exact_record"] = 0
                else:
                    field_errors = {}
                    for field in FIELD_NAMES:
                        field_errors[f"{field}_error"] = int(pred.get(field) != expected.get(field))
                    row.update(field_errors)
                    row["any_field_error"] = int(any(field_errors.values()))
                    row["exact_record"] = int(not any(field_errors.values()))
                record_rows.append(row)

            matched_field_error_count = 0
            for record_id in matched_ids:
                pred = first_pred[record_id]
                expected = truth[record_id]
                matched_field_error_count += int(any(pred.get(field) != expected.get(field) for field in FIELD_NAMES))

            usage = payload.get("raw_response", {}).get("usage", {}) if isinstance(payload.get("raw_response"), dict) else {}
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            run_rows.append(
                {
                    "model": model,
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "target_tokens": task["target_tokens"],
                    "actual_document_tokens": task["actual_document_tokens"],
                    "fact_count": task["fact_count"],
                    "replicate": task["replicate"],
                    "parse_failure": int(parse_error is not None or "error" in payload),
                    "parse_error": parse_error or payload.get("error", ""),
                    "target_records": len(truth),
                    "predicted_records": len(predictions),
                    "matched_records": len(matched_ids),
                    "missing_records": len(missing_ids),
                    "hallucinated_records": len(hallucinated_ids) + blank_id_count,
                    "duplicate_records": duplicate_count,
                    "matched_with_any_field_error": matched_field_error_count,
                    "omission_rate": len(missing_ids) / len(truth) if truth else 0.0,
                    "hallucination_rate": (len(hallucinated_ids) + blank_id_count) / len(predictions) if predictions else 0.0,
                    "duplicate_rate": duplicate_count / len(predictions) if predictions else 0.0,
                    "any_field_error_rate_matched": matched_field_error_count / len(matched_ids) if matched_ids else None,
                    "exact_record_rate_target": sum(1 for r in record_rows if r["task_id"] == task["task_id"] and r["model"] == model and r["exact_record"] == 1) / len(truth) if truth else 0.0,
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "api_prompt_tokens": input_tokens,
                    "api_completion_tokens": output_tokens,
                }
            )

    write_jsonl(EVALS / "run_metrics.jsonl", run_rows)
    write_jsonl(EVALS / "record_metrics.jsonl", record_rows)
    write_jsonl(EVALS / "prediction_metrics.jsonl", pred_rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-4.1-mini", "gpt-5-mini"])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    ensure_dirs()
    log_environment(args.models)

    if args.all or args.generate:
        tasks = generate_tasks(args.replicates)
    else:
        tasks = load_tasks()

    if args.all or args.run_api:
        run_model_calls(tasks, args.models, force=args.force)

    if args.all or args.score:
        score_outputs(tasks, args.models)


if __name__ == "__main__":
    main()
