"""Run a high-completion-budget recovery check for failed GPT-5 outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.error_experiment import (
    FIELD_NAMES,
    build_messages,
    call_chat,
    canonical_record,
    canonical_truth,
    extract_json_object,
    load_tasks,
)


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "model_outputs_high_budget"
EVALS = ROOT / "results" / "evaluations"


def score_payload(task: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    parsed, parse_error = extract_json_object(payload.get("response_content", ""))
    records_raw = []
    if parsed is not None and isinstance(parsed.get("records"), list):
        records_raw = parsed["records"]
    elif parsed is not None:
        parse_error = "records_not_list"

    predictions = [canonical_record(r) for r in records_raw if isinstance(r, dict)]
    truth = {f["record_id"]: canonical_truth(f) for f in task["facts"]}
    pred_counts = Counter(p["record_id"] for p in predictions)
    first_pred: dict[str, dict[str, Any]] = {}
    for pred in predictions:
        first_pred.setdefault(pred["record_id"], pred)

    matched_ids = sorted(set(truth) & set(first_pred))
    missing_ids = sorted(set(truth) - set(first_pred))
    hallucinated_ids = sorted(pid for pid in set(first_pred) - set(truth) if pid)
    blank_id_count = sum(1 for p in predictions if not p["record_id"])
    duplicate_count = sum(max(0, count - 1) for count in pred_counts.values())

    field_errors = 0
    for record_id in matched_ids:
        pred = first_pred[record_id]
        expected = truth[record_id]
        field_errors += int(any(pred.get(field) != expected.get(field) for field in FIELD_NAMES))

    usage = payload.get("raw_response", {}).get("usage", {}) if isinstance(payload.get("raw_response"), dict) else {}
    return {
        "model": payload["model"],
        "task_id": task["task_id"],
        "target_tokens": task["target_tokens"],
        "fact_count": task["fact_count"],
        "max_completion_tokens": payload["max_completion_tokens"],
        "parse_failure": int(parse_error is not None or "error" in payload),
        "parse_error": parse_error or payload.get("error", ""),
        "target_records": len(truth),
        "predicted_records": len(predictions),
        "missing_records": len(missing_ids),
        "hallucinated_records": len(hallucinated_ids) + blank_id_count,
        "duplicate_records": duplicate_count,
        "matched_with_any_field_error": field_errors,
        "omission_rate": len(missing_ids) / len(truth),
        "hallucination_rate": (len(hallucinated_ids) + blank_id_count) / len(predictions) if predictions else 0.0,
        "duplicate_rate": duplicate_count / len(predictions) if predictions else 0.0,
        "any_field_error_rate_matched": field_errors / len(matched_ids) if matched_ids else None,
        "finish_reason": payload.get("raw_response", {}).get("choices", [{}])[0].get("finish_reason"),
        "api_prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "api_completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--task-ids", nargs="+", default=["scaled_load-24000-0", "scaled_load-24000-1"])
    parser.add_argument("--max-completion-tokens", type=int, default=12_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    EVALS.mkdir(parents=True, exist_ok=True)
    tasks = {t["task_id"]: t for t in load_tasks()}
    rows = []

    for task_id in args.task_ids:
        task = tasks[task_id]
        outpath = OUTDIR / f"{args.model}__{task_id}__max{args.max_completion_tokens}.json"
        if outpath.exists() and not args.force:
            payload = json.loads(outpath.read_text(encoding="utf-8"))
        else:
            import time

            started = time.time()
            try:
                completion = call_chat(args.model, build_messages(task), max_completion_tokens=args.max_completion_tokens)
                payload = {
                    "task_id": task_id,
                    "model": args.model,
                    "max_completion_tokens": args.max_completion_tokens,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.time() - started,
                    "response_content": completion.choices[0].message.content or "",
                    "raw_response": completion.model_dump(mode="json"),
                }
            except Exception as exc:
                payload = {
                    "task_id": task_id,
                    "model": args.model,
                    "max_completion_tokens": args.max_completion_tokens,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.time() - started,
                    "response_content": "",
                    "error": repr(exc),
                }
            outpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(score_payload(task, payload))

    df = pd.DataFrame(rows)
    df.to_csv(EVALS / "high_budget_recovery.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
