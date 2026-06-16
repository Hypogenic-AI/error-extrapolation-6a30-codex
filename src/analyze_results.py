"""Analyze scored error-extrapolation results and generate tables/figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVALS = RESULTS / "evaluations"
FIGURES = ROOT / "figures"

SEED = 42
RUN_RATE_METRICS = [
    "omission_rate",
    "hallucination_rate",
    "duplicate_rate",
    "any_field_error_rate_matched",
    "exact_record_error_rate",
]
RECORD_OUTCOMES = [
    "omission",
    "exact_record_error",
    "any_field_error",
    "unit_error",
    "action_error",
    "object_error",
    "count_error",
    "month_error",
    "severity_error",
]
PREDICTION_OUTCOMES = ["hallucinated", "duplicate"]


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def bootstrap_ci(values: pd.Series, n_boot: int = 2_000, alpha: float = 0.05) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(SEED)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize_rates(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in run_df.groupby(["model", "family", "target_tokens", "fact_count"], dropna=False):
        model, family, target_tokens, fact_count = keys
        for metric in RUN_RATE_METRICS:
            lo, hi = bootstrap_ci(group[metric])
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "target_tokens": target_tokens,
                    "fact_count": fact_count,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std": group[metric].std(ddof=1),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_runs": len(group),
                }
            )
    return pd.DataFrame(rows)


def fit_glm(df: pd.DataFrame, outcome: str, formula_rhs: str) -> dict[str, Any] | None:
    data = df.dropna(subset=[outcome, "actual_document_tokens", "model", "family", "fact_count"]).copy()
    if len(data) < 8 or data[outcome].nunique() < 2:
        return None
    data["log2_tokens"] = np.log2(data["actual_document_tokens"])
    data[outcome] = data[outcome].astype(float)
    formula = f"{outcome} ~ {formula_rhs}"
    try:
        model = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()
    except Exception as exc:
        return {"outcome": outcome, "formula": formula, "error": repr(exc), "n": len(data)}
    if "log2_tokens" not in model.params.index:
        return None
    coef = float(model.params["log2_tokens"])
    se = float(model.bse["log2_tokens"])
    p_value = float(model.pvalues["log2_tokens"])
    lo_coef = coef - 1.96 * se
    hi_coef = coef + 1.96 * se
    return {
        "outcome": outcome,
        "formula": formula,
        "n": int(len(data)),
        "events": int(data[outcome].sum()),
        "coef_log2_tokens": coef,
        "se_log2_tokens": se,
        "odds_ratio_per_2x_tokens": safe_exp(coef),
        "ci95_low_or": safe_exp(lo_coef),
        "ci95_high_or": safe_exp(hi_coef),
        "p_value": p_value,
    }


def safe_exp(value: float) -> float:
    """Exponentiate while preserving separation-induced overflow as infinity."""

    if value > 700:
        return float("inf")
    if value < -700:
        return 0.0
    return float(math.exp(value))


def fit_slopes(record_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    record_df = record_df.copy()
    record_df["exact_record_error"] = 1 - record_df["exact_record"]
    for outcome in RECORD_OUTCOMES:
        if outcome not in record_df:
            continue
        for family, group in record_df.groupby("family"):
            result = fit_glm(group, outcome, "log2_tokens + C(model)")
            if result:
                result["family_scope"] = family
                result["level"] = "target_record"
                rows.append(result)
        result = fit_glm(record_df, outcome, "log2_tokens + fact_count + C(model) + C(family)")
        if result:
            result["family_scope"] = "pooled_adjusted"
            result["level"] = "target_record"
            rows.append(result)

    if not pred_df.empty:
        for outcome in PREDICTION_OUTCOMES:
            if outcome not in pred_df:
                continue
            for family, group in pred_df.groupby("family"):
                result = fit_glm(group, outcome, "log2_tokens + C(model)")
                if result:
                    result["family_scope"] = family
                    result["level"] = "predicted_record"
                    rows.append(result)
            result = fit_glm(pred_df, outcome, "log2_tokens + fact_count + C(model) + C(family)")
            if result:
                result["family_scope"] = "pooled_adjusted"
                result["level"] = "predicted_record"
                rows.append(result)

    slopes = pd.DataFrame(rows)
    if "p_value" in slopes:
        mask = slopes["p_value"].notna()
        slopes.loc[mask, "p_value_bh"] = multipletests(slopes.loc[mask, "p_value"], method="fdr_bh")[1]
        slopes["direction"] = np.where(slopes["coef_log2_tokens"] > 0, "increases", "decreases")
        slopes["interpretation"] = slopes.apply(classify_slope, axis=1)
    return slopes


def classify_slope(row: pd.Series) -> str:
    if pd.isna(row.get("p_value_bh")):
        return "not_estimated"
    or_value = row.get("odds_ratio_per_2x_tokens", math.nan)
    p_adj = row.get("p_value_bh", math.nan)
    lo = row.get("ci95_low_or", math.nan)
    hi = row.get("ci95_high_or", math.nan)
    if or_value > 1 and lo > 1 and p_adj < 0.05:
        return "grows"
    if or_value < 1 and hi < 1 and p_adj < 0.05:
        return "declines"
    if 0.8 <= or_value <= 1.25:
        return "approximately_stable"
    if or_value > 1.25:
        return "suggestive_growth"
    if or_value < 0.8:
        return "suggestive_decline"
    return "ambiguous"


def descriptive_stats(run_df: pd.DataFrame, record_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, Any]:
    stats_dict: dict[str, Any] = {
        "n_runs": int(len(run_df)),
        "n_target_records": int(len(record_df)),
        "n_predicted_records": int(len(pred_df)),
        "models": sorted(run_df["model"].dropna().unique().tolist()),
        "families": sorted(run_df["family"].dropna().unique().tolist()),
        "target_token_bins": sorted(run_df["target_tokens"].dropna().unique().tolist()),
        "parse_failures": int(run_df["parse_failure"].sum()),
        "total_prompt_tokens": int(run_df["api_prompt_tokens"].dropna().sum()) if "api_prompt_tokens" in run_df else 0,
        "total_completion_tokens": int(run_df["api_completion_tokens"].dropna().sum()) if "api_completion_tokens" in run_df else 0,
        "elapsed_seconds_total": float(run_df["elapsed_seconds"].dropna().sum()) if "elapsed_seconds" in run_df else math.nan,
    }
    for metric in RUN_RATE_METRICS:
        stats_dict[metric] = {
            "mean": float(run_df[metric].mean()),
            "std": float(run_df[metric].std(ddof=1)),
            "min": float(run_df[metric].min()),
            "max": float(run_df[metric].max()),
        }
    return stats_dict


def sensitivity_summary(run_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize how conclusions change when parse failures are excluded or recovered."""

    def metric_block(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_runs": int(len(df)),
            "parse_failures": int(df["parse_failure"].sum()) if "parse_failure" in df else 0,
            "mean_omission_rate": float(df["omission_rate"].mean()) if len(df) else math.nan,
            "mean_hallucination_rate": float(df["hallucination_rate"].mean()) if len(df) else math.nan,
            "mean_duplicate_rate": float(df["duplicate_rate"].mean()) if len(df) else math.nan,
            "mean_any_field_error_rate_matched": float(df["any_field_error_rate_matched"].mean()) if len(df) else math.nan,
            "mean_exact_record_rate_target": float(df["exact_record_rate_target"].mean()) if "exact_record_rate_target" in df and len(df) else math.nan,
        }

    out = {
        "main_all_runs": metric_block(run_df),
        "excluding_parse_failures": metric_block(run_df[run_df["parse_failure"] == 0]),
    }
    recovery_path = EVALS / "high_budget_recovery.csv"
    if recovery_path.exists():
        recovery = pd.read_csv(recovery_path)
        out["high_budget_recovery_failed_cases"] = {
            "n_runs": int(len(recovery)),
            "parse_failures": int(recovery["parse_failure"].sum()),
            "mean_omission_rate": float(recovery["omission_rate"].mean()),
            "mean_hallucination_rate": float(recovery["hallucination_rate"].mean()),
            "mean_duplicate_rate": float(recovery["duplicate_rate"].mean()),
            "mean_any_field_error_rate_matched": float(recovery["any_field_error_rate_matched"].mean()),
            "mean_reasoning_tokens": float(recovery["reasoning_tokens"].mean()),
            "mean_completion_tokens": float(recovery["api_completion_tokens"].mean()),
        }
    return out


def plot_rates(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_df = summary[summary["metric"].isin(RUN_RATE_METRICS)].copy()
    label_map = {
        "omission_rate": "Omission",
        "hallucination_rate": "Hallucination",
        "duplicate_rate": "Duplicate",
        "any_field_error_rate_matched": "Field error among matched",
        "exact_record_error_rate": "Exact-record error",
    }
    plot_df["metric_label"] = plot_df["metric"].map(label_map)

    sns.set_theme(style="whitegrid", context="notebook")
    metrics = [label_map[m] for m in RUN_RATE_METRICS]
    families = ["fixed_load", "scaled_load"]
    models = sorted(plot_df["model"].unique())
    fig, axes = plt.subplots(len(metrics), len(families), figsize=(12, 12), sharex=True)
    colors = dict(zip(models, sns.color_palette("deep", n_colors=len(models))))

    for row, metric_label in enumerate(metrics):
        for col, family in enumerate(families):
            ax = axes[row, col]
            sub = plot_df[(plot_df["metric_label"] == metric_label) & (plot_df["family"] == family)]
            for model in models:
                msub = sub[sub["model"] == model].sort_values("target_tokens")
                if msub.empty:
                    continue
                ax.plot(
                    msub["target_tokens"],
                    msub["mean"],
                    marker="o",
                    linewidth=2,
                    label=model,
                    color=colors[model],
                )
            ax.set_xscale("log")
            ax.set_ylim(bottom=-0.02)
            ax.set_ylabel(metric_label)
            if row == 0:
                ax.set_title(family)
            if row == len(metrics) - 1:
                ax.set_xlabel("Target source tokens")
            else:
                ax.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False)
    fig.suptitle("Error rates by source length", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(FIGURES / "error_rates_by_length.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_slope_heatmap(slopes: pd.DataFrame) -> None:
    if slopes.empty or "odds_ratio_per_2x_tokens" not in slopes:
        return
    subset = slopes[(slopes["family_scope"].isin(["fixed_load", "scaled_load"]))].copy()
    if subset.empty:
        return
    subset["log2_or"] = np.log2(subset["odds_ratio_per_2x_tokens"].replace(0, np.nan))
    subset["log2_or_plot"] = subset["log2_or"].replace([np.inf, -np.inf], [4.0, -4.0]).clip(-4, 4)
    subset["annot"] = subset["odds_ratio_per_2x_tokens"].apply(
        lambda x: "inf" if np.isinf(x) else ("0" if x == 0 else f"{x:.2f}")
    )
    pivot = subset.pivot_table(
        index="outcome",
        columns="family_scope",
        values="log2_or_plot",
        aggfunc="first",
    )
    annot = subset.pivot_table(index="outcome", columns="family_scope", values="annot", aggfunc="first").reindex_like(pivot)
    plt.figure(figsize=(7.5, max(4, 0.5 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="vlag",
        center=0.0,
        vmin=-4,
        vmax=4,
        linewidths=0.5,
        cbar_kws={"label": "log2 odds ratio per 2x tokens (capped)"},
    )
    plt.title("Length slope by error type")
    plt.xlabel("Task family")
    plt.ylabel("Error outcome")
    plt.tight_layout()
    plt.savefig(FIGURES / "length_slope_heatmap.png", dpi=180)
    plt.close()


def spearman_table(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in RUN_RATE_METRICS:
        for (model, family), group in run_df.groupby(["model", "family"]):
            if group[metric].nunique(dropna=True) < 2:
                rho, p_value = (math.nan, math.nan)
            else:
                rho, p_value = stats.spearmanr(group["actual_document_tokens"], group[metric], nan_policy="omit")
            rows.append({"model": model, "family": family, "metric": metric, "spearman_rho": rho, "p_value": p_value})
    table = pd.DataFrame(rows)
    mask = table["p_value"].notna()
    if mask.any():
        table.loc[mask, "p_value_bh"] = multipletests(table.loc[mask, "p_value"], method="fdr_bh")[1]
    return table


def main() -> None:
    run_df = read_jsonl(EVALS / "run_metrics.jsonl")
    record_df = read_jsonl(EVALS / "record_metrics.jsonl")
    pred_df = read_jsonl(EVALS / "prediction_metrics.jsonl")

    run_df["exact_record_error_rate"] = 1 - run_df["exact_record_rate_target"]

    for df in [run_df, record_df, pred_df]:
        if not df.empty:
            df["log2_tokens"] = np.log2(df["actual_document_tokens"])

    run_df.to_csv(EVALS / "run_metrics.csv", index=False)
    record_df.to_csv(EVALS / "record_metrics.csv", index=False)
    pred_df.to_csv(EVALS / "prediction_metrics.csv", index=False)

    summary = summarize_rates(run_df)
    summary.to_csv(EVALS / "summary_by_condition.csv", index=False)

    slopes = fit_slopes(record_df, pred_df)
    slopes.to_csv(EVALS / "length_slopes.csv", index=False)

    spearman = spearman_table(run_df)
    spearman.to_csv(EVALS / "spearman_by_run_metric.csv", index=False)

    desc = descriptive_stats(run_df, record_df, pred_df)
    (EVALS / "descriptive_stats.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    (EVALS / "sensitivity_summary.json").write_text(
        json.dumps(sensitivity_summary(run_df), indent=2),
        encoding="utf-8",
    )

    plot_rates(summary)
    plot_slope_heatmap(slopes)


if __name__ == "__main__":
    main()
