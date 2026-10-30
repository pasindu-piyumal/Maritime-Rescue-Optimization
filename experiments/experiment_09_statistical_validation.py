
"""Experiment 09: Statistical validation of Experiment 08 results.

Input:
    results/experiment_08_scenario_results.csv

Outputs:
    results/experiment_09_pairwise_tests.csv
    results/experiment_09_bootstrap_summary.csv
    visualizations/experiment_09_mean_differences.png

Methods are compared on the same simulated scenarios using paired tests.
Results apply to synthetic scenarios and do not validate real-world
maritime rescue performance.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "experiment_08_scenario_results.csv"
RESULTS_DIR = ROOT / "results"
VIS_DIR = ROOT / "visualizations"

BOOTSTRAP_REPS = 10_000
RANDOM_SEED = 42
ALPHA = 0.05

METRICS = {
    "total_travel_time_min": "Total travel time (minutes)",
    "maximum_response_time_min": "Maximum response time (minutes)",
    "critical_mean_response_min": "Critical-incident mean response (minutes)",
}

METHODS = [
    "Minimum Total Travel Time",
    "Minimum Maximum Response Time",
    "Critical-Incident Priority-First",
]


# ============================================================
# HOLM-BONFERRONI MULTIPLE-TEST CORRECTION
# ============================================================

def holm_adjust(p_values):
    """Return Holm-adjusted p-values in their original order."""

    p_values = np.asarray(p_values, dtype=float)

    if p_values.ndim != 1:
        raise ValueError("p_values must be a one-dimensional array.")

    if not np.all(np.isfinite(p_values)):
        raise ValueError("All p-values must be finite.")

    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("P-values must be between 0 and 1.")

    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values)
    adjusted = np.empty(number_of_tests, dtype=float)

    running_max = 0.0

    for rank, original_index in enumerate(order):
        multiplier = number_of_tests - rank
        adjusted_value = min(
            1.0,
            multiplier * p_values[original_index]
        )

        running_max = max(running_max, adjusted_value)
        adjusted[original_index] = running_max

    return adjusted


# ============================================================
# PAIRED BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_mean_ci(
    differences,
    rng,
    reps=BOOTSTRAP_REPS,
):
    """Calculate a percentile bootstrap 95% CI for the mean paired difference."""

    differences = np.asarray(differences, dtype=float)

    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError(
            "At least two paired differences are required."
        )

    if not np.all(np.isfinite(differences)):
        raise ValueError("Paired differences contain non-finite values.")

    # Resample paired differences, not the two methods independently.
    draws = rng.choice(
        differences,
        size=(reps, len(differences)),
        replace=True,
    )

    bootstrap_means = draws.mean(axis=1)

    mean_difference = float(differences.mean())
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    return mean_difference, ci_lower, ci_upper


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Validate files and folders
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run Experiment 08 before running Experiment 09."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "scenario",
        "method",
        *METRICS.keys(),
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep only the three methods under comparison.
    df = df[df["method"].isin(METHODS)].copy()

    if df.empty:
        raise ValueError(
            "No rows found for the configured experiment methods."
        )

    # Each scenario-method pair must identify one observation.
    duplicates = df.duplicated(
        subset=["scenario", "method"],
        keep=False,
    )

    if duplicates.any():
        examples = df.loc[
            duplicates, ["scenario", "method"]
        ].head(10)

        raise ValueError(
            "Duplicate scenario-method rows found. "
            "Resolve them before performing paired comparisons.\n"
            f"{examples.to_string(index=False)}"
        )

    # Ensure metrics contain valid numeric values.
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="raise")

        if not np.isfinite(df[metric].to_numpy(dtype=float)).all():
            raise ValueError(
                f"Metric {metric} contains missing or non-finite values."
            )

    rng = np.random.default_rng(RANDOM_SEED)

    test_rows = []
    bootstrap_rows = []

    # --------------------------------------------------------
    # 2. Run paired comparisons for each metric
    # --------------------------------------------------------

    for metric, metric_label in METRICS.items():

        wide = df.pivot(
            index="scenario",
            columns="method",
            values=metric,
        )

        # Retain scenarios containing all three methods.
        wide = wide.reindex(columns=METHODS).dropna(
            subset=METHODS
        )

        if len(wide) < 2:
            raise ValueError(
                f"Not enough complete paired scenarios for {metric}."
            )

        print(
            f"\nMetric: {metric_label}"
            f"\nComplete paired scenarios: {len(wide)}"
        )

        for method_a, method_b in combinations(METHODS, 2):

            # Positive difference: method A has a worse (higher) time.
            differences = (
                wide[method_a] - wide[method_b]
            ).to_numpy(dtype=float)

            mean_diff, ci_lower, ci_upper = bootstrap_mean_ci(
                differences,
                rng,
            )

            # Wilcoxon signed-rank test: two-sided.
            #
            # If all differences are zero, there is no observed
            # paired difference. Handle this explicitly.
            if np.all(differences == 0):
                statistic = 0.0
                p_value = 1.0
                test_note = "All paired differences are zero"

            else:
                statistic, p_value = wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                test_note = ""

            test_rows.append({
                "metric": metric,
                "method_A": method_a,
                "method_B": method_b,
                "paired_scenarios": len(differences),
                "mean_difference_A_minus_B": mean_diff,
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
                "test_note": test_note,
            })

            bootstrap_rows.append({
                "metric": metric,
                "metric_label": metric_label,
                "method_A": method_a,
                "method_B": method_b,
                "paired_scenarios": len(differences),
                "mean_difference_A_minus_B": mean_diff,
                "bootstrap_95CI_lower": ci_lower,
                "bootstrap_95CI_upper": ci_upper,
            })

    # --------------------------------------------------------
    # 3. Apply Holm correction to all nine comparisons
    # --------------------------------------------------------

    tests = pd.DataFrame(test_rows)

    # There are 3 methods and 3 metrics:
    # 3 pairwise comparisons x 3 metrics = 9 tests.
    tests["holm_adjusted_p_value"] = holm_adjust(
        tests["p_value"].to_numpy()
    )

    tests["significant_at_0.05_after_Holm"] = (
        tests["holm_adjusted_p_value"] < ALPHA
    )

    # Keep full numerical precision in the output CSVs.
    bootstrap = pd.DataFrame(bootstrap_rows)

    # --------------------------------------------------------
    # 4. Save statistical results
    # --------------------------------------------------------

    test_path = (
        RESULTS_DIR / "experiment_09_pairwise_tests.csv"
    )

    bootstrap_path = (
        RESULTS_DIR / "experiment_09_bootstrap_summary.csv"
    )

    tests.to_csv(test_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    # --------------------------------------------------------
    # 5. Plot mean paired differences and 95% bootstrap CIs
    # --------------------------------------------------------

    plot_data = bootstrap.copy()

    # Shorten method names for chart readability.
    short_names = {
        "Minimum Total Travel Time": "Minimum Total",
        "Minimum Maximum Response Time": "Minimum Maximum",
        "Critical-Incident Priority-First": "Priority-First",
    }

    plot_data["comparison"] = (
        plot_data["method_A"].map(short_names)
        + " − "
        + plot_data["method_B"].map(short_names)
    )

    metrics_order = list(METRICS.keys())

    fig, axes = plt.subplots(
        nrows=len(metrics_order),
        ncols=1,
        figsize=(12, 12),
    )

    for ax, metric in zip(axes, metrics_order):

        subset = (
            plot_data[plot_data["metric"] == metric]
            .reset_index(drop=True)
        )

        y_positions = np.arange(len(subset))

        means = subset[
            "mean_difference_A_minus_B"
        ].to_numpy()

        ci_lowers = subset[
            "bootstrap_95CI_lower"
        ].to_numpy()

        ci_uppers = subset[
            "bootstrap_95CI_upper"
        ].to_numpy()

        # Matplotlib requires non-negative error-bar distances.
        lower_errors = np.maximum(means - ci_lowers, 0)
        upper_errors = np.maximum(ci_uppers - means, 0)

        ax.errorbar(
            means,
            y_positions,
            xerr=np.vstack([lower_errors, upper_errors]),
            fmt="o",
            capsize=4,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            subset["comparison"],
            fontsize=9,
        )

        ax.set_title(METRICS[metric])

        ax.set_xlabel(
            "Mean paired difference A − B (minutes); "
            "positive means A has a higher time"
        )

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.3,
        )

    fig.suptitle(
        "Experiment 09: Paired Statistical Comparison "
        "(95% Bootstrap Confidence Intervals)",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    chart_path = (
        VIS_DIR / "experiment_09_mean_differences.png"
    )

    fig.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 6. Print readable results without rounding away p-values
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("EXPERIMENT 09 COMPLETED")
    print("=" * 90)

    print(f"Input file: {INPUT_FILE}")
    print(f"Paired tests CSV: {test_path}")
    print(f"Bootstrap summary CSV: {bootstrap_path}")
    print(f"Confidence interval chart: {chart_path}")

    print("\nPAIRWISE TESTS — HOLM-ADJUSTED RESULTS")

    display_columns = [
        "metric",
        "method_A",
        "method_B",
        "paired_scenarios",
        "mean_difference_A_minus_B",
        "p_value",
        "holm_adjusted_p_value",
        "significant_at_0.05_after_Holm",
    ]

    print(
        tests[display_columns].to_string(
            index=False,
            formatters={
                "mean_difference_A_minus_B": "{:.6f}".format,
                "p_value": "{:.8g}".format,
                "holm_adjusted_p_value": "{:.8g}".format,
            },
        )
    )

    print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")

    ci_columns = [
        "metric",
        "method_A",
        "method_B",
        "paired_scenarios",
        "mean_difference_A_minus_B",
        "bootstrap_95CI_lower",
        "bootstrap_95CI_upper",
    ]

    print(
        bootstrap[ci_columns].to_string(
            index=False,
            formatters={
                "mean_difference_A_minus_B": "{:.6f}".format,
                "bootstrap_95CI_lower": "{:.6f}".format,
                "bootstrap_95CI_upper": "{:.6f}".format,
            },
        )
    )

    print("\nINTERPRETATION NOTES")
    print(
        "1. Positive A-minus-B differences indicate a higher "
        "time metric for method A."
    )
    print(
        "2. Wilcoxon tests are two-sided; Holm correction is "
        "applied across all nine comparisons."
    )
    print(
        "3. Bootstrap intervals are percentile 95% CIs and "
        "are not adjusted for multiple comparisons."
    )
    print(
        "4. Statistical significance does not automatically "
        "imply practical importance."
    )
    print(
        "5. Findings concern synthetic scenarios and do not "
        "validate real-world maritime rescue performance."
    )


if __name__ == "__main__":
    main()
