
"""
Experiment 06: Fairness and Worst-Case Response-Time Analysis

Compares:
    1. Greedy Baseline
    2. Strict Priority-First

Metrics:
    - Incident coverage
    - Total travel time
    - Mean travel time
    - Maximum travel time
    - Standard deviation
    - Response-time range
    - Jain's fairness index

Input:
    results/experiment_01_travel_times.csv

Outputs:
    results/experiment_06_fairness_results.csv
    results/experiment_06_fairness_allocations.csv
    visualizations/experiment_06_fairness_comparison.png

Note:
Travel times and incident priorities are simulated.
"""

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "experiment_01_travel_times.csv"
RESULTS_DIR = ROOT / "results"
VIS_DIR = ROOT / "visualizations"

CAPACITY_PER_STATION = 2

PRIORITIES = {
    "D1": "Critical",
    "D2": "High",
    "D3": "Normal",
    "D4": "Critical",
    "D5": "High",
}

PRIORITY_ORDER = ["Critical", "High", "Normal"]


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


# ==================================================
# 2. LOAD INPUT DATA
# ==================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run Experiment 1 first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "Rescue Station",
        "Distress Location",
        "Travel Time (minutes)",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "Rescue Station",
            "Distress Location",
            "Travel Time (minutes)",
        ]
    ).copy()

    df["Rescue Station"] = df["Rescue Station"].astype(str)
    df["Distress Location"] = df["Distress Location"].astype(str)

    df["Travel Time (minutes)"] = pd.to_numeric(
        df["Travel Time (minutes)"], errors="raise"
    )

    if (df["Travel Time (minutes)"] < 0).any():
        raise ValueError("Travel times cannot be negative.")

    stations = sorted(
        df["Rescue Station"].unique(),
        key=natural_key,
    )

    incidents = sorted(
        df["Distress Location"].unique(),
        key=natural_key,
    )

    missing_priorities = set(incidents) - set(PRIORITIES)

    if missing_priorities:
        raise ValueError(
            "Add priorities for these incidents in PRIORITIES: "
            + ", ".join(sorted(missing_priorities, key=natural_key))
        )

    matrix = df.pivot_table(
        index="Distress Location",
        columns="Rescue Station",
        values="Travel Time (minutes)",
        aggfunc="min",
    ).reindex(index=incidents, columns=stations)

    if matrix.isna().any().any():
        raise ValueError(
            "Some incident-station travel times are missing."
        )

    return stations, incidents, matrix


# ==================================================
# 3. GREEDY BASELINE
# ==================================================

def run_greedy(stations, incidents, matrix):
    remaining = {
        station: CAPACITY_PER_STATION
        for station in stations
    }

    records = []

    for incident in incidents:
        choices = [
            (float(matrix.loc[incident, station]), station)
            for station in stations
            if remaining[station] > 0
        ]

        if not choices:
            continue

        travel_time, station = min(
            choices,
            key=lambda item: (
                item[0], natural_key(item[1])
            ),
        )

        vessel_number = (
            CAPACITY_PER_STATION - remaining[station] + 1
        )

        remaining[station] -= 1

        records.append({
            "method": "Greedy Baseline",
            "distress_location": incident,
            "priority": PRIORITIES[incident],
            "rescue_station": station,
            "vessel_number": vessel_number,
            "travel_time_min": travel_time,
        })

    return records


# ==================================================
# 4. STRICT PRIORITY-FIRST ALLOCATION
# ==================================================

def run_priority_first(stations, incidents, matrix):
    available_slots = [
        (station, vessel_number)
        for station in stations
        for vessel_number in range(
            1, CAPACITY_PER_STATION + 1
        )
    ]

    records = []

    for priority in PRIORITY_ORDER:
        group = [
            incident for incident in incidents
            if PRIORITIES[incident] == priority
        ]

        if not group or not available_slots:
            continue

        costs = [
            [
                float(matrix.loc[incident, station])
                for station, _ in available_slots
            ]
            for incident in group
        ]

        row_indices, col_indices = linear_sum_assignment(costs)

        used_slots = set()

        for row_index, col_index in zip(
            row_indices, col_indices
        ):
            incident = group[row_index]
            station, vessel_number = available_slots[col_index]

            records.append({
                "method": "Strict Priority-First",
                "distress_location": incident,
                "priority": PRIORITIES[incident],
                "rescue_station": station,
                "vessel_number": vessel_number,
                "travel_time_min": float(
                    matrix.loc[incident, station]
                ),
            })

            used_slots.add(col_index)

        available_slots = [
            slot for index, slot in enumerate(available_slots)
            if index not in used_slots
        ]

    return records


# ==================================================
# 5. FAIRNESS METRICS
# ==================================================

def calculate_metrics(group, total_incidents):
    times = group["travel_time_min"].astype(float)

    if times.empty:
        return {
            "assigned_incidents": 0,
            "coverage_percent": 0.0,
            "total_travel_time_min": 0.0,
            "mean_travel_time_min": None,
            "maximum_travel_time_min": None,
            "minimum_travel_time_min": None,
            "response_time_range_min": None,
            "response_time_std_min": None,
            "jains_fairness_index": None,
        }

    # Jain's index is calculated on inverse response times:
    # higher values indicate more balanced response performance.
    # If any response time is zero, use a small positive epsilon.
    epsilon = 1e-9
    service_scores = 1.0 / (times + epsilon)

    numerator = service_scores.sum() ** 2
    denominator = len(service_scores) * (
        service_scores.pow(2).sum()
    )

    jain_index = (
        numerator / denominator
        if denominator > 0
        else None
    )

    return {
        "assigned_incidents": int(
            group["distress_location"].nunique()
        ),
        "coverage_percent": round(
            100 * group["distress_location"].nunique()
            / total_incidents, 2
        ) if total_incidents else 0.0,
        "total_travel_time_min": round(times.sum(), 2),
        "mean_travel_time_min": round(times.mean(), 2),
        "maximum_travel_time_min": round(times.max(), 2),
        "minimum_travel_time_min": round(times.min(), 2),
        "response_time_range_min": round(
            times.max() - times.min(), 2
        ),
        "response_time_std_min": round(
            times.std(ddof=0), 2
        ),
        "jains_fairness_index": round(jain_index, 4),
    }


# ==================================================
# 6. VISUALIZATION
# ==================================================

def create_chart(allocations, incidents):
    # Chart 1: Response time for each incident.
    plot_data = allocations.pivot(
        index="distress_location",
        columns="method",
        values="travel_time_min",
    ).reindex(incidents)

    labels = [
        f"{incident}\n({PRIORITIES[incident]})"
        for incident in incidents
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title(
        "Experiment 06: Response Time by Incident"
    )
    ax.set_xlabel("Distress Incident (Priority)")
    ax.set_ylabel("Travel Time (minutes)")
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(title="Allocation Method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    chart_path = (
        VIS_DIR / "experiment_06_fairness_comparison.png"
    )

    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    return chart_path


# ==================================================
# 7. MAIN
# ==================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    stations, incidents, matrix = load_data()

    greedy_records = run_greedy(
        stations, incidents, matrix
    )

    priority_records = run_priority_first(
        stations, incidents, matrix
    )

    allocations = pd.DataFrame(
        greedy_records + priority_records
    )

    allocations_path = (
        RESULTS_DIR / "experiment_06_fairness_allocations.csv"
    )

    allocations.to_csv(allocations_path, index=False)

    summary_records = []

    for method in [
        "Greedy Baseline",
        "Strict Priority-First",
    ]:
        group = allocations[allocations["method"] == method]

        metrics = calculate_metrics(group, len(incidents))
        metrics["method"] = method

        summary_records.append(metrics)

    summary = pd.DataFrame(summary_records)

    summary_path = (
        RESULTS_DIR / "experiment_06_fairness_results.csv"
    )

    summary.to_csv(summary_path, index=False)

    chart_path = create_chart(allocations, incidents)

    print("\n" + "=" * 70)
    print("EXPERIMENT 06 COMPLETED")
    print("=" * 70)

    print("\nFairness and response-time results:")
    print(summary.to_string(index=False))

    print("\nGenerated files:")
    print(f"Allocations: {allocations_path}")
    print(f"Summary:     {summary_path}")
    print(f"Chart:       {chart_path}")

    print(
        "\nInterpretation note: Jain's fairness index here is based "
        "on inverse travel times. It measures balance, not absolute "
        "speed or priority correctness. Lower travel time is better."
    )


if __name__ == "__main__":
    main()
