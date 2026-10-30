
"""
Experiment 05: Weather and Sea-Condition Sensitivity Analysis

Uses simulated weather multipliers on the original travel times.

Conditions:
    Normal:   +0%
    Moderate: +25%
    Severe:   +60%

Compares:
    1. Greedy Baseline
    2. Strict Priority-First

Input:
    results/experiment_01_travel_times.csv

Outputs:
    results/experiment_05_weather_results.csv
    results/experiment_05_weather_allocations.csv
    visualizations/experiment_05_weather_comparison.png

Important:
Weather multipliers are illustrative assumptions, not measured
meteorological data or validated maritime travel-time predictions.
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

# Simulated incident priorities
PRIORITIES = {
    "D1": "Critical",
    "D2": "High",
    "D3": "Normal",
    "D4": "Critical",
    "D5": "High",
}

PRIORITY_ORDER = ["Critical", "High", "Normal"]

# Illustrative travel-time multipliers
WEATHER_CONDITIONS = {
    "Normal": 1.00,
    "Moderate": 1.25,
    "Severe": 1.60,
}


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


# ==================================================
# 2. LOAD DATA
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
        df["Rescue Station"].unique(), key=natural_key
    )
    incidents = sorted(
        df["Distress Location"].unique(), key=natural_key
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
# 3. GREEDY ALLOCATION
# ==================================================

def run_greedy(stations, incidents, matrix):
    remaining = {
        station: CAPACITY_PER_STATION
        for station in stations
    }

    records = []

    for incident in incidents:
        available = [
            (float(matrix.loc[incident, station]), station)
            for station in stations
            if remaining[station] > 0
        ]

        if not available:
            continue

        travel_time, station = min(
            available,
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

        cost_matrix = [
            [
                float(matrix.loc[incident, station])
                for station, _ in available_slots
            ]
            for incident in group
        ]

        row_indices, col_indices = linear_sum_assignment(
            cost_matrix
        )

        used_slots = set()

        for row_index, col_index in zip(
            row_indices, col_indices
        ):
            incident = group[row_index]
            station, vessel_number = available_slots[col_index]

            records.append({
                "method": "Strict Priority-First",
                "distress_location": incident,
                "priority": priority,
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
# 5. RUN WEATHER SCENARIOS
# ==================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    stations, incidents, original_matrix = load_data()

    all_allocations = []
    summary_rows = []

    print("\nExperiment 05: Weather Sensitivity Analysis")
    print("Stations:", stations)
    print("Incidents:", incidents)

    for condition, multiplier in WEATHER_CONDITIONS.items():
        print(
            f"\nScenario: {condition} "
            f"(travel-time multiplier = {multiplier:.2f})"
        )

        # Apply the same weather multiplier to every route.
        scenario_matrix = original_matrix * multiplier

        greedy = run_greedy(
            stations, incidents, scenario_matrix
        )

        priority = run_priority_first(
            stations, incidents, scenario_matrix
        )

        for record in greedy + priority:
            record["weather_condition"] = condition
            record["weather_multiplier"] = multiplier

        scenario_allocations = greedy + priority
        all_allocations.extend(scenario_allocations)

        for method in [
            "Greedy Baseline",
            "Strict Priority-First",
        ]:
            group = [
                row for row in scenario_allocations
                if row["method"] == method
            ]

            assigned = len({
                row["distress_location"] for row in group
            })

            times = [
                row["travel_time_min"] for row in group
            ]

            critical_times = [
                row["travel_time_min"] for row in group
                if row["priority"] == "Critical"
            ]

            summary_rows.append({
                "weather_condition": condition,
                "weather_multiplier": multiplier,
                "method": method,
                "assigned_incidents": assigned,
                "total_incidents": len(incidents),
                "coverage_percent": round(
                    100 * assigned / len(incidents), 2
                ) if incidents else 0,
                "total_travel_time_min": round(
                    sum(times), 2
                ),
                "mean_travel_time_min": round(
                    sum(times) / len(times), 2
                ) if times else None,
                "maximum_travel_time_min": round(
                    max(times), 2
                ) if times else None,
                "critical_mean_travel_time_min": round(
                    sum(critical_times) / len(critical_times), 2
                ) if critical_times else None,
            })

    # ==================================================
    # 6. SAVE CSV OUTPUTS
    # ==================================================

    allocations_df = pd.DataFrame(all_allocations)
    summary_df = pd.DataFrame(summary_rows)

    allocation_path = (
        RESULTS_DIR / "experiment_05_weather_allocations.csv"
    )

    summary_path = (
        RESULTS_DIR / "experiment_05_weather_results.csv"
    )

    allocations_df.to_csv(allocation_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # ==================================================
    # 7. CREATE COMPARISON CHART
    # ==================================================

    chart_data = summary_df.pivot(
        index="weather_condition",
        columns="method",
        values="total_travel_time_min",
    ).reindex(list(WEATHER_CONDITIONS.keys()))

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Experiment 05: Weather Impact on Rescue Travel Time"
    )
    ax.set_xlabel("Simulated Weather Condition")
    ax.set_ylabel("Total Assigned Travel Time (minutes)")
    ax.set_xticklabels(
        chart_data.index, rotation=0
    )
    ax.legend(title="Allocation Method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    chart_path = (
        VIS_DIR / "experiment_05_weather_comparison.png"
    )

    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ==================================================
    # 8. PRINT RESULTS
    # ==================================================

    print("\n" + "=" * 70)
    print("EXPERIMENT 05 COMPLETED")
    print("=" * 70)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nGenerated files:")
    print(f"Allocations: {allocation_path}")
    print(f"Summary:     {summary_path}")
    print(f"Chart:       {chart_path}")

    print(
        "\nResearch limitation: weather multipliers are simulated "
        "assumptions, not empirical maritime measurements."
    )


if __name__ == "__main__":
    main()
