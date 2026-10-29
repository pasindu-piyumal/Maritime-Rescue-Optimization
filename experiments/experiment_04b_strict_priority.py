
"""
Experiment 04B: Strict Priority-First Maritime Rescue Allocation

Priority order:
    1. Critical
    2. High
    3. Normal

Within each priority level, use minimum-cost assignment
to minimize total travel time while respecting station capacity.

Input:
    results/experiment_01_travel_times.csv

Outputs:
    results/experiment_04b_allocations.csv
    results/experiment_04b_comparison.csv
    visualizations/experiment_04b_comparison.png

Priority labels are simulated assumptions.
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
# Change these labels to match your experimental design.
PRIORITIES = {
    "D1": "Critical",
    "D2": "High",
    "D3": "Normal",
    "D4": "Critical",
    "D5": "High",
}

PRIORITY_ORDER = ["Critical", "High", "Normal"]


def natural_key(value):
    """Sort identifiers naturally, e.g. D2 before D10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
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
            f"Found columns: {list(df.columns)}"
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
            "Define priorities for these incidents in PRIORITIES: "
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
    """Assign incidents in ID order to the nearest available station."""

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
            key=lambda x: (x[0], natural_key(x[1])),
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
# 4. STRICT PRIORITY-FIRST OPTIMIZATION
# ==================================================

def run_priority_first(stations, incidents, matrix):
    """
    Process Critical, High, then Normal incidents.

    For each priority group, use the Hungarian algorithm
    to minimize the group's total travel time using only
    the remaining station capacity slots.

    Higher-priority groups reserve their assignments first.
    """

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
            incident
            for incident in incidents
            if PRIORITIES[incident] == priority
        ]

        if not group or not available_slots:
            continue

        # Rows = incidents in this priority group.
        # Columns = currently available vessel slots.
        costs = [
            [
                float(matrix.loc[incident, station])
                for station, vessel_number in available_slots
            ]
            for incident in group
        ]

        row_indices, col_indices = linear_sum_assignment(costs)

        assigned_incidents = set()
        used_slot_indices = set()

        for row_index, col_index in zip(
            row_indices, col_indices
        ):
            incident = group[row_index]
            station, vessel_number = available_slots[col_index]

            travel_time = float(matrix.loc[incident, station])

            records.append({
                "method": "Strict Priority-First",
                "distress_location": incident,
                "priority": priority,
                "rescue_station": station,
                "vessel_number": vessel_number,
                "travel_time_min": travel_time,
            })

            assigned_incidents.add(incident)
            used_slot_indices.add(col_index)

        # Remove slots used by this priority group.
        available_slots = [
            slot
            for index, slot in enumerate(available_slots)
            if index not in used_slot_indices
        ]

        unassigned = set(group) - assigned_incidents

        if unassigned:
            print(
                f"Warning: no available capacity for "
                f"{priority} incidents: "
                f"{sorted(unassigned, key=natural_key)}"
            )

    return records


# ==================================================
# 5. SUMMARIZE RESULTS
# ==================================================

def summarize(allocations, incidents):
    summaries = []

    for method, group in allocations.groupby(
        "method", sort=False
    ):
        critical = group[group["priority"] == "Critical"]

        summaries.append({
            "method": method,
            "assigned_incidents": (
                group["distress_location"].nunique()
            ),
            "total_incidents": len(incidents),
            "coverage_percent": round(
                100 * group["distress_location"].nunique()
                / len(incidents), 2
            ) if incidents else 0,
            "total_travel_time_min": round(
                group["travel_time_min"].sum(), 2
            ),
            "mean_travel_time_min": round(
                group["travel_time_min"].mean(), 2
            ) if len(group) else None,
            "maximum_travel_time_min": round(
                group["travel_time_min"].max(), 2
            ) if len(group) else None,
            "critical_incidents_assigned": (
                critical["distress_location"].nunique()
            ),
            "critical_mean_travel_time_min": round(
                critical["travel_time_min"].mean(), 2
            ) if len(critical) else None,
        })

    return pd.DataFrame(summaries)


# ==================================================
# 6. PLOT COMPARISON
# ==================================================

def create_plot(allocations, incidents):
    plot_data = allocations.pivot(
        index="distress_location",
        columns="method",
        values="travel_time_min",
    ).reindex(incidents)

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    labels = [
        f"{incident}\n({PRIORITIES[incident]})"
        for incident in incidents
    ]

    ax.set_title(
        "Experiment 04B: Strict Priority-First Allocation"
    )
    ax.set_xlabel("Distress Incident (Priority)")
    ax.set_ylabel("Travel Time (minutes)")
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(title="Allocation Method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    chart_path = (
        VIS_DIR / "experiment_04b_comparison.png"
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

    print("\nStations:", stations)
    print("Incidents:", incidents)
    print("Capacity per station:", CAPACITY_PER_STATION)

    greedy_records = run_greedy(
        stations, incidents, matrix
    )

    priority_records = run_priority_first(
        stations, incidents, matrix
    )

    allocations = pd.DataFrame(
        greedy_records + priority_records
    )

    allocation_path = (
        RESULTS_DIR / "experiment_04b_allocations.csv"
    )

    allocations.to_csv(allocation_path, index=False)

    comparison = summarize(allocations, incidents)

    comparison_path = (
        RESULTS_DIR / "experiment_04b_comparison.csv"
    )

    comparison.to_csv(comparison_path, index=False)

    chart_path = create_plot(allocations, incidents)

    print("\n" + "=" * 65)
    print("EXPERIMENT 04B COMPLETED")
    print("=" * 65)

    print("\nComparison results:")
    print(comparison.to_string(index=False))

    print("\nAllocation details:")
    print(
        allocations.sort_values(
            ["method", "priority", "distress_location"]
        ).to_string(index=False)
    )

    print("\nGenerated files:")
    print(f"Allocations: {allocation_path}")
    print(f"Comparison:  {comparison_path}")
    print(f"Chart:       {chart_path}")

    print(
        "\nResearch note: priorities and travel times are simulated. "
        "This method processes each priority group in order and "
        "minimizes travel time within each group."
    )


if __name__ == "__main__":
    main()
