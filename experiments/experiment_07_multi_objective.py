
"""
Experiment 07: Multi-Objective Maritime Rescue Allocation

Methods:
1. Minimum Total Travel Time
2. Minimum Maximum Response Time
3. Critical-Incident Priority-First

Input:
    results/experiment_01_travel_times.csv

Outputs:
    results/experiment_07_allocations.csv
    results/experiment_07_comparison.csv
    visualizations/experiment_07_objective_comparison.png
    visualizations/experiment_07_incident_comparison.png

All incident priorities and travel times are simulated.
"""

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. CONFIGURATION
# ============================================================

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


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Make sure Experiment 01 has been completed."
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
            f"Missing CSV columns: {sorted(missing)}"
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
            "Add priorities in PRIORITIES for: "
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


def make_slots(stations):
    """Represent each station's vessel capacity as individual slots."""
    return [
        (station, vessel_number)
        for station in stations
        for vessel_number in range(1, CAPACITY_PER_STATION + 1)
    ]


def create_record(method, incident, station, vessel, matrix):
    return {
        "method": method,
        "distress_location": incident,
        "priority": PRIORITIES[incident],
        "rescue_station": station,
        "vessel_number": vessel,
        "travel_time_min": float(matrix.loc[incident, station]),
    }


# ============================================================
# 3. METHOD 1: MINIMUM TOTAL TRAVEL TIME
# ============================================================

def minimum_total_time(stations, incidents, matrix):
    """
    Hungarian assignment minimizes the sum of travel times.
    Each vessel slot can serve at most one incident.
    """
    slots = make_slots(stations)

    costs = [
        [float(matrix.loc[incident, station])
         for station, _ in slots]
        for incident in incidents
    ]

    row_indices, col_indices = linear_sum_assignment(costs)

    records = []

    for row, col in zip(row_indices, col_indices):
        incident = incidents[row]
        station, vessel = slots[col]

        records.append(
            create_record(
                "Minimum Total Travel Time",
                incident,
                station,
                vessel,
                matrix,
            )
        )

    return records


# ============================================================
# 4. METHOD 2: MINIMUM MAXIMUM RESPONSE TIME
# ============================================================

def can_assign_with_limit(incidents, slots, matrix, time_limit):
    """
    Check whether every incident can be assigned to a distinct
    vessel slot with travel time <= time_limit.

    Uses a bipartite matching search.
    """

    # A slot can potentially serve an incident if its station's
    # travel time is within the response-time threshold.
    slot_to_incident = {}

    def augment(incident, visited_slots):
        eligible_slots = sorted(
            [
                index
                for index, (station, _) in enumerate(slots)
                if float(matrix.loc[incident, station]) <= time_limit
            ],
            key=lambda index: (
                float(matrix.loc[incident, slots[index][0]]),
                natural_key(slots[index][0]),
                slots[index][1],
            ),
        )

        for slot_index in eligible_slots:
            if slot_index in visited_slots:
                continue

            visited_slots.add(slot_index)

            if (
                slot_index not in slot_to_incident
                or augment(
                    slot_to_incident[slot_index],
                    visited_slots,
                )
            ):
                slot_to_incident[slot_index] = incident
                return True

        return False

    for incident in sorted(
        incidents,
        key=lambda item: sum(
            float(matrix.loc[item, station]) <= time_limit
            for station, _ in slots
        ),
    ):
        if not augment(incident, set()):
            return False

    return True


def minimum_maximum_time(stations, incidents, matrix):
    """
    First find the smallest feasible maximum response time.
    Then minimize total travel time while keeping every assignment
    within that optimal maximum-time threshold.
    """
    slots = make_slots(stations)

    if len(incidents) > len(slots):
        raise ValueError(
            "There are more incidents than available vessel slots. "
            "Full coverage is impossible with the configured capacity."
        )

    possible_times = sorted({
        float(matrix.loc[incident, station])
        for incident in incidents
        for station in stations
    })

    low = 0
    high = len(possible_times) - 1

    while low < high:
        middle = (low + high) // 2
        threshold = possible_times[middle]

        if can_assign_with_limit(
            incidents, slots, matrix, threshold
        ):
            high = middle
        else:
            low = middle + 1

    optimal_maximum = possible_times[low]

    # Keep only assignments that satisfy the minimum maximum time.
    # A large penalty prevents the solver from choosing an invalid
    # assignment when a feasible full assignment exists.
    big_m = max(possible_times) * (len(incidents) + 2) + 1

    costs = []

    for incident in incidents:
        row = []

        for station, _ in slots:
            travel_time = float(matrix.loc[incident, station])

            if travel_time <= optimal_maximum:
                row.append(travel_time)
            else:
                row.append(big_m)

        costs.append(row)

    row_indices, col_indices = linear_sum_assignment(costs)

    records = []

    for row, col in zip(row_indices, col_indices):
        incident = incidents[row]
        station, vessel = slots[col]
        travel_time = float(matrix.loc[incident, station])

        if travel_time > optimal_maximum:
            raise RuntimeError(
                "Could not construct a valid minimum-maximum assignment."
            )

        records.append(
            create_record(
                "Minimum Maximum Response Time",
                incident,
                station,
                vessel,
                matrix,
            )
        )

    return records


# ============================================================
# 5. METHOD 3: CRITICAL-INCIDENT PRIORITY-FIRST
# ============================================================

def priority_first(stations, incidents, matrix):
    """
    Assign Critical incidents first, then High, then Normal.
    Within each priority level, minimize total travel time.
    """
    available_slots = make_slots(stations)
    records = []

    for priority in PRIORITY_ORDER:
        group = [
            incident for incident in incidents
            if PRIORITIES[incident] == priority
        ]

        if not group:
            continue

        if not available_slots:
            break

        costs = [
            [
                float(matrix.loc[incident, station])
                for station, _ in available_slots
            ]
            for incident in group
        ]

        row_indices, col_indices = linear_sum_assignment(costs)

        used_indices = set()

        for row, col in zip(row_indices, col_indices):
            incident = group[row]
            station, vessel = available_slots[col]

            records.append(
                create_record(
                    "Critical-Incident Priority-First",
                    incident,
                    station,
                    vessel,
                    matrix,
                )
            )

            used_indices.add(col)

        available_slots = [
            slot for index, slot in enumerate(available_slots)
            if index not in used_indices
        ]

    return records


# ============================================================
# 6. CALCULATE COMPARISON METRICS
# ============================================================

def calculate_summary(group, total_incidents):
    times = group["travel_time_min"].astype(float)

    assigned = group["distress_location"].nunique()

    critical = group[group["priority"] == "Critical"]
    high = group[group["priority"] == "High"]
    normal = group[group["priority"] == "Normal"]

    return {
        "assigned_incidents": int(assigned),
        "total_incidents": int(total_incidents),
        "coverage_percent": round(
            100 * assigned / total_incidents, 2
        ) if total_incidents else 0.0,
        "total_travel_time_min": round(times.sum(), 2),
        "mean_travel_time_min": round(times.mean(), 2)
        if len(times) else None,
        "maximum_travel_time_min": round(times.max(), 2)
        if len(times) else None,
        "response_time_std_min": round(
            times.std(ddof=0), 2
        ) if len(times) else None,
        "critical_mean_time_min": round(
            critical["travel_time_min"].mean(), 2
        ) if len(critical) else None,
        "high_mean_time_min": round(
            high["travel_time_min"].mean(), 2
        ) if len(high) else None,
        "normal_mean_time_min": round(
            normal["travel_time_min"].mean(), 2
        ) if len(normal) else None,
    }


# ============================================================
# 7. CREATE CHARTS
# ============================================================

def create_charts(allocations, summary, incidents):
    method_order = [
        "Minimum Total Travel Time",
        "Minimum Maximum Response Time",
        "Critical-Incident Priority-First",
    ]

    # Chart A: compare aggregate objectives.
    chart_summary = summary.set_index("method").reindex(method_order)

    ax = chart_summary[
        ["total_travel_time_min", "maximum_travel_time_min"]
    ].plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title("Experiment 07: Multi-Objective Allocation Comparison")
    ax.set_xlabel("Allocation Objective")
    ax.set_ylabel("Travel Time (minutes)")
    ax.set_xticklabels(method_order, rotation=12, ha="right")
    ax.legend([
        "Total Travel Time",
        "Maximum Response Time",
    ], title="Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    objective_chart = (
        VIS_DIR / "experiment_07_objective_comparison.png"
    )

    plt.savefig(objective_chart, dpi=200, bbox_inches="tight")
    plt.close()

    # Chart B: response time for each incident and method.
    plot_data = allocations.pivot(
        index="distress_location",
        columns="method",
        values="travel_time_min",
    ).reindex(index=incidents, columns=method_order)

    ax = plot_data.plot(
        kind="bar",
        figsize=(12, 6),
    )

    labels = [
        f"{incident}\n({PRIORITIES[incident]})"
        for incident in incidents
    ]

    ax.set_title("Experiment 07: Response Time by Incident")
    ax.set_xlabel("Distress Incident (Priority)")
    ax.set_ylabel("Travel Time (minutes)")
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(title="Allocation Method", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    incident_chart = (
        VIS_DIR / "experiment_07_incident_comparison.png"
    )

    plt.savefig(incident_chart, dpi=200, bbox_inches="tight")
    plt.close()

    return objective_chart, incident_chart


# ============================================================
# 8. MAIN
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    stations, incidents, matrix = load_data()

    if len(incidents) > len(make_slots(stations)):
        raise ValueError(
            f"{len(incidents)} incidents but only "
            f"{len(make_slots(stations))} vessel slots. "
            "Increase capacity or use a partial-coverage model."
        )

    print("\nRunning Experiment 07...")
    print("Rescue stations:", stations)
    print("Distress incidents:", incidents)
    print("Capacity per station:", CAPACITY_PER_STATION)

    all_records = []

    all_records.extend(
        minimum_total_time(stations, incidents, matrix)
    )

    all_records.extend(
        minimum_maximum_time(stations, incidents, matrix)
    )

    all_records.extend(
        priority_first(stations, incidents, matrix)
    )

    allocations = pd.DataFrame(all_records)

    allocations_path = (
        RESULTS_DIR / "experiment_07_allocations.csv"
    )

    allocations.to_csv(allocations_path, index=False)

    summary_records = []

    for method in [
        "Minimum Total Travel Time",
        "Minimum Maximum Response Time",
        "Critical-Incident Priority-First",
    ]:
        group = allocations[allocations["method"] == method]

        metrics = calculate_summary(group, len(incidents))
        metrics["method"] = method
        summary_records.append(metrics)

    summary = pd.DataFrame(summary_records)

    # Keep a readable column order.
    column_order = [
        "method",
        "assigned_incidents",
        "total_incidents",
        "coverage_percent",
        "total_travel_time_min",
        "mean_travel_time_min",
        "maximum_travel_time_min",
        "response_time_std_min",
        "critical_mean_time_min",
        "high_mean_time_min",
        "normal_mean_time_min",
    ]

    summary = summary[column_order]

    summary_path = (
        RESULTS_DIR / "experiment_07_comparison.csv"
    )

    summary.to_csv(summary_path, index=False)

    objective_chart, incident_chart = create_charts(
        allocations, summary, incidents
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 07 COMPLETED")
    print("=" * 75)

    print("\nObjective comparison:")
    print(summary.to_string(index=False))

    print("\nAssignments:")
    print(
        allocations.sort_values(
            ["method", "distress_location"],
            key=lambda series: series.map(
                lambda value: (
                    natural_key(value)
                    if series.name == "distress_location"
                    else value
                )
            ) if False else series
        ).to_string(index=False)
    )

    print("\nGenerated files:")
    print(f"Allocations: {allocations_path}")
    print(f"Comparison:  {summary_path}")
    print(f"Objective chart: {objective_chart}")
    print(f"Incident chart:  {incident_chart}")

    print(
        "\nResearch note: travel times and priorities are simulated. "
        "The results describe this model, not validated real-world "
        "maritime rescue performance."
    )


if __name__ == "__main__":
    main()
