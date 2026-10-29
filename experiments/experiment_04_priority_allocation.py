from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "experiment_01_travel_times.csv"
RESULTS_DIR = ROOT / "results"
VIS_DIR = ROOT / "visualizations"

CAPACITY_PER_STATION = 2

# Simulated incident priorities:
# Critical = 3, High = 2, Normal = 1
PRIORITIES = {
    "D1": ("Critical", 3),
    "D2": ("High", 2),
    "D3": ("Normal", 1),
    "D4": ("Critical", 3),
    "D5": ("High", 2),
}


def natural_key(value):
    """Sort identifiers naturally: R1, R2, R10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def main():
    # --------------------------------------------------
    # 2. LOAD TRAVEL-TIME DATA
    # --------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Check that Experiment 1 has generated "
            "results/experiment_01_travel_times.csv."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "Rescue Station",
        "Distress Location",
        "Travel Time (minutes)",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}\n"
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

    stations = sorted(
        df["Rescue Station"].unique(), key=natural_key
    )

    incidents = sorted(
        df["Distress Location"].unique(), key=natural_key
    )

    missing_priorities = set(incidents) - set(PRIORITIES)

    if missing_priorities:
        raise ValueError(
            "Add priority settings for these incidents in PRIORITIES: "
            + ", ".join(sorted(missing_priorities, key=natural_key))
        )

    # Rows = incidents, columns = rescue stations
    time_matrix = df.pivot_table(
        index="Distress Location",
        columns="Rescue Station",
        values="Travel Time (minutes)",
        aggfunc="min",
    ).reindex(index=incidents, columns=stations)

    if time_matrix.isna().any().any():
        raise ValueError(
            "Some incident-station travel times are missing."
        )

    print("\nStations:", stations)
    print("Incidents:", incidents)
    print("Capacity per station:", CAPACITY_PER_STATION)

    # --------------------------------------------------
    # 3. ALLOCATION RECORD HELPER
    # --------------------------------------------------

    def make_record(
        method, incident, station, travel_time, vessel_number
    ):
        priority, weight = PRIORITIES[incident]

        return {
            "method": method,
            "distress_location": incident,
            "priority": priority,
            "priority_weight": weight,
            "rescue_station": station,
            "vessel_number": vessel_number,
            "travel_time_min": float(travel_time),
        }

    # --------------------------------------------------
    # 4. METHOD A: GREEDY BASELINE
    # --------------------------------------------------
    # Process incidents in identifier order.
    # Assign each to the nearest station with spare capacity.

    greedy_name = "Greedy Baseline"

    remaining_capacity = {
        station: CAPACITY_PER_STATION
        for station in stations
    }

    greedy_records = []

    for incident in incidents:
        available = [
            (
                float(time_matrix.loc[incident, station]),
                station,
            )
            for station in stations
            if remaining_capacity[station] > 0
        ]

        if not available:
            print(f"No available vessel for {incident}")
            continue

        travel_time, station = min(
            available,
            key=lambda item: (
                item[0],
                natural_key(item[1]),
            ),
        )

        vessel_number = (
            CAPACITY_PER_STATION
            - remaining_capacity[station]
            + 1
        )

        remaining_capacity[station] -= 1

        greedy_records.append(
            make_record(
                greedy_name,
                incident,
                station,
                travel_time,
                vessel_number,
            )
        )

    # --------------------------------------------------
    # 5. METHOD B: PRIORITY-AWARE OPTIMIZATION
    # --------------------------------------------------
    # Represent each vessel capacity as a separate slot.
    # Minimize travel_time / priority_weight.
    #
    # This makes shorter travel times for Critical
    # incidents more influential in the objective.

    optimization_name = "Priority-Aware Optimization"

    vessel_slots = [
        (station, vessel_number)
        for station in stations
        for vessel_number in range(
            1, CAPACITY_PER_STATION + 1
        )
    ]

    cost_matrix = []

    for incident in incidents:
        _, priority_weight = PRIORITIES[incident]

        row = [
            float(time_matrix.loc[incident, station])
            / priority_weight
            for station, vessel_number in vessel_slots
        ]

        cost_matrix.append(row)

    optimized_records = []

    if incidents and vessel_slots:
        row_indices, column_indices = linear_sum_assignment(
            cost_matrix
        )

        for row_index, column_index in zip(
            row_indices, column_indices
        ):
            incident = incidents[row_index]

            station, vessel_number = vessel_slots[
                column_index
            ]

            travel_time = float(
                time_matrix.loc[incident, station]
            )

            optimized_records.append(
                make_record(
                    optimization_name,
                    incident,
                    station,
                    travel_time,
                    vessel_number,
                )
            )

    # --------------------------------------------------
    # 6. SAVE ALLOCATION DETAILS
    # --------------------------------------------------

    allocations = pd.DataFrame(
        greedy_records + optimized_records
    )

    allocation_file = (
        RESULTS_DIR / "experiment_04_priority_allocations.csv"
    )

    allocations.to_csv(allocation_file, index=False)

    # --------------------------------------------------
    # 7. CALCULATE COMPARISON METRICS
    # --------------------------------------------------

    summary_records = []

    for method in [greedy_name, optimization_name]:
        group = allocations[allocations["method"] == method]

        assigned_incidents = group[
            "distress_location"
        ].nunique()

        critical = group[group["priority"] == "Critical"]

        summary_records.append({
            "method": method,
            "assigned_incidents": assigned_incidents,
            "total_incidents": len(incidents),
            "coverage_percent": round(
                100 * assigned_incidents / len(incidents), 2
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
            "critical_incidents_assigned": critical[
                "distress_location"
            ].nunique(),
            "critical_mean_travel_time_min": round(
                critical["travel_time_min"].mean(), 2
            ) if len(critical) else None,
            "priority_weighted_cost": round(
                (
                    group["travel_time_min"]
                    / group["priority_weight"]
                ).sum(),
                2,
            ),
        })

    comparison = pd.DataFrame(summary_records)

    comparison_file = (
        RESULTS_DIR / "experiment_04_priority_comparison.csv"
    )

    comparison.to_csv(comparison_file, index=False)

    # --------------------------------------------------
    # 8. VISUALIZE RESPONSE TIMES BY INCIDENT
    # --------------------------------------------------

    plot_data = allocations.pivot(
        index="distress_location",
        columns="method",
        values="travel_time_min",
    ).reindex(incidents)

    # Include priority level in each incident label.
    labels = [
        f"{incident}\n({PRIORITIES[incident][0]})"
        for incident in incidents
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title(
        "Experiment 04: Priority-Aware Rescue Allocation"
    )
    ax.set_xlabel("Distress Incident and Priority")
    ax.set_ylabel("Travel Time (minutes)")
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(title="Allocation Method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()

    chart_file = (
        VIS_DIR / "experiment_04_priority_comparison.png"
    )

    plt.savefig(chart_file, dpi=200, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------
    # 9. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT 04 COMPLETED")
    print("=" * 65)

    print("\nComparison Results:")
    print(comparison.to_string(index=False))

    print("\nGenerated files:")
    print(f"Allocations: {allocation_file}")
    print(f"Comparison:  {comparison_file}")
    print(f"Chart:       {chart_file}")

    print(
        "\nNOTE: Incident priorities are simulated assumptions, "
        "not real emergency records."
    )


if __name__ == "__main__":
    main()
