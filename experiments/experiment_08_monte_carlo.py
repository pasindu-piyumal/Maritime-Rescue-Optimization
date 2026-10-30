
"""
Experiment 08: Monte Carlo Evaluation of Maritime Rescue Allocation

Uses the existing Experiment 01 travel-time CSV.
All simulations are synthetic and reproducible.

Methods:
1. Minimum Total Travel Time
2. Minimum Maximum Response Time
3. Critical-Incident Priority-First

Outputs:
    results/experiment_08_scenario_results.csv
    results/experiment_08_summary.csv
    visualizations/experiment_08_total_time.png
    visualizations/experiment_08_coverage.png
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "experiment_01_travel_times.csv"
RESULTS_DIR = ROOT / "results"
VIS_DIR = ROOT / "visualizations"

N_SCENARIOS = 500
RANDOM_SEED = 42

BASE_CAPACITY_PER_STATION = 2
MIN_CAPACITY = 1
MAX_CAPACITY = 3

# Random travel-time variation: 0.75 to 1.50 times baseline.
TRAVEL_TIME_MIN_FACTOR = 0.75
TRAVEL_TIME_MAX_FACTOR = 1.50

PRIORITIES = {
    "D1": "Critical",
    "D2": "High",
    "D3": "Normal",
    "D4": "Critical",
    "D5": "High",
}

PRIORITY_ORDER = ["Critical", "High", "Normal"]

METHODS = [
    "Minimum Total Travel Time",
    "Minimum Maximum Response Time",
    "Critical-Incident Priority-First",
]


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


# ------------------------------------------------------------
# LOAD BASELINE
# ------------------------------------------------------------

def load_baseline():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}\n"
            "Run Experiment 01 first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "Rescue Station",
        "Distress Location",
        "Travel Time (minutes)",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    df["Rescue Station"] = df["Rescue Station"].astype(str)
    df["Distress Location"] = df["Distress Location"].astype(str)
    df["Travel Time (minutes)"] = pd.to_numeric(
        df["Travel Time (minutes)"], errors="raise"
    )

    if (df["Travel Time (minutes)"] <= 0).any():
        raise ValueError("Travel times must be positive.")

    stations = sorted(df["Rescue Station"].unique(), key=natural_key)
    incidents = sorted(df["Distress Location"].unique(), key=natural_key)

    missing_priorities = set(incidents) - set(PRIORITIES)
    if missing_priorities:
        raise ValueError(
            "Add priorities for: "
            + ", ".join(sorted(missing_priorities, key=natural_key))
        )

    matrix = df.pivot_table(
        index="Distress Location",
        columns="Rescue Station",
        values="Travel Time (minutes)",
        aggfunc="min",
    ).reindex(index=incidents, columns=stations)

    if matrix.isna().any().any():
        raise ValueError("Missing station-to-incident travel times.")

    return stations, incidents, matrix.astype(float)


# ------------------------------------------------------------
# SCENARIO GENERATION
# ------------------------------------------------------------

def generate_scenario(base_matrix, stations, incidents, rng):
    """
    Apply a random travel-time multiplier to each station-incident
    pair and vary vessel capacity by station.

    Capacity is kept sufficient to assign every incident, allowing
    fair comparison of response times across all three methods.
    """
    multipliers = rng.uniform(
        TRAVEL_TIME_MIN_FACTOR,
        TRAVEL_TIME_MAX_FACTOR,
        size=base_matrix.shape,
    )

    scenario_matrix = base_matrix.copy() * multipliers

    capacities = {
        station: int(
            rng.integers(MIN_CAPACITY, MAX_CAPACITY + 1)
        )
        for station in stations
    }

    # Ensure enough total vessels for all incidents.
    while sum(capacities.values()) < len(incidents):
        station = stations[int(rng.integers(0, len(stations)))]
        if capacities[station] < MAX_CAPACITY:
            capacities[station] += 1

        # If all stations have reached the maximum, stop safely.
        if all(capacities[s] == MAX_CAPACITY for s in stations):
            break

    return scenario_matrix, capacities


def create_slots(stations, capacities):
    return [
        (station, vessel_number)
        for station in stations
        for vessel_number in range(1, capacities[station] + 1)
    ]


# ------------------------------------------------------------
# METHOD 1: MINIMUM TOTAL TRAVEL TIME
# ------------------------------------------------------------

def minimum_total_assignment(matrix, incidents, slots):
    costs = [
        [float(matrix.loc[incident, station]) for station, _ in slots]
        for incident in incidents
    ]

    rows, cols = linear_sum_assignment(costs)

    return {
        incidents[row]: (
            slots[col][0],
            slots[col][1],
            float(matrix.loc[incidents[row], slots[col][0]]),
        )
        for row, col in zip(rows, cols)
    }


# ------------------------------------------------------------
# METHOD 2: MINIMUM MAXIMUM RESPONSE TIME
# ------------------------------------------------------------

def has_full_matching(matrix, incidents, slots, threshold):
    """Check whether all incidents can be assigned under a time limit."""
    slot_to_incident = {}

    def try_match(incident, visited):
        eligible = [
            idx for idx, (station, _) in enumerate(slots)
            if float(matrix.loc[incident, station]) <= threshold
        ]

        eligible.sort(
            key=lambda idx: float(
                matrix.loc[incident, slots[idx][0]]
            )
        )

        for idx in eligible:
            if idx in visited:
                continue

            visited.add(idx)

            if (
                idx not in slot_to_incident
                or try_match(slot_to_incident[idx], visited)
            ):
                slot_to_incident[idx] = incident
                return True

        return False

    # Assign incidents with fewer available options first.
    ordered = sorted(
        incidents,
        key=lambda incident: sum(
            float(matrix.loc[incident, station]) <= threshold
            for station, _ in slots
        ),
    )

    return all(try_match(incident, set()) for incident in ordered)


def minimum_maximum_assignment(matrix, incidents, slots):
    possible_times = sorted({
        float(matrix.loc[incident, station])
        for incident in incidents
        for station, _ in slots
    })

    low, high = 0, len(possible_times) - 1

    while low < high:
        mid = (low + high) // 2

        if has_full_matching(
            matrix, incidents, slots, possible_times[mid]
        ):
            high = mid
        else:
            low = mid + 1

    best_max = possible_times[low]

    # Minimize total time while respecting the optimal maximum.
    costs = [
        [
            float(matrix.loc[incident, station])
            if float(matrix.loc[incident, station]) <= best_max
            else 1e9
            for station, _ in slots
        ]
        for incident in incidents
    ]

    rows, cols = linear_sum_assignment(costs)

    result = {}

    for row, col in zip(rows, cols):
        incident = incidents[row]
        station, vessel = slots[col]
        travel_time = float(matrix.loc[incident, station])

        if travel_time > best_max:
            raise RuntimeError("Invalid minimum-maximum assignment.")

        result[incident] = (station, vessel, travel_time)

    return result


# ------------------------------------------------------------
# METHOD 3: STRICT PRIORITY-FIRST
# ------------------------------------------------------------

def priority_first_assignment(matrix, incidents, slots):
    available = slots.copy()
    result = {}

    for priority in PRIORITY_ORDER:
        group = [
            incident for incident in incidents
            if PRIORITIES[incident] == priority
        ]

        if not group:
            continue

        costs = [
            [float(matrix.loc[incident, station])
             for station, _ in available]
            for incident in group
        ]

        rows, cols = linear_sum_assignment(costs)
        used = set()

        for row, col in zip(rows, cols):
            incident = group[row]
            station, vessel = available[col]

            result[incident] = (
                station,
                vessel,
                float(matrix.loc[incident, station]),
            )
            used.add(col)

        available = [
            slot for idx, slot in enumerate(available)
            if idx not in used
        ]

    return result


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def evaluate_assignment(scenario_id, method, assignment, capacities):
    times = [item[2] for item in assignment.values()]

    critical_times = [
        item[2] for incident, item in assignment.items()
        if PRIORITIES[incident] == "Critical"
    ]

    return {
        "scenario": scenario_id,
        "method": method,
        "assigned_incidents": len(assignment),
        "total_incidents": len(PRIORITIES),
        "coverage_percent": (
            100 * len(assignment) / len(PRIORITIES)
        ),
        "total_travel_time_min": sum(times),
        "mean_response_time_min": np.mean(times) if times else np.nan,
        "maximum_response_time_min": max(times) if times else np.nan,
        "critical_mean_response_min": (
            np.mean(critical_times) if critical_times else np.nan
        ),
        "total_available_vessels": sum(capacities.values()),
    }


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    stations, incidents, base_matrix = load_baseline()
    rng = np.random.default_rng(RANDOM_SEED)

    print("Experiment 08: Monte Carlo Evaluation")
    print("Stations:", stations)
    print("Incidents:", incidents)
    print("Scenarios:", N_SCENARIOS)
    print("Random seed:", RANDOM_SEED)

    records = []

    for scenario_id in range(1, N_SCENARIOS + 1):
        matrix, capacities = generate_scenario(
            base_matrix, stations, incidents, rng
        )

        slots = create_slots(stations, capacities)

        if len(slots) < len(incidents):
            raise RuntimeError(
                "Scenario has insufficient vessel capacity."
            )

        assignments = {
            METHODS[0]: minimum_total_assignment(
                matrix, incidents, slots
            ),
            METHODS[1]: minimum_maximum_assignment(
                matrix, incidents, slots
            ),
            METHODS[2]: priority_first_assignment(
                matrix, incidents, slots
            ),
        }

        for method in METHODS:
            records.append(
                evaluate_assignment(
                    scenario_id,
                    method,
                    assignments[method],
                    capacities,
                )
            )

        if scenario_id % 50 == 0:
            print(f"Completed {scenario_id}/{N_SCENARIOS} scenarios")

    results = pd.DataFrame(records)

    results_path = RESULTS_DIR / "experiment_08_scenario_results.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby("method")
        .agg(
            scenarios=("scenario", "count"),
            mean_total_time_min=("total_travel_time_min", "mean"),
            std_total_time_min=("total_travel_time_min", "std"),
            mean_response_time_min=("mean_response_time_min", "mean"),
            mean_maximum_response_min=("maximum_response_time_min", "mean"),
            mean_critical_response_min=("critical_mean_response_min", "mean"),
            mean_coverage_percent=("coverage_percent", "mean"),
        )
        .reset_index()
    )

    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(2)

    summary_path = RESULTS_DIR / "experiment_08_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Chart 1: distribution of total travel time across scenarios.
    plt.figure(figsize=(11, 6))

    results.boxplot(
        column="total_travel_time_min",
        by="method",
        rot=12,
    )

    plt.suptitle("")
    plt.title("Experiment 08: Total Travel Time Across Scenarios")
    plt.xlabel("Allocation Method")
    plt.ylabel("Total Travel Time (minutes)")
    plt.tight_layout()

    chart1 = VIS_DIR / "experiment_08_total_time.png"
    plt.savefig(chart1, dpi=200, bbox_inches="tight")
    plt.close()

    # Chart 2: coverage by method.
    coverage = summary.set_index("method")["mean_coverage_percent"]

    plt.figure(figsize=(9, 5))
    coverage.plot(kind="bar")
    plt.title("Experiment 08: Mean Incident Coverage")
    plt.xlabel("Allocation Method")
    plt.ylabel("Coverage (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=12, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    chart2 = VIS_DIR / "experiment_08_coverage.png"
    plt.savefig(chart2, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSUMMARY RESULTS")
    print(summary.to_string(index=False))

    print("\nSaved files:")
    print(results_path)
    print(summary_path)
    print(chart1)
    print(chart2)

    print(
        "\nLimitation: these are synthetic scenarios. "
        "Independent travel-time multipliers approximate uncertainty; "
        "they are not a validated ocean-weather model."
    )


if __name__ == "__main__":
    main()
