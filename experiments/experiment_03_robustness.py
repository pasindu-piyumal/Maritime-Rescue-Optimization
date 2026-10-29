
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "results" / "experiment_01_travel_times.csv"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "visualizations"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INCIDENT_COUNT = 5
BASE_CAPACITY = 2

# --------------------------------------------------
# Load and standardize travel-time data
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df = df.rename(columns={
    "Rescue Station": "rescue_station",
    "Distress Location": "distress_location",
    "Travel Time (minutes)": "travel_time_min",
    "Shortest Path": "shortest_path",
})

required = {
    "rescue_station",
    "distress_location",
    "travel_time_min",
}

if not required.issubset(df.columns):
    raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

cost = df.pivot(
    index="rescue_station",
    columns="distress_location",
    values="travel_time_min",
).sort_index().sort_index(axis=1)

if cost.isna().any().any():
    raise ValueError("Missing station-to-incident travel times.")

stations = list(cost.index)
incidents = list(cost.columns)

if len(incidents) != INCIDENT_COUNT:
    print(f"Warning: expected {INCIDENT_COUNT} incidents, found {len(incidents)}.")

# --------------------------------------------------
# Assignment functions
# --------------------------------------------------

def allocate(cost_matrix, available_stations, capacity, method):
    """Allocate incidents subject to each station's vessel capacity."""

    available_incidents = list(cost_matrix.columns)
    available_stations = list(available_stations)

    if not available_stations or not available_incidents:
        return []

    results = []

    if method == "Greedy baseline":
        # Process incidents in a fixed order, matching Experiment 2B.
        remaining_capacity = {
            station: capacity for station in available_stations
        }

        for incident in available_incidents:
            candidates = [
                (cost_matrix.loc[station, incident], station)
                for station in available_stations
                if remaining_capacity[station] > 0
            ]

            # If all vessels are occupied, this incident remains unassigned.
            if not candidates:
                continue

            travel_time, station = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )

            vessel_number = (
                capacity - remaining_capacity[station] + 1
            )

            results.append({
                "rescue_station": station,
                "distress_location": incident,
                "vessel_number": vessel_number,
                "travel_time_min": float(travel_time),
            })

            remaining_capacity[station] -= 1

    elif method == "Optimization":
        from scipy.optimize import linear_sum_assignment

        slots = [
            (station, vessel_no)
            for station in available_stations
            for vessel_no in range(1, capacity + 1)
        ]

        matrix = [
            [
                cost_matrix.loc[station, incident]
                for incident in available_incidents
            ]
            for station, vessel_no in slots
        ]

        row_ids, col_ids = linear_sum_assignment(matrix)

        for r, c in zip(row_ids, col_ids):
            station, vessel_number = slots[r]
            incident = available_incidents[c]

            results.append({
                "rescue_station": station,
                "distress_location": incident,
                "vessel_number": vessel_number,
                "travel_time_min": float(
                    cost_matrix.loc[station, incident]
                ),
            })

    else:
        raise ValueError(f"Unknown allocation method: {method}")

    return results


# --------------------------------------------------
# Experiment A: vary vessel capacity
# --------------------------------------------------
allocation_rows = []
summary_rows = []

capacity_scenarios = [1, 2, 3]

for capacity in capacity_scenarios:
    for method in ["Greedy baseline", "Optimization"]:
        assignments = allocate(
            cost,
            stations,
            capacity,
            method,
        )

        for row in assignments:
            allocation_rows.append({
                "scenario": f"Capacity_{capacity}_per_station",
                "method": method,
                "capacity_per_station": capacity,
                "station_unavailable": "None",
                **row,
            })

        times = [r["travel_time_min"] for r in assignments]

        summary_rows.append({
            "scenario": f"Capacity_{capacity}_per_station",
            "method": method,
            "capacity_per_station": capacity,
            "station_unavailable": "None",
            "incidents_assigned": len(assignments),
            "incidents_unassigned": len(incidents) - len(assignments),
            "coverage_percent": 100 * len(assignments) / len(incidents),
            "total_travel_time_min": sum(times),
            "mean_travel_time_min": (
                sum(times) / len(times) if times else None
            ),
            "maximum_travel_time_min": max(times) if times else None,
        })

# --------------------------------------------------
# Experiment B: remove one station at a time
# --------------------------------------------------
for failed_station in stations:
    remaining_stations = [
        s for s in stations if s != failed_station
    ]

    for method in ["Greedy baseline", "Optimization"]:
        assignments = allocate(
            cost,
            remaining_stations,
            BASE_CAPACITY,
            method,
        )

        for row in assignments:
            allocation_rows.append({
                "scenario": f"Station_failure_{failed_station}",
                "method": method,
                "capacity_per_station": BASE_CAPACITY,
                "station_unavailable": failed_station,
                **row,
            })

        times = [r["travel_time_min"] for r in assignments]

        summary_rows.append({
            "scenario": f"Station_failure_{failed_station}",
            "method": method,
            "capacity_per_station": BASE_CAPACITY,
            "station_unavailable": failed_station,
            "incidents_assigned": len(assignments),
            "incidents_unassigned": len(incidents) - len(assignments),
            "coverage_percent": 100 * len(assignments) / len(incidents),
            "total_travel_time_min": sum(times),
            "mean_travel_time_min": (
                sum(times) / len(times) if times else None
            ),
            "maximum_travel_time_min": max(times) if times else None,
        })

# --------------------------------------------------
# Save results
# --------------------------------------------------
allocations_df = pd.DataFrame(allocation_rows)
summary_df = pd.DataFrame(summary_rows)

allocations_df.to_csv(
    RESULTS_DIR / "experiment_03_allocations.csv",
    index=False,
)

summary_df.to_csv(
    RESULTS_DIR / "experiment_03_robustness_results.csv",
    index=False,
)

print("\nEXPERIMENT 3: ROBUSTNESS ANALYSIS")
print("\nSummary results:")
print(
    summary_df[
        [
            "scenario",
            "method",
            "incidents_assigned",
            "coverage_percent",
            "total_travel_time_min",
            "mean_travel_time_min",
        ]
    ].round(2).to_string(index=False)
)

# --------------------------------------------------
# Plot A: total travel time under capacity scenarios
# --------------------------------------------------
capacity_results = summary_df[
    summary_df["scenario"].str.startswith("Capacity_")
]

pivot_capacity = capacity_results.pivot(
    index="capacity_per_station",
    columns="method",
    values="total_travel_time_min",
)

ax = pivot_capacity.plot(
    kind="bar",
    figsize=(9, 5),
)

ax.set_title("Effect of Vessel Capacity on Total Travel Time")
ax.set_xlabel("Available Vessels per Station")
ax.set_ylabel("Total Travel Time (minutes)")
ax.legend(title="Allocation Method")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "experiment_03_capacity_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()

# --------------------------------------------------
# Plot B: incident coverage after station failure
# --------------------------------------------------
failure_results = summary_df[
    summary_df["scenario"].str.startswith("Station_failure_")
]

pivot_failure = failure_results.pivot(
    index="station_unavailable",
    columns="method",
    values="coverage_percent",
)

ax = pivot_failure.plot(
    kind="bar",
    figsize=(9, 5),
)

ax.set_title("Incident Coverage When a Rescue Station Fails")
ax.set_xlabel("Unavailable Rescue Station")
ax.set_ylabel("Incidents Assigned (%)")
ax.legend(title="Allocation Method")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "experiment_03_station_failure.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()

print("\nExperiment 3 completed.")
print(f"CSV results: {RESULTS_DIR}")
print(f"Charts: {PLOTS_DIR}")
