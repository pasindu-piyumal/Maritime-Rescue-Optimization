
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Project paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "results" / "experiment_01_travel_times.csv"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "visualizations"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VESSELS_PER_STATION = 2

# Load and standardize Experiment 1 data
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
    raise ValueError("Some station-to-incident travel times are missing.")

stations = list(cost.index)
incidents = list(cost.columns)

print("\nTravel-time matrix (minutes):")
print(cost)

# Create two available vessel slots per station
slots = [
    (station, vessel_number)
    for station in stations
    for vessel_number in range(1, VESSELS_PER_STATION + 1)
]

# Optimization: assign every incident to a unique vessel slot
expanded_cost = [
    [cost.loc[station, incident] for incident in incidents]
    for station, vessel_number in slots
]

row_indices, col_indices = linear_sum_assignment(expanded_cost)

optimized_rows = []

for row, col in zip(row_indices, col_indices):
    station, vessel_number = slots[row]
    incident = incidents[col]

    optimized_rows.append({
        "method": "Capacity-constrained optimization",
        "distress_location": incident,
        "rescue_station": station,
        "vessel_number": vessel_number,
        "travel_time_min": cost.loc[station, incident],
    })

optimized_df = pd.DataFrame(optimized_rows)

# Baseline: process incidents in order and choose the nearest
# station that still has an available vessel.
remaining = {station: VESSELS_PER_STATION for station in stations}
baseline_rows = []

for incident in incidents:
    available = [
        station for station in stations
        if remaining[station] > 0
    ]

    if not available:
        raise ValueError("Insufficient vessel capacity for all incidents.")

    station = min(
        available,
        key=lambda s: cost.loc[s, incident]
    )

    vessel_number = (
        VESSELS_PER_STATION - remaining[station] + 1
    )
    remaining[station] -= 1

    baseline_rows.append({
        "method": "Capacity-constrained greedy baseline",
        "distress_location": incident,
        "rescue_station": station,
        "vessel_number": vessel_number,
        "travel_time_min": cost.loc[station, incident],
    })

baseline_df = pd.DataFrame(baseline_rows)

# Compare both methods using the same five incidents
allocations = pd.concat(
    [baseline_df, optimized_df],
    ignore_index=True,
)

summary = (
    allocations.groupby("method")
    .agg(
        assigned_incidents=("distress_location", "nunique"),
        total_travel_time_min=("travel_time_min", "sum"),
        mean_travel_time_min=("travel_time_min", "mean"),
        maximum_travel_time_min=("travel_time_min", "max"),
    )
    .reset_index()
)

print("\nGreedy baseline allocation:")
print(baseline_df.to_string(index=False))

print("\nOptimized allocation:")
print(optimized_df.to_string(index=False))

print("\nComparison:")
print(summary.round(2).to_string(index=False))

# Calculate improvement relative to the baseline
baseline_total = baseline_df["travel_time_min"].sum()
optimized_total = optimized_df["travel_time_min"].sum()

improvement = (
    (baseline_total - optimized_total) / baseline_total * 100
    if baseline_total > 0 else 0
)

print(f"\nTotal travel-time reduction: {improvement:.2f}%")

# Save revised results without overwriting earlier experiments
allocations.to_csv(
    RESULTS_DIR / "experiment_02b_allocations.csv",
    index=False,
)

summary.to_csv(
    RESULTS_DIR / "experiment_02b_comparison.csv",
    index=False,
)

# Plot total travel-time comparison
plt.figure(figsize=(9, 5))
plt.bar(
    summary["method"],
    summary["total_travel_time_min"],
)
plt.title("Capacity-Constrained Rescue Allocation")
plt.ylabel("Total Travel Time (minutes)")
plt.xlabel("Allocation Method")
plt.xticks(rotation=10, ha="right")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "experiment_02b_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("\nExperiment 2B completed.")
