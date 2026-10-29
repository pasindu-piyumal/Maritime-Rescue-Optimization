from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "results" / "experiment_01_travel_times.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "visualizations"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Load Experiment 1 results
df = pd.read_csv(INPUT_FILE)

# Standardize column names to match the experiment code
df = df.rename(columns={
    "Rescue Station": "rescue_station",
    "Distress Location": "distress_location",
    "Travel Time (minutes)": "travel_time_min",
    "Shortest Path": "shortest_path",
})

print("\nInput data preview:")
print(df.head())

# Expected columns from Experiment 1:
# rescue_station, distress_location, travel_time_min, shortest_path
# Adjust these names below if your CSV uses different column names.

required_columns = {
    "rescue_station",
    "distress_location",
    "travel_time_min",
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV columns are {list(df.columns)}. "
        "Update the column names in this script to match your file."
    )

# Build the travel-time matrix
cost_matrix = df.pivot(
    index="rescue_station",
    columns="distress_location",
    values="travel_time_min",
)

cost_matrix = cost_matrix.sort_index().sort_index(axis=1)

print("\nTravel-time matrix (minutes):")
print(cost_matrix)

# Baseline: assign each incident to its nearest station independently.
# This is a comparison baseline, not a complete vessel-capacity model.
baseline_rows = []

for incident in cost_matrix.columns:
    station = cost_matrix[incident].idxmin()
    travel_time = cost_matrix.loc[station, incident]

    baseline_rows.append({
        "method": "Nearest-station baseline",
        "distress_location": incident,
        "rescue_station": station,
        "travel_time_min": travel_time,
    })

baseline_df = pd.DataFrame(baseline_rows)

# Optimization: one assignment per station and per incident.
# With 3 stations and 5 incidents, this assigns 3 incidents.
matrix = cost_matrix.to_numpy()
row_indices, col_indices = linear_sum_assignment(matrix)

optimized_rows = []

for row, col in zip(row_indices, col_indices):
    optimized_rows.append({
        "method": "Minimum-cost assignment",
        "distress_location": cost_matrix.columns[col],
        "rescue_station": cost_matrix.index[row],
        "travel_time_min": matrix[row, col],
    })

optimized_df = pd.DataFrame(optimized_rows)

print("\nBaseline allocation:")
print(baseline_df.to_string(index=False))

print("\nOptimized allocation:")
print(optimized_df.to_string(index=False))

# Summary metrics
summary = pd.DataFrame([
    {
        "method": "Nearest-station baseline",
        "assigned_incidents": len(baseline_df),
        "total_travel_time_min": baseline_df["travel_time_min"].sum(),
        "mean_travel_time_min": baseline_df["travel_time_min"].mean(),
    },
    {
        "method": "Minimum-cost assignment",
        "assigned_incidents": len(optimized_df),
        "total_travel_time_min": optimized_df["travel_time_min"].sum(),
        "mean_travel_time_min": optimized_df["travel_time_min"].mean(),
    },
])

print("\nMethod comparison:")
print(summary.round(2).to_string(index=False))

# Save the detailed allocations and summary
allocation_results = pd.concat(
    [baseline_df, optimized_df],
    ignore_index=True,
)

allocation_results.to_csv(
    RESULTS_DIR / "experiment_02_allocations.csv",
    index=False,
)

summary.to_csv(
    RESULTS_DIR / "experiment_02_comparison.csv",
    index=False,
)

# Plot optimized assignments
plt.figure(figsize=(9, 5))
plt.bar(
    optimized_df["distress_location"],
    optimized_df["travel_time_min"],
)
plt.title("Optimized Rescue Allocation: Travel Times")
plt.xlabel("Distress Location")
plt.ylabel("Travel Time (minutes)")
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "experiment_02_allocation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("\nExperiment 2 completed.")
print(f"Results saved to: {RESULTS_DIR}")
print(f"Plot saved to: {PLOTS_DIR}")