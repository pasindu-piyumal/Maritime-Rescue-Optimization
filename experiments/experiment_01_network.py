
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# --------------------------------------------------
# Experiment 1: Simulated Maritime Rescue Network
# --------------------------------------------------

# Resolve paths relative to the project root.
# Expected location: project_root/experiments/experiment_01_network.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "experiment_01_travel_times.csv"
FIGURE_PATH = VISUALIZATIONS_DIR / "experiment_01_network.png"

# Create an undirected weighted graph
G = nx.Graph()

# Node positions for visualization (illustrative only)
positions = {
    "R1": (0, 0),
    "R2": (4, 0),
    "R3": (8, 0),
    "W1": (1, 1),
    "W2": (2, 0),
    "W3": (3, 1),
    "W4": (5, 1),
    "W5": (7, 1),
    "D1": (0, 3),
    "D2": (3, 3),
    "D3": (6, 3),
    "D4": (8, 3),
    "D5": (4, -2),
}

# Node categories
rescue_stations = ["R1", "R2", "R3"]
waypoints = ["W1", "W2", "W3", "W4", "W5"]
distress_locations = ["D1", "D2", "D3", "D4", "D5"]

# Add nodes
G.add_nodes_from(
    (node, {"node_type": "rescue_station"})
    for node in rescue_stations
)
G.add_nodes_from(
    (node, {"node_type": "waypoint"})
    for node in waypoints
)
G.add_nodes_from(
    (node, {"node_type": "distress"})
    for node in distress_locations
)

# Edges represent navigable connections.
# Weights are simulated travel times in minutes.
edges = [
    ("R1", "W1", 12),
    ("R1", "W2", 18),
    ("W1", "W2", 8),
    ("W1", "D1", 10),
    ("W1", "D2", 18),
    ("W2", "D2", 12),
    ("W2", "D5", 20),
    ("W2", "W3", 15),
    ("W3", "D2", 9),
    ("W3", "D3", 11),
    ("W3", "R2", 10),
    ("R2", "W4", 12),
    ("W4", "D3", 8),
    ("W4", "D4", 15),
    ("W4", "W5", 9),
    ("W5", "D4", 12),
    ("W5", "D5", 18),
    ("R3", "W5", 10),
    ("R3", "D4", 14),
    ("R2", "D5", 16),
]

G.add_weighted_edges_from(edges, weight="travel_time")

# --------------------------------------------------
# Calculate shortest travel times
# --------------------------------------------------

results = []

for station in rescue_stations:
    for incident in distress_locations:
        try:
            travel_time = nx.shortest_path_length(
                G,
                source=station,
                target=incident,
                weight="travel_time",
            )

            path = nx.shortest_path(
                G,
                source=station,
                target=incident,
                weight="travel_time",
            )

            results.append({
                "Rescue Station": station,
                "Distress Location": incident,
                "Travel Time (minutes)": travel_time,
                "Shortest Path": " -> ".join(path),
            })

        except nx.NetworkXNoPath:
            results.append({
                "Rescue Station": station,
                "Distress Location": incident,
                "Travel Time (minutes)": None,
                "Shortest Path": "No available path",
            })

results_df = pd.DataFrame(results)

print("\nSHORTEST TRAVEL TIMES")
print(
    results_df.pivot(
        index="Rescue Station",
        columns="Distress Location",
        values="Travel Time (minutes)",
    ).to_string()
)

print("\nSHORTEST PATH DETAILS")
print(results_df.to_string(index=False))

# Save results for later experiments and report writing
results_df.to_csv(CSV_PATH, index=False)

# --------------------------------------------------
# Visualize the maritime network
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))

nx.draw_networkx_edges(
    G,
    positions,
    edge_color="gray",
    width=1.5,
    ax=ax,
)

nx.draw_networkx_nodes(
    G,
    positions,
    nodelist=rescue_stations,
    node_color="green",
    node_size=900,
    node_shape="s",
    ax=ax,
)

nx.draw_networkx_nodes(
    G,
    positions,
    nodelist=waypoints,
    node_color="skyblue",
    node_size=650,
    ax=ax,
)

nx.draw_networkx_nodes(
    G,
    positions,
    nodelist=distress_locations,
    node_color="red",
    node_size=850,
    node_shape="D",
    ax=ax,
)

nx.draw_networkx_labels(
    G,
    positions,
    font_size=10,
    font_weight="bold",
    ax=ax,
)

edge_labels = {
    (u, v): f"{data['travel_time']} min"
    for u, v, data in G.edges(data=True)
}

nx.draw_networkx_edge_labels(
    G,
    positions,
    edge_labels=edge_labels,
    font_size=8,
    ax=ax,
)

ax.set_title("Simulated Maritime Search-and-Rescue Network")

legend_elements = [
    Line2D(
        [0], [0],
        marker="s",
        color="w",
        label="Rescue Stations",
        markerfacecolor="green",
        markersize=9,
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Maritime Waypoints",
        markerfacecolor="skyblue",
        markersize=9,
    ),
    Line2D(
        [0], [0],
        marker="D",
        color="w",
        label="Distress Locations",
        markerfacecolor="red",
        markersize=9,
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=9,
    frameon=True,
    labelspacing=0.8,
    handletextpad=0.8,
    borderpad=0.7,
    handlelength=1.5,
)

ax.axis("off")
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")

print("\nExperiment 1 completed successfully.")
print(f"Saved CSV: {CSV_PATH}")
print(f"Saved figure: {FIGURE_PATH}")

plt.show()
plt.close(fig)
