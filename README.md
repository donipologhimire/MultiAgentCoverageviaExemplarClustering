# Multi-Agent Coverage via Exemplar Clustering

This repository contains the implementation of a multi-agent coverage strategy using exemplar clustering for optimal deployment points and visibility graph traversability calculations.

## Requirements

Before running the scripts, you need to install the PyVisGraph package:

```bash
pip install pyvisgraph
```

Or install from the source:

```bash
git clone https://github.com/TaipanRex/pyvisgraph.git
cd pyvisgraph
pip install -e .
```

## Terrain Environments

Our approach was evaluated on different terrain types:

**Hilly Terrain 1**:
![Hilly Terrain](figures/2025_visibility_hills_problem_statement.png)

**Hilly Terrain 2**:
![Non-Convex Terrain](figures/2025_visibility_traversability_non_convex.png)

## Running the Experiments

### 1. Visibility Graph Generation

The visibility graph provides efficient path planning by connecting visible points in the environment:

![Visibility Graph](figures/2025_visibility_VisGraph.png)

### 2. RRT Path Planning with Traversability

Run the RRT path planner that accounts for terrain traversability:

```bash
python scripts/RRT_start_traversability.py
```

This will generate traversability and elevation maps showing optimal paths:

**Traversability Map**:
![Traversability Map](figures/2025_visibility_traversability.png)

**Elevation Map**:
![Elevation Map](figures/2025_visibility_elevation_map.png)

### 3. Visibility Graph for Deployment

The visibility graph helps identify optimal deployment locations for multi-agent coverage:

![Deployment Hotspots](figures/2025_visibility_deployment_hotspot.png)

### 4. Agent Deployment Policy

Finally, run the deployment policy script to determine the optimal placement of agents:

```bash
python scripts/deployment_policy.py
```

This script uses exemplar clustering to identify the best positions for deploying agents to maximize coverage while considering terrain traversability constraints.

## Results

The figures directory contains all the result visualizations from the experiments. The deployment policy uses a cost matrix approach to identify exemplar points that provide optimal coverage of the target region.

## Implementation Details

The implementation combines:
- RRT* path planning with traversability costs
- Visibility graph construction for efficient path planning
- Exemplar clustering for identifying optimal deployment points
- Terrain analysis for traversability estimation

Each script can be run independently or as part of the complete workflow for multi-agent coverage planning.
