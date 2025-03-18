import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import xml.etree.ElementTree as ET
import random
from scipy.ndimage import maximum_filter, gaussian_gradient_magnitude
import pandas as pd
import time
import multiprocessing as mp
from functools import partial
import tqdm

# Try to import GPU libraries, use CPU if not available
try:
    import cupy as cp
    import numba
    from numba import cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries not available. Using CPU parallelization only.")

# Configuration Parameters
STEP_LEN = 5.0
ITER_MAX = 2800
CRIT_STEP_HEIGHT = 0.3
CRIT_FLATNESS = 0.5236
CRIT_SLOPE = 0.5236
WEIGHT_STEP = 0.4
WEIGHT_FLATNESS = 0.4
WEIGHT_SLOPE = 0.2
TRAV_LIMIT = 0.6

def extract_elevation_from_dae(dae_path, scale=(1.0, 1.0, 0.5)):
    """Extract elevation data from a .dae file and create a normalized grid."""
    tree = ET.parse(dae_path)
    root = tree.getroot()
    ns = {'collada': 'http://www.collada.org/2005/11/COLLADASchema'}
    
    positions = []
    for float_array in root.findall(".//collada:float_array", ns):
        values = list(map(float, float_array.text.split()))
        positions.extend(values)
    
    vertices = np.array(positions).reshape(-1, 3)
    scaled_vertices = vertices * np.array(scale)
    
    x, y, z = scaled_vertices[:, 0], scaled_vertices[:, 1], scaled_vertices[:, 2]
    
    grid_x, grid_y = np.linspace(x.min(), x.max()+70, 100), np.linspace(y.min(), y.max()+70, 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    grid_z = np.nan_to_num(grid_z, nan=0)
    
    if scale == (0.125, 0.125, 0.125):
        grid_z = -np.nan_to_num(grid_z, nan=0)
        grid_z[grid_z > 200] = 0
        grid_z[(grid_z < 15.0) & (grid_z > 10)] = 7.5
        grid_z[(grid_z < 10.0) & (grid_z > 5)] = 5
        grid_z[(grid_z < 5.0) & (grid_z > 0)] = 5.0
        grid_z[grid_z < 0] = 0.0
    else:
        grid_z[grid_z < 0] = 0

    elevation = (grid_z - grid_z.min()) / (grid_z.max() - grid_z.min())
    return elevation, grid_x, grid_y, grid_z

def compute_traversability_map(elevation):
    """Generate traversability map combining step height, flatness, and slope."""
    step_height_map = maximum_filter(np.abs(elevation), size=5)
    
    # Compute flatness
    flatness = np.zeros_like(elevation)
    half = 2  # size=5 -> half=2
    for i in range(half, elevation.shape[0]-half):
        for j in range(half, elevation.shape[1]-half):
            region = elevation[i-half:i+half+1, j-half:j+half+1]
            x, y = np.meshgrid(np.linspace(-1, 1, region.shape[0]), np.linspace(-1, 1, region.shape[1]))
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
            z = region.reshape(-1, 1)
            A, _, _, _ = np.linalg.lstsq(np.hstack([y, x, np.ones_like(x)]), z, rcond=None)
            normal = np.array([A[0][0], A[1][0], -1])
            angle = np.arccos(normal[2] / np.linalg.norm(normal))
            flatness[i, j] = angle
    
    flatness_norm = flatness / 1.57
    flatness_norm = np.clip(flatness_norm, 0, 1)
    
    # Compute slope
    slope = gaussian_gradient_magnitude(elevation, sigma=1)
    
    # Normalize each factor by its critical value
    step_norm = np.clip(step_height_map / CRIT_STEP_HEIGHT, 0, 1)
    flatness_norm = np.clip(flatness_norm / CRIT_FLATNESS, 0, 1)
    slope_norm = np.clip(slope / CRIT_SLOPE, 0, 1)

    # Combine maps with weights
    traversability = (
        WEIGHT_STEP * step_norm +
        WEIGHT_FLATNESS * flatness_norm +
        WEIGHT_SLOPE * slope_norm
    )

    return np.clip(traversability, 0, 1)

def distance(a, b):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def rrt_star(start, goal, grid, traversability_map, step_len=STEP_LEN, iter_max=ITER_MAX, neighbor_radius=5.0):
    """RRT* path planning with traversability costs."""
    # Early exit condition: if start == goal, return empty path and zero cost
    if start == goal:
        return [], 0.0
        
    nodes = [start]
    parents = {start: None}
    costs = {start: 0.0}

    kdtree = KDTree(nodes)

    for i in range(iter_max):
        # Random sampling
        rand_point = (random.uniform(0, grid.shape[0]), random.uniform(0, grid.shape[1]))
        
        # Find the nearest node
        _, idx = kdtree.query(rand_point)
        nearest = nodes[idx]
        
        # Move towards the sampled point
        direction = np.array(rand_point) - np.array(nearest)
        if np.linalg.norm(direction) == 0:
            continue
        direction = direction / np.linalg.norm(direction)
        new_node = tuple(np.array(nearest) + step_len * direction)
        
        # Check bounds
        if not (0 <= new_node[0] < grid.shape[0] and 0 <= new_node[1] < grid.shape[1]):
            continue

        # Evaluate traversability at this point
        t_value = traversability_map[int(new_node[1]), int(new_node[0])]
        if t_value > TRAV_LIMIT:
            continue

        # Find neighbors
        neighbor_indices = kdtree.query_ball_point(new_node, neighbor_radius)
        neighbors = [nodes[i] for i in neighbor_indices]

        best_parent = None
        best_cost = float('inf')
        for nbr in neighbors:
            t_new = traversability_map[int(new_node[1]), int(new_node[0])]
            new_cost = costs[nbr] + distance(nbr, new_node) * (1.0 + t_new)
            if new_cost < best_cost:
                best_cost = new_cost
                best_parent = nbr

        if best_parent is None:
            continue

        # Add node
        nodes.append(new_node)
        parents[new_node] = best_parent
        costs[new_node] = best_cost
        kdtree = KDTree(nodes)

        # Rewiring
        neighbor_indices = kdtree.query_ball_point(new_node, neighbor_radius)
        neighbors = [nodes[i] for i in neighbor_indices if nodes[i] != new_node and nodes[i] != best_parent]

        for nbr in neighbors:
            t_nbr = traversability_map[int(nbr[1]), int(nbr[0])]
            direct_cost = costs[new_node] + distance(new_node, nbr) * (1.0 + t_nbr)
            if direct_cost < costs[nbr]:
                parents[nbr] = new_node
                costs[nbr] = direct_cost

        # Check if goal is reached
        if distance(new_node, goal) < step_len:
            t_goal = traversability_map[int(goal[1]), int(goal[0])]
            goal_cost = costs[new_node] + distance(new_node, goal) * (1.0 + t_goal)
            parents[goal] = new_node
            costs[goal] = goal_cost
            nodes.append(goal)
            break

    # Extract path
    if goal not in parents:
        return [], float('inf')

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    return path[::-1], costs.get(goal, float('inf'))

def filter_points(points, traversability_map, threshold=0.6):
    """Filter points based on traversability and bounds."""
    points = np.array(points)
    x_coords, y_coords = points[:, 0], points[:, 1]
    
    # First check boundary conditions
    boundary_mask = (
        (x_coords >= 20) & (x_coords < traversability_map.shape[1]) &
        (y_coords >= 20) & (y_coords < traversability_map.shape[0])
    )
    
    valid_points = points[boundary_mask]
    
    if len(valid_points) == 0:
        return valid_points
    
    # Then check traversability for points within bounds
    x_valid, y_valid = valid_points[:, 0], valid_points[:, 1]
    traversability_mask = traversability_map[y_valid.astype(int), x_valid.astype(int)] <= threshold
    
    return valid_points[traversability_mask]

def process_pair_cpu(start, goal, elevation, traversability_map):
    """Process a single start-goal pair and return results."""
    if goal == start:
        path, cost = [], 0.0
    else:
        path, cost = rrt_star(start, goal, elevation, traversability_map)
    
    return {
        "Start_X": start[0],
        "Start_Y": start[1],
        "Goal_X": goal[0],
        "Goal_Y": goal[1],
        "Path_Length": len(path),
        "Traversability_Cost": cost
    }

if GPU_AVAILABLE:
    # Define Numba JIT compiled version of RRT* for GPU acceleration
    @numba.cuda.jit
    def rrt_star_gpu_kernel(start_x, start_y, goal_x, goal_y, trav_map, results_cost, results_path_len):
        idx = cuda.grid(1)
        if idx < start_x.shape[0]:
            start = (start_x[idx], start_y[idx])
            goal = (goal_x[idx], goal_y[idx])
            
            # Simple distance-based cost as fallback
            # In a real implementation, this would be the full RRT* algorithm
            if start == goal:
                results_cost[idx] = 0.0
                results_path_len[idx] = 0
            else:
                # Direct distance + traversability penalty
                tx = int(goal[0])
                ty = int(goal[1]) 
                t_value = 0.0
                if 0 <= tx < trav_map.shape[1] and 0 <= ty < trav_map.shape[0]:
                    t_value = trav_map[ty, tx]
                
                direct_dist = ((start[0] - goal[0])**2 + (start[1] - goal[1])**2)**0.5
                results_cost[idx] = direct_dist * (1.0 + t_value)
                results_path_len[idx] = int(direct_dist / STEP_LEN) + 1

    def process_pairs_gpu(start_points, goal_points, traversability_map):
        """Process all start-goal pairs using GPU."""
        n_pairs = len(start_points) * len(goal_points)
        all_starts_x = np.zeros(n_pairs, dtype=np.float32)
        all_starts_y = np.zeros(n_pairs, dtype=np.float32)
        all_goals_x = np.zeros(n_pairs, dtype=np.float32)
        all_goals_y = np.zeros(n_pairs, dtype=np.float32)
        
        # Create flattened arrays of all start-goal pairs
        idx = 0
        for start in start_points:
            for goal in goal_points:
                all_starts_x[idx] = start[0]
                all_starts_y[idx] = start[1]
                all_goals_x[idx] = goal[0]
                all_goals_y[idx] = goal[1]
                idx += 1
        
        # Transfer data to GPU
        d_starts_x = cp.array(all_starts_x)
        d_starts_y = cp.array(all_starts_y)
        d_goals_x = cp.array(all_goals_x)
        d_goals_y = cp.array(all_goals_y)
        d_trav_map = cp.array(traversability_map)
        
        d_results_cost = cp.zeros(n_pairs, dtype=cp.float32)
        d_results_path_len = cp.zeros(n_pairs, dtype=cp.int32)
        
        # Launch kernel
        threads_per_block = 128
        blocks_per_grid = (n_pairs + threads_per_block - 1) // threads_per_block
        rrt_star_gpu_kernel[blocks_per_grid, threads_per_block](
            d_starts_x, d_starts_y, d_goals_x, d_goals_y, d_trav_map,
            d_results_cost, d_results_path_len
        )
        
        # Retrieve results
        results_cost = cp.asnumpy(d_results_cost)
        results_path_len = cp.asnumpy(d_results_path_len)
        
        # Format results
        results = []
        cost_matrix = []
        idx = 0
        start_idx = 0
        for start in start_points:
            costs = []
            for goal in goal_points:
                cost = results_cost[idx]
                path_len = results_path_len[idx]
                results.append({
                    "Start_X": start[0],
                    "Start_Y": start[1],
                    "Goal_X": goal[0],
                    "Goal_Y": goal[1],
                    "Path_Length": path_len,
                    "Traversability_Cost": cost
                })
                costs.append(cost)
                idx += 1
            cost_matrix.append(costs)
            start_idx += 1
        
        return results, cost_matrix

def process_pairs_cpu_parallel(start_points, goal_points, elevation, traversability_map):
    """Process all start-goal pairs using multiprocessing."""
    total_pairs = len(start_points) * len(goal_points)
    pairs = [(start, goal) for start in start_points for goal in goal_points]
    
    # Create a partial function with fixed parameters
    process_func = partial(process_pair, elevation=elevation, traversability_map=traversability_map)
    
    # Determine number of CPUs to use
    n_cpus = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    print(f"Using {n_cpus} CPU cores for parallel processing")
    
    # Create multiprocessing pool and process all pairs
    results = []
    cost_matrix = []
    
    # Initialize empty cost matrix
    for _ in range(len(start_points)):
        cost_matrix.append([])
    
    # Process pairs in batches to show progress
    with mp.Pool(n_cpus) as pool:
        with tqdm.tqdm(total=total_pairs, desc="Processing pairs") as pbar:
            pair_idx = 0
            for start_idx, start in enumerate(start_points):
                row_costs = []
                for goal in goal_points:
                    result = process_pair(start, goal, elevation, traversability_map)
                    results.append(result)
                    row_costs.append(result["Traversability_Cost"])
                    pair_idx += 1
                    pbar.update(1)
                cost_matrix[start_idx] = row_costs
                
    return results, cost_matrix

def process_pair(start, goal, elevation, traversability_map):
    """Process a single start-goal pair."""
    if goal == start:
        path, cost = [], 0.0
    else:
        path, cost = rrt_star(start, goal, elevation, traversability_map)
    
    return {
        "Start_X": start[0],
        "Start_Y": start[1],
        "Goal_X": goal[0],
        "Goal_Y": goal[1],
        "Path_Length": len(path),
        "Traversability_Cost": cost
    }

def visualize_results(start_points, goal_points, selected_points, traversability_map, elevation):
    """Visualize the deployment results."""
    grid_size = traversability_map.shape[0]
    grid_x = np.linspace(0, 100, grid_size)
    grid_y = np.linspace(0, 100, grid_size)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(grid_x, grid_y, traversability_map, levels=100, cmap='viridis')
    
    # Plot all points
    ax.scatter(*zip(*goal_points), color='black', s=10, alpha=0.6, label='Target')
    ax.scatter(*zip(*start_points), color='blue', s=20, marker='s', alpha=0.6, label='Candidates')
    ax.scatter(*zip(*selected_points), color='red', s=250, marker='$\u2316$', label='Deployed Points')
    
    # Add colorbar and labels
    cbar = fig.colorbar(contour, ax=ax, pad=0.01, shrink=0.8, aspect=20)
    cbar.set_label("Traversability", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([20, 99])
    ax.set_ylim([20, 99])
    
    # Add legend
    legend = ax.legend(loc='upper center', 
                       bbox_to_anchor=(0.61, 1.11),
                       fontsize=15, 
                       frameon=True, 
                       ncol=3,
                       handletextpad=0.3,
                       columnspacing=1.0,
                       borderpad=0.2)
    legend.get_frame().set_alpha(0.5)
    
    plt.tight_layout()
    plt.show()

def find_exemplars(cost_matrix, num_exemplars=4):
    """Find optimal exemplar points using greedy algorithm."""
    cost_matrix = np.array(cost_matrix)  # Convert to NumPy array
    updated_matrix = cost_matrix.copy()
    
    exemplar_indices = []
    for _ in range(num_exemplars):
        # Calculate average costs for each candidate
        row_averages = np.mean(updated_matrix, axis=1)
        
        # Mark existing exemplars with infinite cost to exclude them
        for idx in exemplar_indices:
            row_averages[idx] = np.inf
        
        # Find the candidate with minimum average cost
        exemplar_index = np.argmin(row_averages)
        exemplar_indices.append(exemplar_index)
        
        # Update cost matrix - for each target, use the minimum cost between 
        # its current cost and the cost through the new exemplar
        exemplar_row = updated_matrix[exemplar_index]
        for i in range(updated_matrix.shape[0]):
            if i not in exemplar_indices:  # Skip exemplars
                updated_matrix[i] = np.minimum(updated_matrix[i], exemplar_row)
    
    return exemplar_indices

def main():
    """Main execution function."""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dae_path = os.path.join(project_root, "worlds", "paper_non_convex.dae")
    
    # Process terrain data
    print("Loading and processing terrain data...")
    scale = (1.0, 1.0, 0.5)
    elevation, grid_x, grid_y, grid_z = extract_elevation_from_dae(dae_path, scale)
    rows, cols = elevation.shape  # Get dimensions
    clip_index = 0
    elevation = elevation[clip_index:, clip_index:]  # Apply clipping
    traversability_map = compute_traversability_map(elevation)
    
    # Generate and filter points
    print("Generating candidate and target points...")
    grid_size = 8
    x1 = np.linspace(25, 95, grid_size)
    y1 = np.linspace(14, 95, grid_size)
    X1, Y1 = np.meshgrid(x1, y1)
    start_points = [(int(x), int(y)) for x, y in zip(X1.ravel(), Y1.ravel())]
    
    # Define target point configurations
    goal_config = {
        (38, 75): {'num_points': 50, 'x_var': (-5, 5), 'y_var': (-15, 6)},
        (35, 55): {'num_points': 30, 'x_var': (-5, 5), 'y_var': (-5, 5)},
        (80, 38): {'num_points': 40, 'x_var': (-5, 5), 'y_var': (-6, 6)},
        (87, 70): {'num_points': 50, 'x_var': (-5, 5), 'y_var': (-5, 22)},
        (64, 40): {'num_points': 40, 'x_var': (-5, 5), 'y_var': (-6, 6)},
        (85, 93): {'num_points': 50, 'x_var': (-5, 5), 'y_var': (-10, 22)}
    }

    # Generate goal points
    goal_points = []
    for mean_point, config in goal_config.items():
        mean_x, mean_y = mean_point
        points = [(
            mean_x + np.random.randint(*config['x_var']), 
            mean_y + np.random.randint(*config['y_var'])
        ) for _ in range(config['num_points'])]
        goal_points.extend(points)
    
    # Filter points based on traversability
    filtered_start_points = filter_points(start_points, traversability_map, threshold=0.6)
    filtered_goal_points = filter_points(goal_points, traversability_map)
    
    # Convert to tuples for consistency
    start_points = [tuple(point) for point in filtered_start_points]
    goal_points = [tuple(point) for point in filtered_goal_points]
    
    print(f"After filtering: {len(start_points)} start points, {len(goal_points)} goal points")
    
    # Calculate paths and costs using the appropriate method
    print("Calculating paths and costs...")
    start_time = time.time()
    
    if GPU_AVAILABLE and False:  # Add a flag to enable/disable GPU processing
        print("Using GPU acceleration")
        results, cost_matrix = process_pairs_gpu(start_points, goal_points, traversability_map)
    else:
        print("Using CPU parallel processing")
        with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
            # Create all pairs
            pairs = [(start, goal) for start in start_points for goal in goal_points]
            
            # Create a partial function with fixed parameters
            process_func = partial(process_pair, elevation=elevation, traversability_map=traversability_map)
            
            # Process all pairs in parallel with progress display
            results = list(tqdm.tqdm(
                pool.starmap(process_func, pairs), 
                total=len(pairs),
                desc="Processing pairs"
            ))
        
        # Organize results into cost_matrix
        cost_matrix = []
        idx = 0
        for start in start_points:
            costs = []
            for goal in goal_points:
                costs.append(results[idx]["Traversability_Cost"])
                idx += 1
            cost_matrix.append(costs)
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Find exemplars (deployment points)
    exemplar_indices = find_exemplars(cost_matrix, num_exemplars=4)
    selected_start_points = [start_points[i] for i in exemplar_indices]
    print(f"Selected deployment points: {selected_start_points}")
    
    # Format results for saving
    results_df = pd.DataFrame(results)
    cost_df = pd.DataFrame(cost_matrix)
    
    # Save results
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    csv_result_path = os.path.join(project_root, "data", "rrt_star_results_parallel.csv")
    csv_cost_path = os.path.join(project_root, "data", "rrt_star_costs_parallel.csv")
    results_df.to_csv(csv_result_path, index=False)
    cost_df.to_csv(csv_cost_path, index=False, header=False)
    
    # Visualize results
    visualize_results(start_points, goal_points, selected_start_points, traversability_map, elevation)

if __name__ == "__main__":
    main()
