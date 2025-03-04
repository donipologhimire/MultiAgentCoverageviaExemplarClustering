import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import xml.etree.ElementTree as ET
import random
from scipy.ndimage import maximum_filter, gaussian_gradient_magnitude
import pandas as pd

# Configuration Parameters
# ----------------------
# RRT* Parameters
STEP_LEN = 3.0        # Step size for RRT* path planning
ITER_MAX = 1000        # Maximum iterations for RRT*

# Traversability Parameters
CRIT_STEP_HEIGHT = 0.3    # Critical value for step height
CRIT_FLATNESS = 0.5236    # Critical value for terrain flatness
CRIT_SLOPE = 0.5236       # Critical value for slope

# Weights for traversability calculation
WEIGHT_STEP = 0.4         # Weight for step height
WEIGHT_FLATNESS = 0.4     # Weight for flatness
WEIGHT_SLOPE = 0.2        # Weight for slope

# Threshold for untraversable terrain
TRAV_LIMIT = 0.6

def extract_elevation_from_dae(dae_path, scale=(1.0, 1.0, 0.5)):
    """
    Extract elevation data from a .dae file and create a normalized grid.
    
    Args:
        dae_path (str): Path to the .dae file
        scale (tuple): Scaling factors for x, y, z coordinates
    
    Returns:
        tuple: (elevation, grid_x, grid_y, grid_z)
    """
    tree = ET.parse(dae_path)
    root = tree.getroot()
    ns = {'collada': 'http://www.collada.org/2005/11/COLLADASchema'}
    
    positions = []
    for float_array in root.findall(".//collada:float_array", ns):
        values = list(map(float, float_array.text.split()))
        positions.extend(values)
    
    vertices = np.array(positions).reshape(-1, 3)
    scaled_vertices = vertices * np.array(scale)
    
    x = scaled_vertices[:, 0]
    y = scaled_vertices[:, 1]
    z = scaled_vertices[:, 2]
    
    # Create an interpolated grid
    grid_x, grid_y = np.linspace(x.min(), x.max()+70, 100), np.linspace(y.min(), y.max()+70, 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    grid_z = np.nan_to_num(grid_z, nan=0)
    # minimum index of grid_z
    index = np.unravel_index(np.argmin(grid_z, axis=None), grid_z.shape)
    
    if scale == (0.125, 0.125, 0.125):
        grid_z = -np.nan_to_num(grid_z, nan=0)
        grid_z[grid_z > 200] = 0
        # grid_z[grid_z > 200] = 0
        grid_z[(grid_z < 15.0) & (grid_z > 10)] = 7.5
        grid_z[(grid_z < 10.0) & (grid_z > 5)] = 5
        grid_z[(grid_z < 5.0) & (grid_z > 0)] = 5.0
        grid_z[grid_z < 0] = 0.0
        # grid_z[grid_z <= 10] = 5.0
        # grid_z = griddata((grid_x.flatten(), grid_y.flatten()), grid_z.flatten(), (grid_x, grid_y), method='linear')
    else:
        grid_z[grid_z < 0] = 0

    # Normalize elevation to [0, 1]
    elevation = (grid_z - grid_z.min()) / (grid_z.max() - grid_z.min())
    # elevation = grid_z
    return elevation,grid_x,grid_y,grid_z

def compute_step_height_map(elevation, size=5):
    """
    Compute step height map using maximum filter.
    
    Args:
        elevation (ndarray): Elevation grid
        size (int): Filter size
    
    Returns:
        ndarray: Step height map
    """
    return maximum_filter(np.abs(elevation), size=size)

def compute_flatness_map(elevation, size=5):
    """
    Compute flatness map by fitting planes to local regions.
    
    Args:
        elevation (ndarray): Elevation grid
        size (int): Local region size
    
    Returns:
        ndarray: Normalized flatness map
    """
    flatness = np.zeros_like(elevation)
    half = size // 2
    for i in range(half, elevation.shape[0]-half):
        for j in range(half, elevation.shape[1]-half):
            region = elevation[i-half:i+half+1, j-half:j+half+1]
            # Fit a plane z = A1*y + A2*x + A3
            x, y = np.meshgrid(np.linspace(-1, 1, region.shape[0]), np.linspace(-1, 1, region.shape[1]))
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
            z = region.reshape(-1, 1)
            A, _, _, _ = np.linalg.lstsq(np.hstack([y, x, np.ones_like(x)]), z, rcond=None)
            normal = np.array([A[0][0], A[1][0], -1])
            angle = np.arccos(normal[2] / np.linalg.norm(normal))
            flatness[i, j] = angle
    # Normalize angle from [0, pi/2] ~ [0,1.57] to [0,1]
    flatness_norm = flatness / 1.57
    flatness_norm = np.clip(flatness_norm, 0, 1)
    return flatness_norm

def compute_slope_map(elevation, sigma=1):
    """
    Compute slope map using gaussian gradient magnitude.
    
    Args:
        elevation (ndarray): Elevation grid
        sigma (float): Gaussian sigma parameter
    
    Returns:
        ndarray: Slope map
    """
    slope = gaussian_gradient_magnitude(elevation, sigma=sigma)
    # slope could range widely. We might want to normalize it.
    # A rough normalization: divide by a constant or use CRIT_SLOPE as a reference.
    return slope

def compute_traversability_map(elevation):
    """
    Generate traversability map combining step height, flatness, and slope.
    
    Args:
        elevation (ndarray): Elevation grid
    
    Returns:
        ndarray: Traversability map
    """
    step_height_map = compute_step_height_map(elevation)
    flatness_map = compute_flatness_map(elevation)
    slope_map = compute_slope_map(elevation)

    # Normalize each factor by its critical value
    step_norm = step_height_map / CRIT_STEP_HEIGHT
    flatness_norm = flatness_map / CRIT_FLATNESS
    slope_norm = slope_map / CRIT_SLOPE

    # Clip values to avoid excessive scaling
    step_norm = np.clip(step_norm, 0, 1)
    flatness_norm = np.clip(flatness_norm, 0, 1)
    # Slope might need clipping too
    slope_norm = np.clip(slope_norm, 0, 1)

    # Combine maps with weights
    traversability = (
        WEIGHT_STEP * step_norm +
        WEIGHT_FLATNESS * flatness_norm +
        WEIGHT_SLOPE * slope_norm
    )

    # traversability should be in [0,1] after weighting if everything is chosen well.
    # If needed, we can ensure normalization:
    traversability = np.clip(traversability, 0, 1)

    return traversability

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def rrt_star(start, goal, grid, traversability_map, step_len=STEP_LEN, iter_max=ITER_MAX, neighbor_radius=5.0):
    """
    RRT* path planning with traversability costs.
    
    Args:
        start (tuple): Start position
        goal (tuple): Goal position
        grid (ndarray): Environment grid
        traversability_map (ndarray): Traversability costs
        step_len (float): Step length
        iter_max (int): Maximum iterations
        neighbor_radius (float): Radius for neighbor search
    
    Returns:
        tuple: (path, cost)
    """
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
        # If you want a hard cutoff:
        if t_value > TRAV_LIMIT:
            continue

        # Find neighbors within NEIGHBOR_RADIUS
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

        # If we didn't find a suitable parent, skip
        if best_parent is None:
            continue

        # Add new node with best parent
        nodes.append(new_node)
        parents[new_node] = best_parent
        costs[new_node] = best_cost
        kdtree = KDTree(nodes)

        # Rewiring
        # Try to see if passing through new_node improves cost for its neighbors
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
            # Connect goal
            t_goal = traversability_map[int(goal[1]), int(goal[0])]
            goal_cost = costs[new_node] + distance(new_node, goal) * (1.0 + t_goal)
            parents[goal] = new_node
            costs[goal] = goal_cost
            nodes.append(goal)
            break

    # Extract path
    if goal not in parents:
        # print("Goal not reached. Try adjusting parameters or weights.")
        return [],float('inf')

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    return path[::-1],costs[goal]


def filter_points(points, traversability_map, threshold=0.6):
    """
    Filter points based on traversability and bounds.
    
    Args:
        points (ndarray): Array of points
        traversability_map (ndarray): Traversability map
        threshold (float): Traversability threshold
    
    Returns:
        ndarray: Filtered points
    """
    points = np.array(points)  # Ensure input is a NumPy array
    x_coords, y_coords = points[:, 0], points[:, 1]
    mask = (
        (x_coords >= 20) & (x_coords < traversability_map.shape[1]) &  # Within x bounds
        (y_coords >= 20) & (y_coords < traversability_map.shape[0]) &  # Within y bounds
                # Below traversability threshold
        (traversability_map[y_coords, x_coords] <= threshold) 
    )
    return points[mask]

# Main Execution
if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dae_path = os.path.join(project_root, "worlds", "paper_non_convex.dae")
    
    # Process terrain data
    scale = (1.0, 1.0, 0.5)
    elevation, grid_x, grid_y, grid_z = extract_elevation_from_dae(dae_path, scale)
    rows, cols = elevation.shape  # Get dimensions
    clip_index = 0
    elevation = elevation[clip_index:, clip_index:]  # Apply clipping
    traversability_map = compute_traversability_map(elevation)
    
    # Generate and filter points
    grid_size = 7  # From 0 to 100 inclusive
    x1 = np.linspace(15, 95, grid_size)
    y1 = np.linspace(10, 95, grid_size)
    X1, Y1 = np.meshgrid(x1, y1)
    start_points = [(int(x), int(y)) for x, y in zip(X1.ravel(), Y1.ravel())]
    goal_means = [(40, 80), (60, 80),(50,40),(80,60),(70,30)] # terrain1
    goal_points = []
    for mean_x, mean_y in goal_means:
        goal_points.extend([(mean_x + np.random.randint(-18, 15), mean_y + np.random.randint(-2, 8)) for _ in range(40)])
    filtered_start_points = filter_points(start_points, traversability_map)
    filtered_goal_points = filter_points(goal_points, traversability_map)
    start_points = [tuple(point) for point in filtered_start_points]
    goal_points = [tuple(point) for point in filtered_goal_points]
    
    # Calculate paths and costs
    results = []
    counter = 0
    cost_matrix = []
    for start in start_points:
        costs = []
        for goal in goal_points:
            counter = counter + 1
            if goal == start:
                path,cost= [], float(0)  # Return immediately if condition is met
            else:
                path, cost = rrt_star(start, goal, elevation, traversability_map)
            costs.append(cost)
            results.append({
            "Start_X": start[0],
            "Start_Y": start[1],
            "Goal_X": goal[0],
            "Goal_Y": goal[1],
            "Path_Length": len(path),
            "Traversability_Cost": cost
            })
        cost_matrix.append(costs)
    results_df = pd.DataFrame(results)
    cost_df = pd.DataFrame(cost_matrix)
    csv_result_path = os.path.join(project_root, "data", "rrt_star_results.csv")
    csv_cost_path  = os.path.join(project_root, "data", "rrt_star_costs.csv")
    results_df.to_csv(csv_result_path, index=False)
    cost_df.to_csv(csv_cost_path, index=False, header=False)

    # Exemplar clustering
    cost_matrix = np.array(cost_matrix)  # Convert to a NumPy array for easier manipulation
    updated_matrix = cost_matrix.copy() 
    num_iterations = 5
    updated_matrix2 = cost_matrix.copy()
    exemplar_indices = []  # List to store exemplar indices
    for _ in range(num_iterations):
        row_averages = np.mean(updated_matrix2, axis=1)
        for idx in exemplar_indices:
            row_averages[idx] = np.inf  # Exclude exemplar rows by setting their averages to infinity
        exemplar_index = np.argmin(row_averages)
        exemplar_indices.append(exemplar_index)  # Add the exemplar index to the list
        exemplar_row = updated_matrix2[exemplar_index]
        for i in range(updated_matrix2.shape[0]):
            if i not in exemplar_indices:  # Skip exemplar rows
                updated_matrix2[i] = np.minimum(updated_matrix2[i], exemplar_row)
    print(f"Exemplar Indices: {exemplar_indices}")
    print('traversability',traversability_map[89,35])
    fig, ax2 = plt.subplots(figsize=(8, 6))  # Creates a figure with a custom size
    grid_size = 100-clip_index
    grid_x = np.linspace(0, 100, grid_size)
    grid_y = np.linspace(0, 100, grid_size)
    contour2 = ax2.contourf(grid_x, grid_y, traversability_map, levels=100, cmap='viridis')
    ax2.scatter(*zip(*goal_points), color='black', s=15, label='Target')
    ax2.scatter(*zip(*start_points), color='green', s=50, marker= '.', label='Candidate')
    selected_start_points = [start_points[i] for i in exemplar_indices]
    ax2.scatter(*zip(*selected_start_points), color='red', s=50, marker='x', label='Deployed Points')
    cbar = fig.colorbar(contour2, ax=ax2, pad=0.01, shrink=1, aspect=20)
    cbar.set_label("Traversability", fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick font size
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim([20, 99])  # Adjust as neededcustom_world_2
    ax2.set_ylim([18, 99])  # Adjust as needed
    legend = ax2.legend(loc='lower center', fontsize=15, frameon=True,ncol=2)
    legend.get_frame().set_alpha(0.5)  # Set transparency of the legend frame
    plt.show()
