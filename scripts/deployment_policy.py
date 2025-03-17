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
STEP_LEN = 5.0
ITER_MAX = 1800
CRIT_STEP_HEIGHT = 0.3
CRIT_FLATNESS = 0.5236
CRIT_SLOPE = 0.5236
WEIGHT_STEP = 0.4
WEIGHT_FLATNESS = 0.4
WEIGHT_SLOPE = 0.2
TRAV_LIMIT = 0.6

def extract_elevation_from_dae(dae_path, scale=(1.0, 1.0, 0.5)):
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

def compute_step_height_map(elevation, size=5):
    return maximum_filter(np.abs(elevation), size=size)

def compute_flatness_map(elevation, size=5):
    flatness = np.zeros_like(elevation)
    half = size // 2
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
    return flatness_norm

def compute_slope_map(elevation, sigma=1):
    slope = gaussian_gradient_magnitude(elevation, sigma=sigma)
    return slope

def compute_traversability_map(elevation):
    step_height_map = compute_step_height_map(elevation)
    flatness_map = compute_flatness_map(elevation)
    slope_map = compute_slope_map(elevation)

    step_norm = step_height_map / CRIT_STEP_HEIGHT
    flatness_norm = flatness_map / CRIT_FLATNESS
    slope_norm = slope_map / CRIT_SLOPE

    step_norm = np.clip(step_norm, 0, 1)
    flatness_norm = np.clip(flatness_norm, 0, 1)
    slope_norm = np.clip(slope_norm, 0, 1)

    traversability = (
        WEIGHT_STEP * step_norm +
        WEIGHT_FLATNESS * flatness_norm +
        WEIGHT_SLOPE * slope_norm
    )

    traversability = np.clip(traversability, 0, 1)
    return traversability

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def rrt_star(start, goal, grid, traversability_map, step_len=STEP_LEN, iter_max=ITER_MAX, neighbor_radius=5.0):
    nodes = [start]
    parents = {start: None}
    costs = {start: 0.0}
    kdtree = KDTree(nodes)

    for i in range(iter_max):
        rand_point = (random.uniform(0, grid.shape[0]), random.uniform(0, grid.shape[1]))
        _, idx = kdtree.query(rand_point)
        nearest = nodes[idx]
        direction = np.array(rand_point) - np.array(nearest)
        if np.linalg.norm(direction) == 0:
            continue
        direction = direction / np.linalg.norm(direction)
        new_node = tuple(np.array(nearest) + step_len * direction)
        
        if not (0 <= new_node[0] < grid.shape[0] and 0 <= new_node[1] < grid.shape[1]):
            continue

        t_value = traversability_map[int(new_node[1]), int(new_node[0])]
        if t_value > TRAV_LIMIT:
            continue

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

        nodes.append(new_node)
        parents[new_node] = best_parent
        costs[new_node] = best_cost
        kdtree = KDTree(nodes)

        neighbor_indices = kdtree.query_ball_point(new_node, neighbor_radius)
        neighbors = [nodes[i] for i in neighbor_indices if nodes[i] != new_node and nodes[i] != best_parent]

        for nbr in neighbors:
            t_nbr = traversability_map[int(nbr[1]), int(nbr[0])]
            direct_cost = costs[new_node] + distance(new_node, nbr) * (1.0 + t_nbr)
            if direct_cost < costs[nbr]:
                parents[nbr] = new_node
                costs[nbr] = direct_cost

        if distance(new_node, goal) < step_len:
            t_goal = traversability_map[int(goal[1]), int(goal[0])]
            goal_cost = costs[new_node] + distance(new_node, goal) * (1.0 + t_goal)
            parents[goal] = new_node
            costs[goal] = goal_cost
            nodes.append(goal)
            break

    if goal not in parents:
        return [], float('inf')

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    return path[::-1], costs[goal]

def filter_points(points, traversability_map, threshold=0.6):
    points = np.array(points)
    x_coords, y_coords = points[:, 0], points[:, 1]
    
    boundary_mask = (
        (x_coords >= 20) & (x_coords < traversability_map.shape[1]) &
        (y_coords >= 20) & (y_coords < traversability_map.shape[0])
    )
    
    valid_points = points[boundary_mask]
    
    if len(valid_points) == 0:
        return valid_points
    
    x_valid, y_valid = valid_points[:, 0], valid_points[:, 1]
    traversability_mask = traversability_map[y_valid.astype(int), x_valid.astype(int)] <= threshold
    
    return valid_points[traversability_mask]

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dae_path = os.path.join(project_root, "worlds", "paper_non_convex.dae")
    
    scale = (1.0, 1.0, 0.5)
    elevation, grid_x, grid_y, grid_z = extract_elevation_from_dae(dae_path, scale)
    rows, cols = elevation.shape
    clip_index = 0
    elevation = elevation[clip_index:, clip_index:]
    traversability_map = compute_traversability_map(elevation)
    
    grid_size = 8
    x1 = np.linspace(25, 95, grid_size)
    y1 = np.linspace(14, 95, grid_size)
    X1, Y1 = np.meshgrid(x1, y1)
    start_points = [(int(x), int(y)) for x, y in zip(X1.ravel(), Y1.ravel())]
    goal_means = [(35, 75), (40, 85), (50, 33), (87, 66), (65, 30), (83, 90)]
    
    goal_config = {
        (38, 75): {'num_points': 50, 'x_var': (-5, 5), 'y_var': (-15, 6)},
        (35, 55): {'num_points': 30, 'x_var': (-5, 5), 'y_var': (-5, 5)},
        (80, 38): {'num_points': 40, 'x_var': (-5, 5), 'y_var': (-6, 6)},
        (87, 70): {'num_points': 50, 'x_var': (-5, 5), 'y_var': (-5, 22)},
        (64, 40): {'num_points': 40, 'x_var': (-5, 5), 'y_var': (-6, 6)},
        (85, 93): {'num_points': 50, 'x_var': (-5, 5), 'y_var': (-10, 22)}
    }

    goal_points = []
    for mean_point, config in goal_config.items():
        mean_x, mean_y = mean_point
        points = [(
            mean_x + np.random.randint(*config['x_var']), 
            mean_y + np.random.randint(*config['y_var'])
        ) for _ in range(config['num_points'])]
        goal_points.extend(points)
    
    filtered_start_points = filter_points(start_points, traversability_map, threshold=0.6)
    filtered_goal_points = filter_points(goal_points, traversability_map)
    start_points = [tuple(point) for point in filtered_start_points]
    goal_points = [tuple(point) for point in filtered_goal_points]
    
    results = []
    counter = 0
    cost_matrix = []
    total_paths = len(start_points) * len(goal_points)
    for start in start_points:
        costs = []
        for goal in goal_points:
            counter += 1
            print(f"\rProgress: {counter}/{total_paths} ({counter/total_paths*100:.1f}%)", end='', flush=True)
            if goal == start:
                path, cost = [], float(0)
            else:
                #path, cost = rrt_star(start, goal, elevation, traversability_map) # for rrt_star
                path, cost = [], distance(start, goal)
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
    csv_cost_path = os.path.join(project_root, "data", "rrt_star_costs.csv")
    results_df.to_csv(csv_result_path, index=False)
    cost_df.to_csv(csv_cost_path, index=False, header=False)

    cost_matrix = np.array(cost_matrix)
    updated_matrix2 = cost_matrix.copy()
    num_iterations = 4
    exemplar_indices = []
    for _ in range(num_iterations):
        row_averages = np.mean(updated_matrix2, axis=1)
        for idx in exemplar_indices:
            row_averages[idx] = np.inf
        exemplar_index = np.argmin(row_averages)
        exemplar_indices.append(exemplar_index)
        exemplar_row = updated_matrix2[exemplar_index]
        for i in range(updated_matrix2.shape[0]):
            if i not in exemplar_indices:
                updated_matrix2[i] = np.minimum(updated_matrix2[i], exemplar_row)
    
    print(f"Exemplar Indices: {exemplar_indices}")
    fig, ax2 = plt.subplots(figsize=(8, 6))
    grid_size = 100 - clip_index
    grid_x = np.linspace(0, 100, grid_size)
    grid_y = np.linspace(0, 100, grid_size)
    contour2 = ax2.contourf(grid_x, grid_y, traversability_map, levels=100, cmap='viridis')
    ax2.scatter(*zip(*goal_points), color='black', s=5, label='Target')
    ax2.scatter(*zip(*start_points), color='green', s=50, marker='.', label='Candidate')
    selected_start_points = [start_points[i] for i in exemplar_indices]
    ax2.scatter(*zip(*selected_start_points), color='red', s=50, marker='x', label='Deployed Points')
    cbar = fig.colorbar(contour2, ax=ax2, pad=0.01, shrink=1, aspect=20)
    cbar.set_label("Traversability", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim([20, 99])
    ax2.set_ylim([18, 99])
    legend = ax2.legend(loc='lower center', fontsize=15, frameon=True, ncol=2)
    legend.get_frame().set_alpha(0.5)
    plt.show()
