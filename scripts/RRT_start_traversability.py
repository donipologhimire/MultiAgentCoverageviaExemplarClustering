import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import xml.etree.ElementTree as ET
import random
from scipy.ndimage import maximum_filter, gaussian_gradient_magnitude
import matplotlib.animation as animation


# RRT* Parameters
STEP_LEN = 3.0     # Step size
ITER_MAX = 7000       # Maximum iterations

# Critical Values and Weights for traversability
CRIT_STEP_HEIGHT = 0.3
CRIT_FLATNESS = 0.5236
CRIT_SLOPE = 0.5236
# CRIT_STEP_HEIGHT = 0.24
# CRIT_FLATNESS = 0.24
# CRIT_SLOPE = 0.24



WEIGHT_STEP = 0.4
WEIGHT_FLATNESS = 0.4
WEIGHT_SLOPE = 0.2

# If you want a threshold to mark untraversable
TRAV_LIMIT = 0.63

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
    
    x = scaled_vertices[:, 0]
    y = scaled_vertices[:, 1]
    z = scaled_vertices[:, 2]
    print('x max:', x.max())
    print('y max:', y.max())
    # Create an interpolated grid
    grid_x, grid_y = np.linspace(x.min(), x.max()+70, 100), np.linspace(y.min(), y.max()+70, 100)
    # grid_x, grid_y = np.linspace(x.min(), 110, 100), np.linspace(y.min(), 110, 100)
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
    # Rough measure of step height using a maximum filter.
    # For a more accurate measure, consider using local gradient or difference.
    return maximum_filter(np.abs(elevation), size=size)

def compute_flatness_map(elevation, size=5):
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
    slope = gaussian_gradient_magnitude(elevation, sigma=sigma)
    # slope could range widely. We might want to normalize it.
    # A rough normalization: divide by a constant or use CRIT_SLOPE as a reference.
    return slope

def compute_traversability_map(elevation):
    """Generate a traversability map based on step height, flatness, and slope."""
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
    RRT* Implementation with traversability:
    - Costs incorporate traversability. 
      Cost = cost_to_parent + distance * (1 + traversability_value)
    """
    nodes = [start]
    parents = {start: None}
    costs = {start: 0.0}

    kdtree = KDTree(nodes)
    GOAL_SAMPLE_RATE = 0.10
    for i in range(iter_max):
        # Random sampling
        # if random.random() < GOAL_SAMPLE_RATE:
        #     rand_point = goal  # Force goal to be sampled
        # else:
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
        print("Goal not reached. Try adjusting parameters or weights.")
        return []

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    return path[::-1],parents

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(script_dir)
# Construct path to the worlds directory
dae_path = os.path.join(project_root, "worlds", "custom_world_2.dae")

# Optional: Verify the path exists
if not os.path.exists(dae_path):
    raise FileNotFoundError(f"DAE file not found at: {dae_path}")
# scale = (0.125, 0.125, 0.125)
scale = (1.0, 1.0, 0.5)
# Extract normalized elevation grid


elevation,grid_x,grid_y,grid_z = extract_elevation_from_dae(dae_path,scale)
# Apply clipping condition
rows, cols = elevation.shape  # Get dimensions


clip_index = 10
# Extract the upper-right portion
elevation = elevation[clip_index:, clip_index:]

# Compute traversability map
traversability_map = compute_traversability_map(elevation)
print("size_traversability_map", traversability_map.shape)

print("size_elevation_map", elevation.shape)

# traversability_map = elevation
print("maximum", np.max(traversability_map))

print("minimum", np.min(traversability_map))
print("position of max", np.argmax(traversability_map))

# plot the position of the max value
max_pos = np.unravel_index(np.argmax(elevation, axis=None), traversability_map.shape)
min_pos = np.unravel_index(np.argmin(elevation, axis=None), traversability_map.shape)
#plot max_pos   

# Define start and goal points

start = (70, 45)
goal = (70, 80)
start2 = (50,45)


# Run RRT* algorithm with traversability
path,tree1 = rrt_star(start, goal, elevation, traversability_map,iter_max=1000)
path2,tree2 = rrt_star(start2, goal, elevation, traversability_map,iter_max=4000)

# print("path",path[-1])
# Create grid_x and grid_y with 100 points each
grid_size = 100-clip_index
grid_x = np.linspace(0, 100, grid_size)
grid_y = np.linspace(0, 100, grid_size)
# Plot the results


# === Visualization Functions ===
def create_base_plot(grid_x, grid_y, data, title, cmap='viridis'):
    """Create a base plot with common settings"""
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(grid_x, grid_y, data, levels=35, cmap=cmap)
    
    # Remove ticks and set limits
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([10, 98])
    ax.set_ylim([10, 98])
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax, pad=0.02, shrink=1, aspect=20)
    cbar.set_label(title, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    return fig, ax, contour

def plot_tree_and_paths(ax, tree1, tree2, path1, path2, start, start2, goal):
    """Plot the RRT* trees and paths"""
    # Plot trees
    for tree in [tree1, tree2]:
        for node, parent in tree.items():
            if parent is not None:
                ax.plot([node[0], parent[0]], [node[1], parent[1]], 
                       color='white', linewidth=1.30, alpha=0.2)
    
    # Plot paths and points
    if path1:
        ax.plot(*zip(*path1), color="red", linewidth=2, label="Path 1")
    if path2:
        ax.plot(*zip(*path2), color="darkred", linewidth=2, label="Path 2")
    
    ax.scatter(*start, color="red", marker='*', s=100, label="Start 1")
    ax.scatter(*start2, color="darkred", marker='d', s=100, label="Start 2")
    ax.scatter(*goal, color="g", s=100, label="Goal")
    ax.legend(loc='upper right', fontsize=10, frameon=True)

def create_animation(grid_x, grid_y, traversability_map, path1, path2, start, start2, goal):
    """Create an animation of both robots moving along their paths"""
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(grid_x, grid_y, traversability_map, levels=35, cmap='viridis')
    
    # Initialize plot elements
    path_line1, = ax.plot([], [], color="red", linewidth=2, label="path 1")
    path_line2, = ax.plot([], [], color="darkred", linewidth=2, label="path 2")
    robot1_point, = ax.plot([], [], 'r*', markersize=12, label="Robot 1")
    robot2_point, = ax.plot([], [], 'rd', markersize=12, label="Robot 2")
    
    # Plot start and goal points
    ax.scatter(*start, color="red", marker='*', s=100, alpha=0.5)
    ax.scatter(*start2, color="darkred", marker='d', s=100, alpha=0.5)
    ax.scatter(*goal, color="green", s=100)
    
    # Animation setup
    max_frames = max(len(path1), len(path2))
    
    def init():
        path_line1.set_data([], [])
        path_line2.set_data([], [])
        robot1_point.set_data([], [])
        robot2_point.set_data([], [])
        return path_line1, path_line2, robot1_point, robot2_point

    def update(frame):
        # Update path 1
        if frame < len(path1):
            path1_x, path1_y = zip(*path1[:frame+1])
            robot1_x, robot1_y = path1[frame]
            path_line1.set_data(path1_x, path1_y)
            robot1_point.set_data([robot1_x], [robot1_y])
        
        # Update path 2
        if frame < len(path2):
            path2_x, path2_y = zip(*path2[:frame+1])
            robot2_x, robot2_y = path2[frame]
            path_line2.set_data(path2_x, path2_y)
            robot2_point.set_data([robot2_x], [robot2_y])
        
        return path_line1, path_line2, robot1_point, robot2_point

    ax.set_xlim([10, 98])
    ax.set_ylim([10, 98])
    ax.legend(loc='upper right')
    
    anim = animation.FuncAnimation(
        fig, update, frames=max_frames,
        init_func=init, blit=True, interval=50
    )
    
    return anim

# === Main Execution ===
if __name__ == "__main__":
    # ...existing code until plotting section...

    # Create elevation plot
    fig1, ax1, _ = create_base_plot(grid_x, grid_y, elevation, "Elevation")
    plot_tree_and_paths(ax1, tree1, tree2, path, path2, start, start2, goal)

    # Create traversability plot
    fig2, ax2, _ = create_base_plot(grid_x, grid_y, traversability_map, "Traversability")
    plot_tree_and_paths(ax2, tree1, tree2, path, path2, start, start2, goal)

    # Create and save animation
    anim = create_animation(grid_x, grid_y, traversability_map, path, path2, start, start2, goal)
    
    # Save animation (uncomment to save)
    # anim.save('dual_robot_path.gif', writer='pillow', fps=20)
    
    plt.show()
