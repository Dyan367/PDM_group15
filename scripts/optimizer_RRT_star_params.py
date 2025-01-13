import time
import numpy as np
import pandas as pd
import pybullet as p

from environments.custom_aviaries.MPCAviary_tinympc_dynamic2 import MPCAviaryDynamicTinyMPC
from planners.rrt_star_plannerV2 import RRTStarPlannerV2
from planners.bvh_tree import build_bvh
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Define the grid of parameters to test
parameter_grid = {
    "iterations": [20000,50000,100000,250000],
    "step_size": [0.1,0.2,0.50,0.75],
    "goal_sample_rate": [0.25,0.5],
    "search_radius": [1.0 , 2.0]
}

# Create an empty list to store results
results = []
#optimal path found through graphical search
optimal_path = [
    [0.5, 0.5, 1],
    [1.4, 1.1, 0.9],
    [6.1, 1.9, 0.9],
    [6.1, 5.1, 0.9],
    [5.1, 6.1, 0.9],
    [1.9, 6.1, 0.9],
    [1.9, 4.9, 0.9],
    [2.1, 4.1, 2.1],
    [1.9, 3.1, 2.5],
    [1.9, 2.1, 2.9],
    [1.9, 2.1, 4.1],
    [3.1, 1.9, 4.19086229210592],
    [3.1, 3.1, 4.28172458421184],
    [1.9, 3.9, 4.41022327001633],
    [1.9, 5.1, 4.50108556212225],
    [6, 6.3, 4.9],
    [6.5, 6.5, 5.5]
]

# Environment Configuration
mpc_params = {
    'dt': 0.02,
    'N': 100,
    'sim_time': 5000,
    'proximity_threshold': 0.01
}

env = MPCAviaryDynamicTinyMPC(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    neighbourhood_radius=np.inf,
    initial_xyzs=np.array([[0, 0, 1]]),
    initial_rpys=np.array([[0, 0, 0]]),
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=240,
    gui=False,
    record=False,
    obstacles=True,
    user_debug_gui=False,
    vision_attributes=False,
    output_folder='results',
    mpc_params=mpc_params,
    x_target=None,
    obstacle_config={
        'environment_width': 10.0,
        'environment_height': 10.0,
        'wall_thickness': 1.0,
        'wall_height': 1.0,
        'cell_size': 1.0
    },
    seed=42
)

# Reset the environment and retrieve obstacle AABBs
obs, info = env.reset()
aabbs = []
dilation = 0.1  # Dilation amount for safety margin

for obs_id in env.obstacle_ids:
    # Get the AABB for the obstacle
    aabb_min, aabb_max = p.getAABB(obs_id, physicsClientId=env.CLIENT)

    # Convert to numpy arrays and dilate the AABB
    aabb_min = np.array(aabb_min) - dilation
    aabb_max = np.array(aabb_max) + dilation

    # Append the dilated AABB
    aabbs.append({'aabb_min': aabb_min, 'aabb_max': aabb_max})

# Build the BVH tree for collision checking
bvh_tree = build_bvh(aabbs)

# Define start, goal, and search space
start_pos = env.pos[0].copy()
goal_pos = np.array([6.5, 6.5, 5.5])
x_range = [0.1, 8.0]
y_range = [0.1, 8.0]
z_range = [0.1, 8.0]

# Optimal path length (for evaluating optimality)
optimal_path_length = 29.868

# Perform grid search over RRT* parameters
for iterations in parameter_grid["iterations"]:
    for step_size in parameter_grid["step_size"]:
        for goal_sample_rate in parameter_grid["goal_sample_rate"]:
            for search_radius in parameter_grid["search_radius"]:
                print(f"Testing parameters: Iterations={iterations}, Step Size={step_size}, "
                      f"Goal Sample Rate={goal_sample_rate}, Search Radius={search_radius}")

                # Timer: RRT* Planning
                start_time = time.perf_counter()
                planner = RRTStarPlannerV2(
                    start=start_pos,
                    goal=goal_pos,
                    bvh_tree=bvh_tree,
                    x_range=x_range,
                    y_range=y_range,
                    z_range=z_range,
                    max_iter=iterations,
                    step_size=step_size,
                    goal_sample_rate=goal_sample_rate,
                    search_radius=search_radius
                )
                path = planner.plan()
                runtime = time.perf_counter() - start_time

                # Evaluate results
                if path is not None:
                    # Calculate path length
                    waypoints = np.array(path)
                    path_length = 0
                    for i in range(1, len(waypoints)):
                        distance = np.linalg.norm(waypoints[i] - waypoints[i - 1])
                        path_length += distance

                    # Calculate optimality
                    optimality = (optimal_path_length / path_length) * 100
                    completeness = 1  # Path found
                else:
                    path_length = float("inf")
                    optimality = 0
                    completeness = 0  # Path not found

                # Log results
                results.append({
                    "Iterations": iterations,
                    "Step Size": step_size,
                    "Goal Sample Rate": goal_sample_rate,
                    "Search Radius": search_radius,
                    "Completeness": completeness,
                    "Optimality (%)": round(optimality, 2),
                    "Runtime (s)": round(runtime, 2)
                })

                planner.draw_tree(optimal_path=optimal_path)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("rrt_star_parameter_optimization.csv", index=False)
print("Optimization complete. Results saved to rrt_star_parameter_optimization.csv")

# Close the environment
env.close()

# Print results
print(results_df)