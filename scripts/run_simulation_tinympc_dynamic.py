import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
import logging

from gym_pybullet_drones.utils.enums import DroneModel, Physics

from environments.custom_aviaries.MPCAviary_tinympc_dynamic import MPCAviaryDynamicTinyMPC

from planners.rrt_star_plannerV2 import RRTStarPlannerV2


def waypoint_to_x_target(waypoint, current_position, max_velocity=5.0):
    x_target = np.zeros(12)
    x_target[:3] = waypoint
    distance_vector = waypoint - current_position
    distance = np.linalg.norm(distance_vector)
    if distance > 0:
        desired_velocity = (distance_vector / distance) * min(distance, max_velocity)
    else:
        desired_velocity = np.zeros(3)
    x_target[3:6] = desired_velocity
    return x_target

def find_closest_waypoint(current_pos, waypoints, start_idx=0):
    """
    Find the index of the closest waypoint to the current position.

    Args:
        current_pos (np.array): Current position of the drone [x, y, z].
        waypoints (np.array): List of waypoints (Nx3).
        start_idx (int): The starting index to search for the closest waypoint.

    Returns:
        int: Index of the closest waypoint.
    """
    # Compute distances to all remaining waypoints
    distances = np.linalg.norm(waypoints[start_idx:] - current_pos, axis=1)
    # Find the index of the closest waypoint
    closest_idx = np.argmin(distances) + start_idx
    return closest_idx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mpc_params = {
        'dt': 0.02,
        'N': 100,
        'sim_time': 5000,
        'proximity_threshold': 0.01
    }

    start_pos = np.array([0.0, 0.0, 1.0])
    goal_pos = np.array([20, 1, 2.0])

    obstacle_config = {
        'num_obstacles': 24,
        'obstacle_size': [2, 0.5, 10.0],
        'arena_size': 10.0
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
        gui=True,
        record=False,
        obstacles=True,
        user_debug_gui=False,
        vision_attributes=False,
        output_folder='results',
        mpc_params=mpc_params,
        x_target=None,
        obstacle_config=obstacle_config,
        seed=42
    )

    obs, info = env.reset()
    start_pos = env.pos[0].copy()

    obstacles = []
    for obs_id in env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=env.CLIENT)
        shape_data = p.getVisualShapeData(obs_id, physicsClientId=env.CLIENT)[0]
        half_extents = shape_data[3]
        full_size = np.array(half_extents) * 2
        obstacles.append({'position': np.array(pos), 'size': full_size})

    arena_size = env.obstacle_config['arena_size']
    x_range = [-arena_size / 2, arena_size / 2]
    y_range = [-arena_size / 2, arena_size / 2]
    z_range = [0.5, 2.0]

    print(env.obstacles_info)

    planner = RRTStarPlannerV2(
        start=start_pos,
        goal=goal_pos,
        obstacles_info=env.obstacles_info,
        x_range=[-30.0, 30.0],
        y_range=[-30.0, 30.0],
        z_range=z_range,
        max_iter=2500,
        step_size=0.2,
        goal_sample_rate=0.3,
        search_radius=10
    )

    path = planner.plan()
    if path is None:
        print("Failed to find a path!")
        env.close()
        exit()

    for i in range(len(path) - 1):
        p.addUserDebugLine(
            lineFromXYZ=path[i],
            lineToXYZ=path[i + 1],
            lineColorRGB=[1, 0, 0],
            lifeTime=0,
            physicsClientId=env.CLIENT
        )

    waypoints = np.array(path)
    waypoint_idx = 1

    waypoint_threshold = 0.5

    env.set_target(waypoint_to_x_target(waypoints[waypoint_idx], start_pos))

    total_steps = int(mpc_params['sim_time'] / mpc_params['dt'])
    # Initialize variables
    goal_threshold = 0.1  # Threshold for reaching the goal
    action = np.zeros((1, 4))  # Initialize the action array (e.g., motor RPMs)

    # Set the initial target to the first waypoint
    env.set_target(waypoint_to_x_target(waypoints[waypoint_idx], start_pos))

    # Simulation loop
    for step in range(total_steps):
        # Step the environment (action can be passed if needed)
        obs, reward, terminated, truncated, info = env.step(action)

        # Get the drone's current position
        current_pos = obs[:3]

        # Check proximity to the current waypoint
        if np.linalg.norm(current_pos - waypoints[waypoint_idx]) < waypoint_threshold:
            waypoint_idx += 1  # Move to the next waypoint
            if waypoint_idx >= len(waypoints):  # Check if all waypoints are completed
                print(f"Reached final waypoint at step {step}, time {step * mpc_params['dt']:.2f}s")
                break

            # Set the new target for the MPC
            new_target = waypoint_to_x_target(waypoints[waypoint_idx], current_pos)
            env.set_target(new_target)
            print(f"Switching to waypoint {waypoint_idx}: {waypoints[waypoint_idx]}")

        # Dynamically find the closest waypoint (for debugging or visualization)
        closest_idx = find_closest_waypoint(current_pos, waypoints)

        # Draw a red line to the closest waypoint
        p.addUserDebugLine(
            lineFromXYZ=current_pos,
            lineToXYZ=waypoints[closest_idx],
            lineColorRGB=[0, 0, 1],  # Red color
            lifeTime=0.02,
            physicsClientId=env.CLIENT
        )

        # Check if the drone is close enough to the goal
        if np.linalg.norm(current_pos - goal_pos) < goal_threshold:
            print(f"Goal reached at step {step}, time {step * mpc_params['dt']:.2f}s")
            break

        # Log progress
        print(f"Step {step}, Position: {current_pos}, Current Target: {waypoints[waypoint_idx]}")

        # Handle termination or truncation
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    # Plot results and close the environment
    env.plot_results()
    env.close()
