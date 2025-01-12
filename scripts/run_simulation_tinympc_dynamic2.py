import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
import logging

from gym_pybullet_drones.utils.enums import DroneModel, Physics
import sys
import os

# Add the parent directory of 'environments' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.custom_aviaries.MPCAviary_tinympc_dynamic2 import MPCAviaryDynamicTinyMPC
#from environments.custom_aviaries.MPCAviary_tinympc_dynamic3 import MPCAviaryDynamicTinyMPC

from planners.rrt_star_plannerV2 import RRTStarPlannerV2
from planners.bvh_tree import build_bvh


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

    start_pos = np.array([0.0, 0.0, 0.3])
    goal_pos = np.array([6.5, 6.5, 5.5])

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
        obstacle_config={
            'environment_width': 10.0,
            'environment_height': 10.0,
            'wall_thickness': 1.0,
            'wall_height': 1.0,
            'cell_size':1.0
        },
        seed=42
    )

    obs, info = env.reset()
    start_pos = env.pos[0].copy()

    aabbs = []
    dilation = 0.1  # Dilation amount

    for obs_id in env.obstacle_ids:
        # Get the AABB for the obstacle
        aabb_min, aabb_max = p.getAABB(obs_id, physicsClientId=env.CLIENT)

        # Convert to numpy arrays
        aabb_min = np.array(aabb_min)
        aabb_max = np.array(aabb_max)

        # Dilate the AABB
        aabb_min -= dilation
        aabb_max += dilation

        # Append the dilated AABB as a dictionary
        aabbs.append({'aabb_min': aabb_min, 'aabb_max': aabb_max})

    for aabb in aabbs:
        aabb_min = aabb['aabb_min']
        aabb_max = aabb['aabb_max']

        # Calculate center and extent
        center = (aabb_min + aabb_max) / 2
        extent = (aabb_max - aabb_min) / 2

        # Create a transparent visual shape
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=extent,
            rgbaColor=[1, 0, 0, 0.0],  # Green color with 5% opacity
            physicsClientId=env.CLIENT
        )

        # Create the body with only the visual shape (no collision or dynamics)
        p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=center,
            physicsClientId=env.CLIENT
        )

    bvh_tree = build_bvh(aabbs)

    arena_size = env.obstacle_config['environment_width']  # Updated to match new obstacle config
    # x_range = [-arena_size / 2, arena_size / 2]
    # y_range = [-arena_size / 2, arena_size / 2]
    x_range = [0.1, 8.0]
    y_range = [0.1, 8.0]
    z_range = [0.1, 8.0]

    goal_sphere_radius = 0.2  # Radius of the sphere
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=goal_sphere_radius,
        rgbaColor=[0, 0, 0, 1],
        physicsClientId=env.CLIENT
    )

    p.createMultiBody(
        baseVisualShapeIndex=visual_shape_id,
        basePosition=goal_pos.tolist(),  # Position the sphere at the goal position
        physicsClientId=env.CLIENT
    )

    planner = RRTStarPlannerV2(
        start=start_pos,
        goal=goal_pos,
        bvh_tree=bvh_tree,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        max_iter=100000,
        step_size=0.2,
        goal_sample_rate=0.2,
        search_radius=1.0
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
            lineColorRGB=[0, 0, 1],
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
        for obs_id in env.obstacle_ids:
            contact_points = p.getContactPoints(bodyA=env.DRONE_IDS[0], bodyB=obs_id)
            if contact_points:
                print(f"Collision detected with obstacle ID {obs_id}")

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

        # Draw a blue line to the closest waypoint
        p.addUserDebugLine(
            lineFromXYZ=current_pos,
            lineToXYZ=waypoints[closest_idx],
            lineColorRGB=[0, 0, 1],  # blue color
            lifeTime=0.02,
            physicsClientId=env.CLIENT
        )

        # Check if the drone is close enough to the goal
        if np.linalg.norm(current_pos - goal_pos) < goal_threshold:
            print(f"Goal reached at step {step}, time {step * mpc_params['dt']:.2f}s")
            break

        # Log progress
        #print(f"Step {step}, Position: {current_pos}, Current Target: {waypoints[waypoint_idx]}")

        # Handle termination or truncation
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    # Plot results and close the environment
    env.plot_results()
    env.close()

planner.draw_tree()
