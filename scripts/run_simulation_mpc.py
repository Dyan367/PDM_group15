
import os
from sys import platform
import time
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from environments.custom_aviaries.MPCAviary_static import MPCAviaryStatic

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
import cvxpy as cp
import matplotlib.pyplot as plt
import logging

from planners.rrt_star_planner import RRTStarPlanner  


def waypoint_to_x_target(waypoint):
    """
    Converts a 3D waypoint [x, y, z] into the 12D target format used by MPCAviaryStatic:
    [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz].
    Velocity/orientation are set to zero.
    """
    x_target = np.zeros(12)
    x_target[:3] = waypoint  # Position (x, y, z)
    # All other components (velocity, orientation, angular velocity) = 0
    return x_target


if __name__ == "__main__":

    # Define MPC parameters
    mpc_params = {
        'dt': 0.1,          # Time step (seconds)
        'N': 10,            # Prediction horizon (steps)
        'sim_time': 5000,    # Total simulation time (seconds)
        'proximity_threshold': 0.01  # meters
    }

    # Define the initial and final 3D waypoints for the RRT* planner
    start_pos = np.array([0.0, 0.0, 1.0])  # This will be updated from env.pos if needed
    goal_pos = np.array([5.0, 5.0, 1.0])

    # Define obstacle configuration
    obstacle_config = {
        'num_obstacles': 6,
        'obstacle_size': [2, 0.5, 10.0],
        'arena_size': 10.0
    }

    # Initialize the MPCAviary environment
    env = MPCAviaryStatic(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        neighbourhood_radius=np.inf,
        initial_xyzs=np.array([[0, 0, 1]]),  # Starting 1 meter above the ground
        initial_rpys=np.array([[0, 0, 0]]),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=True,            # Set to False for faster/non-graphical simulation
        record=False,
        obstacles=True,      # Enable obstacle addition
        user_debug_gui=False,
        vision_attributes=False,
        output_folder='results',
        mpc_params=mpc_params,
        x_target=None,               # We'll override with waypoints
        obstacle_config=obstacle_config,
        seed=42
    )

    # Reset the environment
    obs, info = env.reset()

    # Update start position from environment if desired
    # The drone's position is stored in env.pos[0], which is [x, y, z]
    start_pos = np.copy(env.pos[0])

    # Extract obstacle info from PyBullet
    obstacles = []
    for obs_id in env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=env.CLIENT)
        # The size is half-extents in PyBullet, so multiply by 2 for full dimension
        shape_data = p.getVisualShapeData(obs_id, physicsClientId=env.CLIENT)[0]
        half_extents = shape_data[3]  # (x, y, z) half-extents
        full_size = np.array(half_extents) * 2
        obstacles.append({'position': np.array(pos), 'size': full_size})

    # Define RRT* search space
    arena_size = env.obstacle_config['arena_size']
    x_range = [-arena_size / 2, arena_size / 2]
    y_range = [-arena_size / 2, arena_size / 2]
    z_range = [0.5, 2.0]  # you can adjust as needed

    # Initialize the RRT* planner
    planner = RRTStarPlanner(
        start=start_pos,
        goal=goal_pos,
        obstacles=obstacles,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        max_iter=1000,
        step_size=0.5,
        goal_sample_rate=0.1,
        search_radius=1.0
    )

    # Compute the path
    path = planner.plan()
    if path is None:
        print("Failed to find a path!")
        env.close()
        exit()

    # Draw the path with PyBullet debug lines
    for i in range(len(path) - 1):
        p.addUserDebugLine(
            lineFromXYZ=path[i],
            lineToXYZ=path[i+1],
            lineColorRGB=[1, 0, 0],
            lifeTime=0,  # 0 = infinite
            physicsClientId=env.CLIENT
        )

    # Convert path to array for easy indexing
    waypoints = np.array(path)
    waypoint_idx = 0
    waypoint_threshold = 0.2  # distance threshold to switch to next waypoint

    # Set the initial MPC target to the first waypoint
    env.set_target(waypoint_to_x_target(waypoints[waypoint_idx]))

    total_steps = int(mpc_params['sim_time'] / mpc_params['dt'])
    for step in range(total_steps):
        obs, reward, terminated, truncated, info = env.step()

        # --- Check if we've reached the current waypoint ---
        current_pos = obs[:3]  # [x, y, z] from the observation
        dist_to_waypoint = np.linalg.norm(current_pos - waypoints[waypoint_idx])

        if dist_to_waypoint < waypoint_threshold:
            # Move on to the next waypoint if possible
            waypoint_idx += 1
            if waypoint_idx >= len(waypoints):
                print(f"Reached final waypoint at step {step}, time {step * mpc_params['dt']}s")
                break
            else:
                env.set_target(waypoint_to_x_target(waypoints[waypoint_idx]))
                print(f"Switching to waypoint {waypoint_idx}: {waypoints[waypoint_idx]}")

        # --- Check if the environment decided we've reached the final x_target ---
        # (e.g., due to the internal proximity_threshold)
        # if terminated:
        #     print(f"Simulation terminated at step {step}, time {step * mpc_params['dt']}s")
        #     break
        if np.linalg.norm(current_pos - goal_pos) < 0.1:
            terminated = True
            print(f"Goal reached! Distance to goal: {np.linalg.norm(current_pos - goal_pos):.4f} meters")
            break  # Exit the loop if goal is reached

    # Plot the results
    env.plot_results()
    env.close()
