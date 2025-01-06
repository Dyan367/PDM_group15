import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
import logging

from gym_pybullet_drones.utils.enums import DroneModel, Physics

# This environment must contain the exact same _addObstacles() as your TinyMPC code:
# (Adjust the import path if needed. Example file name:
#  "MPCAviary_static_dompc_same_obstacles.py".)
from environments.custom_aviaries.MPCAviary_static_dompc import MPCAviaryStaticDoMPC

# If you have a local RRTStarPlanner:
from planners.rrt_star_planner import RRTStarPlanner


def waypoint_to_x_target(waypoint, current_position, max_velocity=5.0):
    """
    Convert [x, y, z] waypoint -> 12D reference state = position + velocity.
    Identical to your TinyMPC approach.
    """
    x_target = np.zeros(12)
    x_target[:3] = waypoint
    dist_vec = waypoint - current_position
    dist = np.linalg.norm(dist_vec)
    if dist > 0:
        desired_vel = (dist_vec / dist) * min(dist, max_velocity)
    else:
        desired_vel = np.zeros(3)
    x_target[3:6] = desired_vel
    return x_target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Identical MPC parameters as your TinyMPC script
    mpc_params = {
        'dt': 0.02,
        'N': 100,
        'sim_time': 5000,
        'proximity_threshold': 0.01
    }

    # Start & goal positions
    start_pos = np.array([0.0, 0.0, 1.0])
    goal_pos  = np.array([5.0, 5.0, 1.0])

    # Obstacle config that matches your TinyMPC shelves
    obstacle_config = {
        'num_obstacles': 6,
        'obstacle_size': [2, 0.5, 10.0],  # same as your shelves
        'arena_size': 10.0
    }

    # Create do-mpc environment that copies the TinyMPC _addObstacles
    env = MPCAviaryStaticDoMPC(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        neighbourhood_radius=np.inf,
        initial_xyzs=np.array([[0, 0, 1]]),
        initial_rpys=np.array([[0, 0, 0]]),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=True,               # Toggle if you want GUI
        record=False,
        obstacles=True,         # same as TinyMPC
        user_debug_gui=False,
        vision_attributes=False,
        output_folder='results',
        mpc_params=mpc_params,
        x_target=None,          # we'll set it from waypoints
        obstacle_config=obstacle_config,
        seed=42                 # critical for matching obstacle placement
    )

    # Reset and retrieve environment's initial state
    obs, info = env.reset()
    # Store the actual drone start pos from environment
    start_pos = env.pos[0].copy()

    # If using RRT*:
    obstacles_data = []
    for obs_id in env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=env.CLIENT)
        shape_data = p.getVisualShapeData(obs_id, physicsClientId=env.CLIENT)[0]
        half_extents = shape_data[3]
        full_size = np.array(half_extents)*2
        obstacles_data.append({'position': np.array(pos), 'size': full_size})

    # Same arena bounds
    arena_size = env.obstacle_config['arena_size']
    x_range = [-arena_size/2, arena_size/2]
    y_range = [-arena_size/2, arena_size/2]
    z_range = [0.5, 2.0]

    planner = RRTStarPlanner(
        start=start_pos,
        goal=goal_pos,
        obstacles=obstacles_data,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        max_iter=1000,
        step_size=0.5,
        goal_sample_rate=0.1,
        search_radius=1.0
    )

    path = planner.plan()
    if path is None:
        print("Failed to find a path!")
        env.close()
        exit()

    # Visualize path in PyBullet
    for i in range(len(path)-1):
        p.addUserDebugLine(
            lineFromXYZ=path[i],
            lineToXYZ=path[i+1],
            lineColorRGB=[1,0,0],
            lifeTime=0,
            physicsClientId=env.CLIENT
        )

    waypoints = np.array(path)
    waypoint_idx = 0
    waypoint_threshold = 0.3

    # Set the initial target
    env.set_target(waypoint_to_x_target(waypoints[waypoint_idx], start_pos))

    # Main simulation loop
    total_steps = int(mpc_params['sim_time'] / mpc_params['dt'])
    for step in range(total_steps):
        obs, reward, terminated, truncated, info = env.step()
        current_pos = obs[:3]

        # Check distance to current waypoint
        dist_wp = np.linalg.norm(current_pos - waypoints[waypoint_idx])
        if dist_wp < waypoint_threshold:
            waypoint_idx += 1
            if waypoint_idx >= len(waypoints):
                print(f"Reached final waypoint at step {step}, t={step*mpc_params['dt']:.2f}s")
                break
            else:
                new_target = waypoint_to_x_target(waypoints[waypoint_idx], current_pos)
                env.set_target(new_target)
                print(f"Switching to waypoint {waypoint_idx}: {waypoints[waypoint_idx]}")

        # Check final goal
        if np.linalg.norm(current_pos - goal_pos) < 0.1:
            print(f"Goal reached! Distance to goal = {np.linalg.norm(current_pos - goal_pos):.4f}")
            break

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    # Plot final results
    env.plot_results()
    # Close environment
    env.close()
