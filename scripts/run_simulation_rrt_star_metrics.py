from http.cookiejar import split_header_words

import numpy as np
import time
import pybullet as p
from environments.custom_aviaries.static_factory_aviary import StaticFactory
from planners.rrt_star_planner import RRTStarPlanner
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("TkAgg")

def main():
    duration_sec = 50
    simulation_freq_hz = 240
    control_freq_hz = 48
    num_steps = int(duration_sec * control_freq_hz)
    gui = True

    # Initialize environment
    env = StaticFactory(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.PYB,
        neighbourhood_radius=np.inf,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=True,
        user_debug_gui=False,
        obstacle_config={
            'num_obstacles': 6,
            'obstacle_size': [2, 0.5, 10.0],
            'arena_size': 10.0
        },
        seed=40
    )
    obs, info = env.reset()
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=1)

    # Metrics Initialization
    metrics = {
        "planning_time": 0,
        "control_times": [],
        "execution_time": 0,
        "control_inputs": [],
        "success": False,
        "cumulative_control": 0,
        "max_control_input": 0,
        "cumulative_jerk": 0,
        "previous_input": None  # None initially, will store the previous input vector
    }

    # Planning
    start_pos = env.pos[0]
    goal_pos = np.array([5.0, 5.0, 1.0])

    obstacles = []
    for obs_id in env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=env.CLIENT)
        size = p.getVisualShapeData(obs_id, physicsClientId=env.CLIENT)[0][3]
        obstacles.append({'position': np.array(pos), 'size': np.array(size) * 2})

    arena_size = env.obstacle_config['arena_size']
    x_range = [-arena_size / 2, arena_size / 2]
    y_range = [-arena_size / 2, arena_size / 2]
    z_range = [0.5, 2.0]

    start_pos = np.copy(env.pos[0])

    planner = RRTStarPlanner(
        start=start_pos,
        goal=goal_pos,
        obstacles=obstacles,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        max_iter=5000,
        step_size=0.5,
        goal_sample_rate=0.1,
        search_radius=1.0
    )

    # Plan path
    start_time = time.time()
    path = planner.plan()
    metrics["planning_time"] = time.time() - start_time

    if path is None:
        print("Failed to find a path!")
        env.close()
        return

    for i in range(len(path) - 1):
        p.addUserDebugLine(
            lineFromXYZ=path[i],
            lineToXYZ=path[i + 1],
            lineColorRGB=[1, 0, 0],
            lifeTime=0,
            physicsClientId=env.CLIENT
        )

    waypoints = np.array(path)
    waypoint_idx = 0
    target_speed = 4.0
    action = np.zeros((1, 4))
    planner.draw_tree(show=False)
    # Execution
    actual_path = []
    start_execution = time.time()

    for i in range(num_steps):
        control_start = time.time()

        current_pos = obs[0][0:3]
        actual_path.append(current_pos)
        if waypoint_idx < len(waypoints):
            target_pos = waypoints[waypoint_idx]
            pos_error = target_pos - current_pos
            distance = np.linalg.norm(pos_error)

            if distance < 0.2:
                waypoint_idx += 1
                continue

            direction = pos_error / distance
            speed = min(distance, env.SPEED_LIMIT)
            velocity_command = direction * speed
            action[0, :] = np.hstack((velocity_command, [target_speed]))
        else:
            action[0, :] = np.array([0.0, 0.0, 0.0, 0.0])

        obs, reward, terminated, truncated, info = env.step(action)
        metrics["control_inputs"].append(action[0, :])
        metrics["control_times"].append(time.time() - control_start)

        # Calculate Control Metrics
        control_magnitude = np.linalg.norm(action[0, :3])  # Ignore the last element if it's not part of the velocity
        metrics["cumulative_control"] += control_magnitude
        metrics["max_control_input"] = max(metrics["max_control_input"], control_magnitude)

        if metrics["previous_input"] is not None:
            jerk = np.linalg.norm(action[0, :3] - metrics["previous_input"])
            metrics["cumulative_jerk"] += jerk
        else:
            # Skip jerk computation for the first step
            metrics["previous_input"] = action[0, :3]

        logger.log(
            drone=0,
            timestamp=i * env.CTRL_TIMESTEP,
            state=obs[0],
            control=np.hstack([target_pos, np.zeros(9)])
        )

        if terminated or truncated:
            print("Simulation ended")
            break

        if waypoint_idx >= len(waypoints):
            metrics["success"] = True
            break

        sleep_time = (1.0 / control_freq_hz) - (time.time() - control_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    metrics["execution_time"] = time.time() - start_execution

    # Clean up
    env.close()
    logger.save()
    logger.save_as_csv("simulation_rrt_star")

    # Print Metrics
    print("\n--- Metrics ---")
    print(f"Planning Time: {metrics['planning_time']:.4f} seconds")
    print(f"Execution Time: {metrics['execution_time']:.4f} seconds")
    print(f"Success: {metrics['success']}")
    print("\n--- Control Metrics ---")
    print(f"Total Control Effort: {metrics['cumulative_control']:.4f}")
    print(f"Max Control Input: {metrics['max_control_input']:.4f}")
    print(f"Cumulative Jerk: {metrics['cumulative_jerk']:.4f}")
    # Convert to NumPy arrays
    actual_path = np.array(actual_path)
    planned_path = np.array(path)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the planned path
    ax.plot(planned_path[:, 0], planned_path[:, 1], planned_path[:, 2], 'r--', label="Planned Path")

    # Plot the actual path
    ax.plot(actual_path[:, 0], actual_path[:, 1], actual_path[:, 2], 'b-', label="Actual Path")

    # Plot start and goal points
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', label="Start", s=50)
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color='red', label="Goal", s=50)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Drone Path (Planned vs. Actual)")
    ax.legend()

    # Show or save the plot
    plt.show()

if __name__ == "__main__":
    main()
