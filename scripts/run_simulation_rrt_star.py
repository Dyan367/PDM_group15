import numpy as np
import time
import pybullet as p
from environments.custom_aviaries.static_factory_aviary import StaticFactory
from planners.rrt_star_planner import RRTStarPlanner
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger


def main():
    duration_sec = 50  
    simulation_freq_hz = 240
    control_freq_hz = 48
    num_steps = int(duration_sec * control_freq_hz)
    gui = True

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

    start_pos = env.pos[0]
    goal_pos = np.array([5.0, 5.0, 1.0])


    obstacles = []
    for obs_id in env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=env.CLIENT)
        size = p.getVisualShapeData(obs_id, physicsClientId=env.CLIENT)[0][3]  
        obstacles.append({'position': np.array(pos), 'size': np.array(size) * 2})  


    arena_size = env.obstacle_config['arena_size']
    x_range = [-arena_size/2, arena_size/2]
    y_range = [-arena_size/2, arena_size/2]
    z_range = [0.5, 2.0]

    # Initialize the RRT* planner
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

    # Plan the path
    path = planner.plan()
    if path is None:
        print("Failed to find a path!")
        env.close()
        return

    # Visualize the path
    for i in range(len(path) - 1):
        p.addUserDebugLine(
            lineFromXYZ=path[i],
            lineToXYZ=path[i+1],
            lineColorRGB=[1, 0, 0],
            lifeTime=0,
            physicsClientId=env.CLIENT
        )

    # Prepare for simulation
    waypoints = np.array(path)
    waypoint_idx = 0
    target_speed = 1.0  
    action = np.zeros((1, 4))

    # Run the simulation
    for i in range(num_steps):
        start_time = time.time()

        current_pos = obs[0][0:3]
        if waypoint_idx < len(waypoints):
            target_pos = waypoints[waypoint_idx]
            pos_error = target_pos - current_pos
            distance = np.linalg.norm(pos_error)

            # Move to the next waypoint if close enough
            if distance < 0.2:
                waypoint_idx += 1
                continue

            # Compute velocity command
            direction = pos_error / distance
            speed = min(distance, env.SPEED_LIMIT)
            velocity_command = direction * speed
            action[0, :] = np.hstack((velocity_command, [target_speed]))
        else:

            action[0, :] = np.array([0.0, 0.0, 0.0, 0.0])


        obs, reward, terminated, truncated, info = env.step(action)


        logger.log(
            drone=0,
            timestamp=i * env.CTRL_TIMESTEP,
            state=obs[0],
            control=np.hstack([target_pos, np.zeros(9)])
        )


        print(f"Step {i}, Position: {current_pos}, Waypoint: {waypoint_idx}/{len(waypoints)}")


        if terminated or truncated:
            print("Simulation ended")
            break

        sleep_time = (1.0 / control_freq_hz) - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)


    env.close()

    logger.save()
    logger.save_as_csv("simulation_rrt_star")

if __name__ == "__main__":
    main()
