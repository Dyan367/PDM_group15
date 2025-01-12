import numpy as np
import time
import pybullet as p
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.custom_aviaries.static_factory_aviary import StaticFactory
from planners.rrt_star_plannerV2 import RRTStarPlannerV2
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from planners.bvh_tree import build_bvh


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
            'environment_width': 10.0,
            'environment_height': 10.0,
            'wall_thickness': 1.0,
            'wall_height': 1.0,
            'cell_size':1.0
        },
        seed=42
    )
    obs, info = env.reset()

    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=1)

    start_pos = env.pos[0]
    goal_pos = np.array([6.5, 6.5, 5.5])



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
            rgbaColor=[1, 0, 0, 0.05],  # Green color with 30% opacity
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

    start_pos = np.copy(env.pos[0])

    # Initialize the RRT* planner
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

        for obs_id in env.obstacle_ids:
            contact_points = p.getContactPoints(bodyA=env.DRONE_IDS[0], bodyB=obs_id)
            if contact_points:
                print(f"Collision detected with obstacle ID {obs_id}")

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


        #print(f"Step {i}, Position: {current_pos}, Waypoint: {waypoint_idx}/{len(waypoints)}")


        if terminated or truncated:
            print("Simulation ended")
            break

        sleep_time = (1.0 / control_freq_hz) - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)


    env.close()

    planner.draw_tree()

    logger.save()
    logger.save_as_csv("simulation_rrt_star")

if __name__ == "__main__":
    main()