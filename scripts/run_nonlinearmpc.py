import numpy as np
import time
import pybullet as p
import sys
import os
# Add the parent directory of 'environments' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.custom_aviaries.static_factory_aviary import StaticFactory
from planners.rrt_star_planner import RRTStarPlanner
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from bvh.bvh import BVHNode, build_bvh
from control.MPCController import Simple_MPC,Linear_MPC
from scipy.integrate import solve_ivp
from nonlinearmpc import mpc_controller, generate_full_state_trajectory_from_array, mpc_to_motor_rpm
import minsnap_trajectories as ms
from scipy.spatial.transform import Rotation

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
            'num_obstacles': 1,
            'obstacle_radius': 0.3,
            'arena_size': 10.0
        },
        seed=40
    )
    obs, info = env.reset()

    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=1)

    obstacles = []
    for obs_id in env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=env.CLIENT)
        radius = p.getVisualShapeData(obs_id, physicsClientId=env.CLIENT)[0][3][0]  # Extract the sphere radius

        aabb_min = np.array(pos) - np.array([radius, radius, radius])
        aabb_max = np.array(pos) + np.array([radius, radius, radius])

        obstacles.append({'aabb_min': aabb_min, 'aabb_max': aabb_max})
    
    bvh = build_bvh(obstacles)
    use_bvh = True


    arena_size = env.obstacle_config['arena_size']
    x_range = [-arena_size/2, arena_size/2]
    y_range = [-arena_size/2, arena_size/2]
    z_range = [0.5, 2.0]

    goal_x = np.random.uniform(-arena_size / 2, arena_size / 2)
    goal_y = np.random.uniform(-arena_size / 2, arena_size / 2)
    goal_z = np.random.uniform(0.5, 2.5)
    goal_pos = np.array([goal_x, goal_y, goal_z])

    

    goal_radius = 0.1
    goal_color = [0.0, 1.0, 0.0, 1.0]
    goal_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=goal_radius,
        rgbaColor=goal_color,
        physicsClientId=env.CLIENT
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_visual,
        basePosition=goal_pos,
        physicsClientId=env.CLIENT
    )

    start_pos = np.copy(env.pos[0])

    # Initialize the RRT* planner
    planner = RRTStarPlanner(
        start=start_pos,
        goal=goal_pos,
        obstacles=obstacles,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        max_iter=1000,
        bvh=bvh,
        use_bvh=use_bvh,
        step_size=0.2,
        goal_sample_rate=0.01,
        search_radius=0.5,
    )

    # Plan the path
    start_time = time.time() 
    path = planner.plan()
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed planner time: {elapsed_time:.4f} seconds")
    # Visualize the path
    for i in range(len(path) - 1):
        p.addUserDebugLine(
            lineFromXYZ=path[i],
            lineToXYZ=path[i+1],
            lineColorRGB=[1, 0, 0],
            lifeTime=0,
            physicsClientId=env.CLIENT
        )



    if path is None:
        print("Failed to find a path!")
        env.close()
        return

    # Prepare for simulation
    waypoints = np.array(path)
    waypoint_idx = 0
    action = np.zeros((1, 4))
    total_duration = 10.0
    vehicle_mass = 0.027

    trajectory = generate_full_state_trajectory_from_array(
        waypoints, total_duration, control_freq_hz, vehicle_mass
    )

    positions = trajectory.position
    velocities = trajectory.velocity
    attitudes = trajectory.attitude
    body_rates = trajectory.body_rates
    

    attitudes = Rotation.from_quat(attitudes).as_euler('xyz')

    ## HERE TUNE MPC PARAMETERS and initialie MPC class

    # Run the simulation
    for i in range(num_steps):
        start_time = time.time()

        x0 = np.hstack([
            obs[0][0:3],       # x, y, z
            obs[0][7:10],     # roll, pitch, yaw
            obs[0][10:13],       # velocity x, y, z
            obs[0][13:16]    # angular velocities p, q, r
        ])

        
        target_pos = waypoints[waypoint_idx]
        
        current_time_index = min(i, len(positions) - 1)
        # x_ref = np.hstack([
        #     positions[current_time_index],  # x, y, z
        #     attitudes[current_time_index], # roll, pitch, yaw
        #     velocities[current_time_index],  # velocity x, y, z
        #     body_rates[current_time_index]   # angular velocities p, q, r
        # ])

        x_ref = np.hstack([[0,0,1],np.zeros(9)])
        
        # Solve the MPC problem
        solution = mpc_controller(x0, x_ref)
        optimal_controls = solution['x'][-4:].full().flatten()  # Extract motor speed controls from solution
        print("Optimal controls: ", optimal_controls)
        rpms = mpc_to_motor_rpm(optimal_controls)

        # Apply the action
        print("RPMs: ", rpms)
        action[0, :] = rpms
        
        
        obs, reward, terminated, truncated, info = env.step(rpms)
        

        logger.log(
            drone=0,
            timestamp=i * env.CTRL_TIMESTEP,
            state=obs[0],
            control=np.hstack([target_pos, np.zeros(9)])
        )
        print(f"Step {i}, Position: {obs[0][0:3]}, Waypoint: {waypoint_idx}/{len(waypoints)}")


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
    print(f"Elapsed planner time: {elapsed_time:.4f} seconds")
    #logger.save()
    #logger.save_as_csv("simulation_rrt_star")

if __name__ == "__main__":
    main()