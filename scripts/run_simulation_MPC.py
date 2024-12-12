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

def quintic_trajectory(start, end, T, dt=0.01):
    """
    Generates a quintic trajectory between two points in 3D space.

    Parameters:
        start (dict): A dictionary with keys `position`, `velocity`, and `acceleration`, each containing
                      a list or numpy array of size 3 (x, y, z components).
        end (dict): A dictionary with the same keys as `start`.
        T (float): The total time for the trajectory.
        dt (float): Time step for the trajectory.

    Returns:
        tuple: A tuple containing three numpy arrays:
               - An array of shape (N, 3), where N is the number of time steps, containing
                 the trajectory points for x, y, and z.
               - An array of shape (N, 3), containing the velocity trajectory for x, y, and z.
               - An array of shape (N, 3), containing the acceleration trajectory for x, y, and z.
    """
    def compute_coefficients(p0, v0, a0, pT, vT, aT, T):
        # Solve for the coefficients of the quintic polynomial
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([p0, v0, a0, pT, vT, aT])
        
        return np.linalg.solve(A, b)

    # Preallocate trajectory
    timesteps = np.arange(0, T + dt, dt)
    trajectory = np.zeros((len(timesteps), 3))
    velocity = np.zeros((len(timesteps), 3))
    acceleration = np.zeros((len(timesteps), 3))

    for i, coord in enumerate(['x', 'y', 'z']):
        p0, v0, a0 = start['position'][i], start['velocity'][i], start['acceleration'][i]
        pT, vT, aT = end['position'][i], end['velocity'][i], end['acceleration'][i]

        # Get the coefficients for the current coordinate
        coeffs = compute_coefficients(p0, v0, a0, pT, vT, aT, T)

        # Evaluate the polynomial, its derivative, and second derivative at each time step
        trajectory[:, i] = (
            coeffs[0] +
            coeffs[1] * timesteps +
            coeffs[2] * timesteps**2 +
            coeffs[3] * timesteps**3 +
            coeffs[4] * timesteps**4 +
            coeffs[5] * timesteps**5
        )

        velocity[:, i] = (
            coeffs[1] +
            2 * coeffs[2] * timesteps +
            3 * coeffs[3] * timesteps**2 +
            4 * coeffs[4] * timesteps**3 +
            5 * coeffs[5] * timesteps**4
        )

        acceleration[:, i] = (
            2 * coeffs[2] +
            6 * coeffs[3] * timesteps +
            12 * coeffs[4] * timesteps**2 +
            20 * coeffs[5] * timesteps**3
        )

    return trajectory, velocity, acceleration


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
        bvh=bvh,
        use_bvh=use_bvh,
        max_iter=5000,
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
    #waypoints = np.array(path)
    #waypoints = np.array([[1.0,1.0,0.5],[1.5,1.5,0.5]])
    #waypoint_idx = 0
    target_speed = 1.0  
    action = np.zeros((1, 4))
    start_point = {
    'position': start_pos,
    'velocity': [0, 0, 0],
    'acceleration': [0, 0, 0]
    }
    end_point = {
        'position': [5.0, 5.0, 1.0],
        'velocity': [0, 0, 0],
        'acceleration': [0, 0, 0]
    }

    trajectory, velocity, acceleration = quintic_trajectory(start_point, end_point, 0.5)
    trajectory_idx = 0
    # Run the simulation
    for i in range(num_steps):
        start_time = time.time()

        current_pos = obs[0][0:3]
        if trajectory_idx < len(trajectory):
            action = np.zeros(9)
            action[0:3] = trajectory[trajectory_idx]  # [x, y, z]
            action[3:6] = velocity[trajectory_idx]  # [dx, dy, dz]
            action[6:9] = acceleration[trajectory_idx]  # [ddx, ddy, ddz]

            pos_error = action[0:3]  - current_pos
            distance = np.linalg.norm(pos_error)
            print(f"Distance: {distance}")
            if distance < 0.2:
                trajectory_idx += 1
                print(f"Waypoint {trajectory_idx}/{len(trajectory)}")
            
        else:

            action[0:3] = trajectory[-1]  # [x, y, z]
            action[3:6] = np.array([0.0,0.0,0.0])  # [dx, dy, dz]
            action[6:9] = np.array([0.0,0.0,0.0])  # [ddx, ddy, ddz]


        obs, reward, terminated, truncated, info = env.step(action)


        logger.log(
            drone=0,
            timestamp=i * env.CTRL_TIMESTEP,
            state=obs[0],
            control=np.hstack([action[0:3] , np.zeros(9)])
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
    print(f"Elapsed planner time: {elapsed_time:.4f} seconds")
    #logger.save()
    #logger.save_as_csv("simulation_rrt_star")

if __name__ == "__main__":
    main()
