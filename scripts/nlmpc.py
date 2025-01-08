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
from control.MPCController import Simple_MPC

##
# The class rpm_calc is used to calculate the RPMs of the motors using MPC
from experimental.linearized_drone_mpc import rpm_calc, RPMCalcCasADi
from experimental.trajectory_generation import generate_reference_states
from scipy.spatial.transform import Rotation
# The class minsnap_trajectories is used to generate the reference states for the drone using min snap trajectory generation
import minsnap_trajectories as ms
# The class hierarchical_control uses 2 MPCs to control the position and attitude of the drone by computing the thrust and torques then converting them to RPMs
from experimental.hierarchical_control import PositionMPC, AttitudeMPC, compute_desired_angular_velocity
import casadi as ca
from scipy.interpolate import interp1d
from experimental.nonlinearmpc import NonlinearMPC

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
    #goal_pos = np.array([2.0, 0.0, 1.0])

    

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
    #waypoints = np.array([waypoints[0], waypoints[-1]])
    waypoint_idx = 0
    target_speed = 1.0  
    action = np.zeros((1, 4))

##############################################################################################################################################################################
# TRAJECTORY GENERATION
##############################################################################################################################################################################

# set drag parameters for trajectory generation
    drag_params = ms.RotorDragParameters(
        cp=0.000001,   # drag_xy_coeff
        dh=2267.18,    # dw_coeff_1
        dv=0.16,       # dw_coeff_2
    )

## generates the reference states for the drone using min snap trajectory generation look at trajectory_generation.py for more details
# I am not 100% sure that this function creates a trajectory that has 0.0 yaw angle so this could be looked into
    reference_states = generate_reference_states(
    waypoints=waypoints,
    set_speed=0.5,  
    vehicle_mass=0.027000,
    drag_params=drag_params,#drag_params,
    yaw= "velocity",  # Fixed yaw angle
    yaw_rate=None  # Fixed yaw angle
)

    # Extract results
    positions = reference_states["positions"]
    velocities = reference_states["velocities"]
    attitudes = reference_states["attitudes"]
    
    attitudes = Rotation.from_quat(attitudes).as_euler('xyz', degrees=False)
    
    time_samples = reference_states["time_samples"]
    angular_velocities =reference_states["angular_velocities"]
    

    pos_interp = interp1d(time_samples, positions, axis=0, kind='linear')
    vel_interp = interp1d(time_samples, velocities, axis=0, kind='linear')
    att_interp = interp1d(time_samples, attitudes, axis=0, kind='linear')
    ang_vel_interp = interp1d(time_samples, angular_velocities, axis=0, kind='linear')
    
##############################################################################################################################################################################
# VISUALIZE THE TRAJECTORY
##############################################################################################################################################################################

    for i in range(len(positions) - 1):
        p.addUserDebugLine(
            lineFromXYZ=positions[i],
            lineToXYZ=positions[i + 1],
            lineColorRGB=[0, 0, 1],  # Red line
            lifeTime=0  # Persistent lines
        )

##############################################################################################################################################################################
# MPC INITIALIZATION
##############################################################################################################################################################################
    #MPC WEIGHTS FOR RPM CALCULATION
    Q_rpm = np.diag([
        50, 50, 50,       # Position weights
        30, 30, 30,    # Velocity weights
        1, 1, 1,        # Attitude weights
        1, 1, 1      # Angular velocity weights (scaled down)
    ])

    R_rpm = np.diag([1, 0.01, 0.01, 0.01])  # Higher priority on smooth inputs

    # Initialize the MPC controller with the desired state
    #rpm_mpc = RPMCalcCasADi(Q=Q_rpm, R=R_rpm, N=5, dt=env.CTRL_TIMESTEP)
    #rpm_mpc = rpm_calc(Q=Q_rpm, R=R_rpm, N=5, dt=env.CTRL_TIMESTEP)

    pos_mpc = PositionMPC(dt = env.CTRL_TIMESTEP, horizon=30)

    att_mpc = AttitudeMPC(dt = env.CTRL_TIMESTEP, horizon=30)
    # ## HERE TUNE MPC PARAMETERS and initialie MPC class
    # Q = np.diag([10, 10, 10, 1, 1, 1])  # State weights
    # R = np.diag([0.1, 0.1,0.1])  # Input weights
    # MPC = Simple_MPC(Q=Q, R=R)


    
    
##############################################################################################################################################################################
# SIMULATION LOOP
##############################################################################################################################################################################
    # Initialize Nonlinear MPC
    mpc = NonlinearMPC(dt=env.CTRL_TIMESTEP, horizon=30)

    for i in range(num_steps):
        start_time = time.time()
        state_vector = env._getDroneStateVector(0)
        current_pos = state_vector[0:3]
        current_vel = state_vector[10:13]
        current_rpy = state_vector[7:10]
        current_omega = state_vector[13:16]
        x0 = np.hstack([current_pos, current_vel, current_rpy, current_omega])

        elapsed_time = i * env.CTRL_TIMESTEP
        ref_positions = np.vstack([pos_interp(elapsed_time + t * env.CTRL_TIMESTEP) for t in range(mpc.horizon + 1)])
        ref_velocities = np.vstack([vel_interp(elapsed_time + t * env.CTRL_TIMESTEP) for t in range(mpc.horizon + 1)])
        ref_attitudes = np.vstack([att_interp(elapsed_time + t * env.CTRL_TIMESTEP) for t in range(mpc.horizon + 1)])
        ref_angular_velocities = np.vstack([ang_vel_interp(elapsed_time + t * env.CTRL_TIMESTEP) for t in range(mpc.horizon + 1)])
        ref_states = np.hstack([ref_positions, ref_velocities, ref_attitudes, ref_angular_velocities]).T

        optimal_u = mpc.solve(x0, ref_states)
        action = np.array([optimal_u])
        print(f"Optimal control: {optimal_u}")

        obs, _, terminated, truncated, _ = env.step(action)
        #logger.log(drone=0, timestamp=i * env.CTRL_TIMESTEP, state=obs[0], control=optimal_u)

        if terminated or truncated:
            break
        sleep_time = (1.0 / control_freq_hz) - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)



    env.close()

    planner.draw_tree()
    logger.save()
    # logger.save_as_csv("simulation_rrt_star")
    print(f"Elapsed planner time: {elapsed_time:.4f} seconds")
    #logger.save()
    #logger.save_as_csv("simulation_rrt_star")

if __name__ == "__main__":
    main()