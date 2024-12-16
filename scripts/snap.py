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
from experimental.linearized_drone_mpc import rpm_calc
from experimental.trajectory_generation import generate_reference_states
from scipy.spatial.transform import Rotation
# The class minsnap_trajectories is used to generate the reference states for the drone using min snap trajectory generation
import minsnap_trajectories as ms
# The class hierarchical_control uses 2 MPCs to control the position and attitude of the drone by computing the thrust and torques then converting them to RPMs
from experimental.hierarchical_control import hierarchical_control

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
    total_time=15.0,  
    vehicle_mass=0.027000,
    drag_params=drag_params,
    yaw= None,
    yaw_rate=None  # Fixed yaw angle
)

    # Extract results
    positions = reference_states["positions"]
    velocities = reference_states["velocities"]
    attitudes = reference_states["attitudes"]
    time_samples = reference_states["time_samples"]
    angular_velocities =reference_states["angular_velocities"]

    # # Trajectory generation does not provide angular velocities and accelerations, so we need to compute them that is done here
    # desired_angular_velocities = []
    # #desired_angular_accelerations = []

    # prev_quaternion = attitudes[0]  # Initialize with the first quaternion
    # prev_angular_velocity = np.zeros(3)  # Initialize with zero angular velocity

    # # Loop over desired trajectory samples
    # for i in range(len(positions)):
    #     # Extract desired position, velocity, and attitude (quaternion)
    #     desired_pos = positions[i]
    #     desired_vel = velocities[i]
    #     desired_quaternion = attitudes[i]  # Quaternion: [x, y, z, w]

    #     if i > 0:  # Skip the first step
    #         # Compute delta time
    #         dt = time_samples[i] - time_samples[i - 1]

    #         # Convert quaternions to rotation objects
    #         current_rot = Rotation.from_quat(desired_quaternion)
    #         previous_rot = Rotation.from_quat(prev_quaternion)

    #         # Compute the relative rotation
    #         delta_rot = current_rot * previous_rot.inv()
    #         delta_angle = delta_rot.as_rotvec()  # Convert to angle-axis representation

    #         # Angular velocity = delta angle / delta time
    #         angular_velocity = angular_velocity = np.zeros(3) #delta_angle / dt

    #         # Compute angular acceleration (alpha = d(angular_velocity)/dt)
    #         #angular_acceleration = (angular_velocity - prev_angular_velocity) / dt
    #     else:
    #         angular_velocity = np.zeros(3)  # Zero angular velocity for the first step
    #         #angular_acceleration = np.zeros(3)  # Zero angular acceleration for the first step

    #     # Store the angular velocity and angular acceleration
    #     desired_angular_velocities.append(angular_velocity)
    #     #desired_angular_accelerations.append(angular_acceleration)

    #     # Update the previous quaternion and angular velocity
    #     prev_quaternion = desired_quaternion
    #     prev_angular_velocity = angular_velocity

    # # Convert to NumPy arrays
    # desired_angular_velocities = np.array(desired_angular_velocities)
    
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
        100, 100, 100,       # Position weights
        1000, 1000, 1000,    # Velocity weights
        1000, 1000, 1000, # Attitude weights
        1, 1, 1 # Angular velocity weights (scaled down)
    ])

    R_rpm = np.diag([0.01, 0.01, 0.01, 0.01])  # Higher priority on smooth inputs

    rpm_mpc = rpm_calc(Q=Q_rpm, R=R_rpm, N=5)

    # ## HERE TUNE MPC PARAMETERS and initialie MPC class
    # Q = np.diag([10, 10, 10, 1, 1, 1])  # State weights
    # R = np.diag([0.1, 0.1,0.1])  # Input weights
    # MPC = Simple_MPC(Q=Q, R=R)


    ## HERE TUNE HIERARCHICAL CONTROLLER PARAMETERS
    Q_pos = np.diag([100, 100, 100, 1, 1, 1])  # State weights for position control MPC
    R_pos = np.diag([0.1, 1, 1]) 

    Q_attitude = np.diag([100, 100, 100, 1, 1, 1])  # State weights for attitude control MPC
    R_attitude = np.diag([0.01, 0.01, 0.01]) 
    # Initialize the hierarchical controller

    hierarchical_controller = hierarchical_control(Q_pos=Q_pos, R_pos=R_pos, Q_attitude=Q_attitude, R_attitude=R_attitude, N=5)

    
##############################################################################################################################################################################
# SIMULATION LOOP
##############################################################################################################################################################################
    time_index = 0
    for i in range(num_steps):
        start_time = time.time()
        
        # get the current state of the drone with id 0
        current_state_vector = env._getDroneStateVector(0)
        current_pos = current_state_vector[0:3]
        current_vel = current_state_vector[10:13]
        angular_velocity_current = current_state_vector[13:16]
        curr_rpy = current_state_vector[7:10]

        if time_index != len(positions) - 1:
            # Compute the desired state at the current time index
            elapsed_time = i * env.CTRL_TIMESTEP  # Calculate time since the start of the simulation

            # Update desired position based on the time index (trajectory following)
            if elapsed_time >= time_samples[time_index + 1]:
                time_index += 1

            # get the trajectory at the current time index
            desired_pos = positions[time_index]
            desired_vel = velocities[time_index]
            desired_attitude = attitudes[time_index]
            angular_velocity = angular_velocities[time_index]  # Precomputed angular velocity

            # Convert quaternion to roll, pitch, yaw
            roll, pitch, yaw = p.getEulerFromQuaternion(desired_attitude)

           #yaw = 0.0  # Fixed yaw angle

            # DESIRED STATE
            desired_state = np.hstack([desired_pos, desired_vel, roll, pitch, yaw, angular_velocity])

            # VISUALIZES THE DESIRED POSITION in CYAN
            p.addUserDebugLine(
                lineFromXYZ=desired_pos,
                lineToXYZ=[desired_pos[0], desired_pos[1], 0],  # Connect to ground for reference
                lineColorRGB=[0, 1, 1],  
                lineWidth=2.0,           # Thicker line
                lifeTime=1/env.CTRL_TIMESTEP  # Persistent for one step
            )

            # Current state of the quadrotor (obtain angular velocity from PyBullet if available)
            current_state = np.hstack([current_pos, current_vel, curr_rpy, angular_velocity_current])

            ###################################################################################################
            # PRETTY PRINT STATEMENT FOR DEBUGGING
            ###################################################################################################
            state_labels = [
                "x (Position)", "y (Position)", "z (Position)",        # Positions
                "vx (Velocity)", "vy (Velocity)", "vz (Velocity)",    # Velocities
                "Roll", "Pitch", "Yaw",                               # Attitude (Euler angles)
                "wx (Angular Velocity)", "wy (Angular Velocity)", "wz (Angular Velocity)"  # Angular velocities
            ]

            # Print Desired State
            print("\n--- Desired State ---")
            for label, value in zip(state_labels, desired_state):
                print(f"{label:<25}: {value}")

            # Print Current State
            print("\n--- Current State ---")
            for label, value in zip(state_labels, current_state):
                print(f"{label:<25}: {value}")
            ###################################################################################################
            # PRETTY PRINT STATEMENT FOR DEBUGGING
            ###################################################################################################

            ###################################################################################################
            # MPC SOLVING FOR RPMS
            ###################################################################################################
            # SINGLE MPC
            rpms, predicted_states, _, _ = rpm_mpc.solve_mpc(current_state, desired_state)

            # 2 MPC CONTROLLERS ONE FOR POSITION THE OTHER FOR ATTITUDE
            #thrust, tau, predicted_states = hierarchical_controller.compute_mpc(current_state, desired_state)
            #rpms = hierarchical_controller.u2rpms(thrust, tau)

            # Print RPMs
            print("\n--- Computed RPMs ---")
            for i, rpm in enumerate(rpms, 1):
                print(f"Motor {i} RPM               : {rpm:.1f}")

            # MAKE SURE TO ADJUST DSLIPOSITIONCONTROL BY COMMENTING OUT COMPUTATIONS IN .computeControl() IN VELOCITYAVIARY.PY SUCH THAT IT RETURNS ACTION = RPMS
            action[0, :] = rpms #np.hstack((u_opt, [target_speed]))
            
        else:
            
            rpms = np.zeros(4)

            action[0, :] = rpms


        # visualize the MPC predictions (scaled!)
        scaling_factor = 0.5  

        for j in range(predicted_states.shape[1] - 1):
            start_point = predicted_states[:3, j]
            next_point = predicted_states[:3, j + 1]

            # Compute the direction vector and scale it
            direction = next_point - start_point
            direction_normalized = direction / np.linalg.norm(direction)  # Normalize the direction vector
            scaled_point = start_point + direction_normalized * scaling_factor  # Scale the line

            # Draw the scaled line
            p.addUserDebugLine(
                lineFromXYZ=start_point,
                lineToXYZ=scaled_point,
                lineColorRGB=[0, 1, 0],
                lineWidth = 2.0,  # Green color for the prediction
                lifeTime=env.CTRL_TIMESTEP,  # Keep the line until the next update
                physicsClientId=env.CLIENT
            )
        

        obs, reward, terminated, truncated, info = env.step(action)
        


        logger.log(
            drone=0,
            timestamp=i * env.CTRL_TIMESTEP,
            state=obs[0],
            control=np.hstack([desired_pos, np.zeros(9)])
        )


        print(f"Step {i}, Position: {current_pos}, Waypoint: {waypoint_idx}/{len(waypoints)}")

        # for body_id in sphere_bodies:
        #     p.removeBody(body_id)

        # # Clear the list for the next step
        # sphere_bodies = []


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