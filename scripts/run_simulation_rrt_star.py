# import numpy as np
# import time
# import pybullet as p
# import sys
# import os

# # Add the parent directory of 'environments' to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from environments.custom_aviaries.MPC_static_factory_aviary import StaticFactoryMPC
# from planners.rrt_star_planner import RRTStarPlanner
# from gym_pybullet_drones.utils.enums import DroneModel, Physics
# from gym_pybullet_drones.utils.Logger import Logger
# from bvh.bvh import BVHNode, build_bvh
# from control.MPCController import Simple_MPC, Linearized_Drone_MPC


# def main():
#     duration_sec = 50  
#     simulation_freq_hz = 240
#     control_freq_hz = 48
#     num_steps = int(duration_sec * control_freq_hz)
#     gui = True

#     env = StaticFactoryMPC(
#         drone_model=DroneModel.CF2X,
#         num_drones=1,
#         physics=Physics.PYB,
#         neighbourhood_radius=np.inf,
#         initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
#         initial_rpys=np.array([[0.0, 0.0, 0.0]]),
#         pyb_freq=simulation_freq_hz,
#         ctrl_freq=control_freq_hz,
#         gui=gui,
#         record=False,
#         obstacles=True,
#         user_debug_gui=False,
#         obstacle_config={
#             'num_obstacles': 1,
#             'obstacle_radius': 0.3,
#             'arena_size': 10.0
#         },
#         seed=40
#     )
#     obs, info = env.reset()

#     logger = Logger(logging_freq_hz=control_freq_hz, num_drones=1)

#     obstacles = []
#     for obs_id in env.obstacle_ids:
#         pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=env.CLIENT)
#         radius = p.getVisualShapeData(obs_id, physicsClientId=env.CLIENT)[0][3][0]  # Extract the sphere radius

#         aabb_min = np.array(pos) - np.array([radius, radius, radius])
#         aabb_max = np.array(pos) + np.array([radius, radius, radius])

#         obstacles.append({'aabb_min': aabb_min, 'aabb_max': aabb_max})
    
#     bvh = build_bvh(obstacles)
#     use_bvh = True


#     arena_size = env.obstacle_config['arena_size']
#     x_range = [-arena_size/2, arena_size/2]
#     y_range = [-arena_size/2, arena_size/2]
#     z_range = [0.5, 2.0]

#     goal_x = np.random.uniform(-arena_size / 2, arena_size / 2)
#     goal_y = np.random.uniform(-arena_size / 2, arena_size / 2)
#     goal_z = np.random.uniform(0.5, 2.5)
#     goal_pos = np.array([goal_x, goal_y, goal_z])

#     goal_radius = 0.1
#     goal_color = [0.0, 1.0, 0.0, 1.0]
#     goal_visual = p.createVisualShape(
#         shapeType=p.GEOM_SPHERE,
#         radius=goal_radius,
#         rgbaColor=goal_color,
#         physicsClientId=env.CLIENT
#     )
#     p.createMultiBody(
#         baseMass=0,
#         baseVisualShapeIndex=goal_visual,
#         basePosition=goal_pos,
#         physicsClientId=env.CLIENT
#     )

#     start_pos = np.copy(env.pos[0])

#     # Initialize the RRT* planner
#     planner = RRTStarPlanner(
#         start=start_pos,
#         goal=goal_pos,
#         obstacles=obstacles,
#         x_range=x_range,
#         y_range=y_range,
#         z_range=z_range,
#         bvh=bvh,
#         use_bvh=use_bvh,
#         max_iter=5000,
#         step_size=0.2,
#         goal_sample_rate=0.01,
#         search_radius=0.5,
#     )
    

#     # Plan the path
#     start_time = time.time() 
#     path = planner.plan()
#     end_time = time.time()  # Record the end time
#     elapsed_time = end_time - start_time
#     print(f"Elapsed planner time: {elapsed_time:.4f} seconds")
#     # Visualize the path
#     for i in range(len(path) - 1):
#         p.addUserDebugLine(
#             lineFromXYZ=path[i],
#             lineToXYZ=path[i+1],
#             lineColorRGB=[1, 0, 0],
#             lifeTime=0,
#             physicsClientId=env.CLIENT
#         )

#     if path is None:
#         print("Failed to find a path!")
#         env.close()
#         return
    

#     # Prepare for simulation
#     #waypoints = np.array(path)
#     waypoints = np.array([[0.0,0.0,1.2],[0.5,0.5,1.2],[0.5,0.5,1.2]])
#     waypoint_idx = 0
#     target_speed = 1.0  
#     action = np.zeros((1, 4))
#     # Cost function and constraints
#     # Q = np.diag([10, 10, 10, 1, 1, 1])  # State weights
#     # R = np.diag([0.1, 0.1,0.05])  # Input weights

#     # MPC = Simple_MPC(Q=Q, R=R)
#     Q = np.diag([10, 10, 10, 1, 1, 1, 5, 5, 5, 0.1, 0.1, 0.1])
#     R = np.diag([1, 1, 1, 0.1])

#     MPC = Linearized_Drone_MPC(Q=Q, R=R, dt=1/control_freq_hz)

#     # Run the simulation
#     for i in range(num_steps):
#         start_time = time.time()

#         current_pos = obs[0][0:3]
#         current_vel = obs[0][3:6]
#         if waypoint_idx < len(waypoints):
#             target_pos = waypoints[waypoint_idx]
#             pos_error = target_pos - current_pos
#             distance = np.linalg.norm(pos_error)

#             # Move to the next waypoint if close enough
#             if distance < 0.2:
#                 waypoint_idx += 1
#                 continue   
            
#             direction = pos_error / distance
#             speed = min(distance, env.SPEED_LIMIT)
#             velocity_command = direction * speed

#             speed = min(distance, env.SPEED_LIMIT)
#             current_state = np.hstack([current_pos, current_vel, np.zeros(6)])
#             print(current_state.shape)
#             des_velocity = speed * (pos_error / distance)
#             des_state = np.hstack([target_pos, np.zeros(9)])
#             print(des_state.shape)
#             u_opt, _ = MPC.compute_mpc_control(
#                 x0=current_state,
#                 x_ref=des_state,
#                 N=5
#             )
#             print("Type of u_opt:", type(u_opt))
#             print("Contents of u_opt:", u_opt)

#             u_opt = np.array(u_opt).reshape(-1, 1)
#             state_dot = MPC.A @ current_state + MPC.B @ u_opt
#             print("State dot:", state_dot[:,0])
#             thrust_vector = state_dot[8:,0]  # Assuming thrust_vector comes from these indices
#             thrust_vector = thrust_vector.flatten()  # Ensure 1D

#             # Create use_MPC flag as a NumPy array
#             use_MPC = np.array([False])  # Shape (1,)

#             # Perform horizontal stacking for action
#             action[0, :] = np.hstack((thrust_vector, use_MPC))
#             print("Action:", action)
#             #action[0, :] = np.hstack((velocity_command, [use_MPC]))
            
#         else:
#             use_MPC = False
#             action[0, :] = np.array([0.0, 0.0, 0.0, use_MPC])

#         # for i in range(num_steps):
#         # start_time = time.time()

#         # current_pos = obs[0][0:3]
#         # if waypoint_idx < len(waypoints):
#         #     target_pos = waypoints[waypoint_idx]
#         #     pos_error = target_pos - current_pos
#         #     distance = np.linalg.norm(pos_error)

#         #     # Move to the next waypoint if close enough
#         #     if distance < 0.2:
#         #         waypoint_idx += 1
#         #         continue

#         #     # Compute velocity command
#         #     direction = pos_error / distance
#         #     speed = min(distance, env.SPEED_LIMIT)
#         #     velocity_command = direction * speed
#         #     action[0, :] = np.hstack((velocity_command, [target_speed]))
#         # else:

#         #     action[0, :] = np.array([0.0, 0.0, 0.0, 0.0])


#         obs, reward, terminated, truncated, info = env.step(action)


#         logger.log(
#             drone=0,
#             timestamp=i * env.CTRL_TIMESTEP,
#             state=obs[0],
#             control=np.hstack([target_pos, np.zeros(9)])
#         )


#         print(f"Step {i}, Position: {current_pos}, Waypoint: {waypoint_idx}/{len(waypoints)}")


#         if terminated or truncated:
#             print("Simulation ended")
#             break

#         sleep_time = (1.0 / control_freq_hz) - (time.time() - start_time)
#         if sleep_time > 0:
#             time.sleep(sleep_time)


#     env.close()

#     planner.draw_tree()
#     print(f"Elapsed planner time: {elapsed_time:.4f} seconds")
#     #logger.save()
#     #logger.save_as_csv("simulation_rrt_star")

# if __name__ == "__main__":
#     main()
