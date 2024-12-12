import numpy as np
import time
from PDM_group15.environments.custom_aviaries.MPC_static_factory_aviary import StaticFactory
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger

def main():

    duration_sec = 20 
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
            'num_obstacles': 4,
            'obstacle_size': [2, 0.5, 10.0],
            'arena_size': 7.0
        },
        seed=40
    )
    obs, info = env.reset()


    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=1)


    target_pos = np.array([5.0, 5.0, 1.0])
    target_speed = 1.0  
    action = np.zeros((1, 4))

    # Run the simulation
    START = time.time()
    for i in range(num_steps):
        start_time = time.time()

        current_pos = obs[0][0:3]
        pos_error = target_pos - current_pos
        distance = np.linalg.norm(pos_error)

        if distance > 0.1:
            direction = pos_error / distance
            speed = min(distance, env.SPEED_LIMIT)
            velocity_command = direction * speed
 
            action[0, :] = np.hstack((velocity_command, [target_speed]))
        else:
            action[0, :] = np.array([0.0, 0.0, 0.0, 0.0])

        obs, reward, terminated, truncated, info = env.step(action)

        # Log the simulation data
        logger.log(
            drone=0,
            timestamp=i * env.CTRL_TIMESTEP,
            state=obs[0],
            control=np.hstack([target_pos, np.zeros(9)])
        )

        # Print simulation info
        print(f"Step {i}, Position: {current_pos}, Action: {action[0]}, Terminated: {terminated}")


        if terminated or truncated:
            print("Simulation ended")
            break


        sleep_time = (1.0 / control_freq_hz) - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    logger.save()
    logger.save_as_csv("simulation_static_factory")

if __name__ == "__main__":
    main()
