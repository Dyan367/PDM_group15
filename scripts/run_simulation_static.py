import numpy as np
import time
from environments.custom_aviaries.static_factory_aviary import StaticFactory
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def main():

    env = StaticFactory(drone_model=DroneModel.CF2X,
                                num_drones=1,
                                physics=Physics.PYB,
                                gui=True,
                                obstacles=False,
                                obstacle_config={'num_obstacles': 2,
                                                 'obstacle_size': [0.5, 0.5, 10.0],
                                                 'arena_size': 5.0},
                                seed=40,
                                initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    obs, info = env.reset()

    duration_sec = 10
    ctrl_freq = env.CTRL_FREQ
    num_steps = int(duration_sec * ctrl_freq)

    target_pos = np.array([5.0, 5.0, 1.0])


    for step in range(num_steps):

        start_time = time.time()
        
        pos_error = target_pos - env.pos[0]
        Kp_pos = np.array([0.2, 0.2, 0.5])  
        
        desired_vel = Kp_pos * pos_error
        
        thrust = np.clip(1.0 + desired_vel[2], -1, 1) 
        
        action = np.array([thrust, thrust, thrust, thrust])  
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Simulation ended")
            break
        
        sleep_time = (1.0 / ctrl_freq) - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        print(f"Step {step}, Position: {env.pos[0]}, Action: {action}, Terminated: {terminated}")

    env.close()

if __name__ == "__main__":
    main()
