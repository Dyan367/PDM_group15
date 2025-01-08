import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import pybullet as p
from environments.factory_environment import create_env

def main():
    # Create the environment
    env, num_steps = create_env(duration_sec=50, simulation_freq_hz=240, control_freq_hz=48, gui=True)
    
    # Reset the environment
    obs, info = env.reset()
    
    # Set the camera to look directly down at the starting position
    start_pos = [12.0, 5.0, 1.0]  # Starting position of the drone
    camera_distance = 20.0        # Distance from the camera to the object
    camera_yaw = 0.0             # Camera yaw angle
    camera_pitch = -89.0         # Camera pitch angle to look straight down
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=start_pos,
        physicsClientId=env.CLIENT
    )
    
    # Run the simulation loop
    for i in range(num_steps):
        # Example action: zero control input (hover in place)
        action = np.zeros((1, 4))  # [vx, vy, vz, yaw_rate]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Simulation ended")
            break
        
        # Maintain control loop timing
        time.sleep(1.0 / env.CTRL_FREQ)

    env.close()

if __name__ == "__main__":
    main()
