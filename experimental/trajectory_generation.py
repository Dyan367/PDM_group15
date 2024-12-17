import numpy as np
import minsnap_trajectories as ms

def generate_reference_states(
    waypoints,
    set_speed,  # Prescribed speed
    total_time=None,
    vehicle_mass=1.0,
    yaw="velocity",
    degree=8,
    drag_params=None,
    yaw_rate=None
):
    """
    Generate reference states for trajectory generation with a prescribed speed.

    Args:
        waypoints (array): List of waypoint positions.
        set_speed (float): Prescribed constant speed (m/s).
        total_time (float, optional): Total duration of the trajectory.
        vehicle_mass (float): Quadrotor mass.
        yaw (str or float): Yaw direction ('velocity' aligns with velocity vector).
        degree (int): Polynomial degree for trajectory generation.
        drag_params (RotorDragParameters, optional): Drag parameters.
        yaw_rate (float, optional): Prescribed yaw rate.

    Returns:
        dict: Dictionary containing positions, velocities, attitudes, time samples, and angular velocities.
    """
    # Calculate the distances between waypoints
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    total_distance = np.sum(distances)

    # Validate that the set speed is achievable
    if set_speed <= 0:
        raise ValueError("Set speed must be greater than zero.")

    # Compute segment times based on set speed
    segment_times = distances / set_speed
    times = [0.0]
    for dt in segment_times:
        times.append(times[-1] + dt)

    # Optionally override total_time if provided
    if total_time is not None:
        scale_factor = total_time / times[-1]
        times = [t * scale_factor for t in times]

    # Create Waypoints with adjusted times
    refs = [ms.Waypoint(time=t, position=pos) for t, pos in zip(times, waypoints)]

    # Generate the trajectory
    trajectory = ms.generate_trajectory(
        references=refs,
        degree=degree,
        idx_minimized_orders=(3, 4),  # Minimize jerk and snap
        num_continuous_orders=3,       # Ensure continuity for position, velocity, and acceleration
        algorithm="closed-form",      # Use closed-form trajectory generation
    )

    # Generate time samples for trajectory evaluation
    time_samples = np.linspace(0, times[-1], 100)

    # Compute the quadrotor trajectory
    quadrotor_trajectory = ms.compute_quadrotor_trajectory(
        trajectory,
        t_sample=time_samples,
        vehicle_mass=vehicle_mass,
        yaw=yaw,
        drag_params=drag_params,
        yaw_rate=yaw_rate
    )

    # Extract positions, velocities, and attitudes
    positions = quadrotor_trajectory.position
    velocities = quadrotor_trajectory.velocity
    attitudes = quadrotor_trajectory.attitude
    angular_velocities = quadrotor_trajectory.body_rates

    return {
        "positions": positions,
        "velocities": velocities,
        "attitudes": attitudes,
        "time_samples": time_samples,
        "angular_velocities": angular_velocities
    }