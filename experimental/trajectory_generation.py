import numpy as np
import minsnap_trajectories as ms


def generate_reference_states(
    waypoints, total_time=10.0, vehicle_mass=1.0, yaw="velocity", degree=8, drag_params=None, yaw_rate=None
):
    # Compute the total distance and normalize segment durations
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    total_distance = np.sum(distances)
    segment_ratios = distances / total_distance

    # Assign times proportional to segment distances
    times = [0.0]
    for ratio in segment_ratios:
        times.append(times[-1] + ratio * total_time)

    # Create Waypoints with estimated times
    refs = [ms.Waypoint(time=t, position=pos) for t, pos in zip(times, waypoints)]

    # Generate the trajectory
    trajectory = ms.generate_trajectory(
        references=refs,
        degree=degree,
        idx_minimized_orders=(3, 4),  # Minimize jerk and snap
        num_continuous_orders=3,  # Ensure continuity for position, velocity, and acceleration
        algorithm="closed-form",  # Use closed-form trajectory generation
    )

    # Generate time samples for trajectory evaluation
    time_samples = np.linspace(0, total_time, 100)

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


