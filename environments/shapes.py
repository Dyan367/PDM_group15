import pybullet as p
import random
import numpy as np

def create_box_shape(size, color=[0.6, 0.4, 0.2, 1], client_id=0):
    """
    Parameters:
    - size: A list or array [x_half, y_half, z_half] representing half-extents of the box.
    - color: RGBA color for the visual shape.
    - client_id: The PyBullet client ID.

    Returns:
    - obstacle_id: The ID of the created multi-body obstacle.
    """
    half = [d/2.0 for d in size]
    # Collision box
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half,
        physicsClientId=client_id
    )
    
    # Visual box
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half,
        rgbaColor=color,
        physicsClientId=client_id
    )

    # Add the shape to the simulation
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        physicsClientId=client_id
    )
    return obstacle_id

def create_sphere_shape(radius, color=[0.2, 0.6, 0.8, 1], client_id=0):
    """
    Parameters:
    - radius: Radius of the sphere.
    - color: RGBA color for the visual shape.
    - client_id: The PyBullet client ID.

    Returns:
    - obstacle_id: The ID of the created multi-body obstacle.
    """
    # Collision sphere
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        physicsClientId=client_id
    )

    # Visual sphere
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client_id
    )

    # Add the shape to the simulation
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        physicsClientId=client_id
    )
    return obstacle_id

def create_cylinder_shape(radius, height, color=[0.5, 0.5, 0.5, 1], client_id=0):
    """
    Parameters:
    - radius: Radius of the cylinder's base.
    - height: Height of the cylinder.
    - color: RGBA color for the visual shape.
    - client_id: The PyBullet client ID.

    Returns:
    - obstacle_id: The ID of the created multi-body obstacle.
    """
    # Collision cylinder
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height,
        physicsClientId=client_id
    )

    # Visual cylinder
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color,
        physicsClientId=client_id
    )

    # Add the shape to the simulation
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        physicsClientId=client_id
    )
    return obstacle_id

def add_bounding_box(bounds, client_id=0):
    """
    Parameters:
    - bounds: List [min_x, max_x, min_y, max_y, min_z, max_z].
    - client_id: The PyBullet client ID.
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bounds

    # Calculate center and size of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    # Create visual bounding box
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size_x / 2, size_y / 2, size_z / 2],
        rgbaColor=[1, 0, 0, 0.3],  # Transparent red box
        physicsClientId=client_id
    )

    # Add the visual shape to the simulation
    p.createMultiBody(
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[center_x, center_y, center_z],
        physicsClientId=client_id
    )


def move_shape_dynamic(obstacle_id, velocity_x, velocity_y, velocity_z, bounds, timestep, client_id=0):
    """
    Parameters:
    - obstacle_id: The ID of the obstacle to move.
    - velocity_x: Movement speed along the x-axis.
    - velocity_y: Movement speed along the y-axis.
    - velocity_z: Movement speed along the z-axis.
    - bounds: List [min_bound, max_bound] specifying movement bounds for x, y, and z.
    - timestep: Control timestep for updating the position.
    - client_id: The PyBullet client ID.

    Returns:
    - Updated velocities: velocity_x, velocity_y, velocity_z after potential direction changes.
    """
    position, _ = p.getBasePositionAndOrientation(obstacle_id, physicsClientId=client_id)

    new_x = position[0] + velocity_x * timestep
    new_y = position[1] + velocity_y * timestep
    new_z = position[2] + velocity_z * timestep

    # Reverse direction if out of bounds for x-axis
    if new_x < bounds[0] or new_x > bounds[1]:
        velocity_x *= -1
        new_x = position[0] + velocity_x * timestep

    # Reverse direction if out of bounds for y-axis
    if new_y < bounds[2] or new_y > bounds[3]:
        velocity_y *= -1
        new_y = position[1] + velocity_y * timestep

    # Reverse direction if out of bounds for z-axis
    if new_z < bounds[4] or new_z > bounds[5]:
        velocity_z *= -1
        new_z = position[2] + velocity_z * timestep

    # Update the position of the shape
    p.resetBasePositionAndOrientation(
        obstacle_id,
        [new_x, new_y, new_z],
        [0, 0, 0, 1],
        physicsClientId=client_id
    )

    return velocity_x, velocity_y, velocity_z

def move_shape_reset(obstacle_id, velocity_x, velocity_y, velocity_z, bounds, timestep, reset_position, client_id=0):
    """
    Parameters:
    - obstacle_id: The ID of the obstacle to move.
    - velocity_x: Movement speed along the x-axis.
    - velocity_y: Movement speed along the y-axis.
    - velocity_z: Movement speed along the z-axis.
    - bounds: List [min_bound, max_bound] specifying movement bounds for x, y, and z.
    - timestep: Control timestep for updating the position.
    - reset_position: List [x, y, z] specifying the reset position.
    - client_id: The PyBullet client ID.
    """
    position, _ = p.getBasePositionAndOrientation(obstacle_id, physicsClientId=client_id)

    new_x = position[0] + velocity_x * timestep
    new_y = position[1] + velocity_y * timestep
    new_z = position[2] + velocity_z * timestep

    # Check if out of bounds
    if (new_x < bounds[0] or new_x > bounds[1] or
        new_y < bounds[2] or new_y > bounds[3] or
        new_z < bounds[4] or new_z > bounds[5]):

        # Reset to "reset position" when out of bounds
        p.resetBasePositionAndOrientation(
            obstacle_id,
            reset_position,
            [0, 0, 0, 1],
            physicsClientId=client_id
        )
    else:
        # Update position based on speed if within bounds
        p.resetBasePositionAndOrientation(
            obstacle_id,
            [new_x, new_y, new_z],
            [0, 0, 0, 1],
            physicsClientId=client_id
        )

def move_shape_random(obstacle_id, bounds, max_speed, timestep, change_interval, client_id=0):
    """
    Parameters:
    - obstacle_id: The ID of the obstacle to move.
    - bounds: List [min_x, max_x, min_y, max_y, min_z, max_z].
    - max_speed: Maximum speed for random movement.
    - timestep: Simulation timestep.
    - change_interval: Time interval for changing direction.
    - client_id: The PyBullet client ID.
    """
    if not hasattr(move_shape_random, "state"):
        move_shape_random.state = {}
    state = move_shape_random.state.setdefault(
        obstacle_id,
        {
            "velocity": [
                np.random.uniform(-max_speed, max_speed),   # Initial x velocity
                np.random.uniform(-max_speed, max_speed),   # Initial y velocity
                0                                           # Initial z velocity
            ],
            "time_since_change": 0
        }
    )

    state["time_since_change"] += timestep

    # Change speed randomly in x, y plane
    if state["time_since_change"] >= change_interval:
        state["velocity"] = [
            np.random.uniform(-max_speed, max_speed),   # x direction
            np.random.uniform(-max_speed, max_speed),   # y direction
            0                                           # z direction
        ]
        state["time_since_change"] = 0

    # Update position based on speed
    position, _ = p.getBasePositionAndOrientation(obstacle_id, physicsClientId=client_id)
    new_position = [
        position[0] + state["velocity"][0] * timestep,  # x
        position[1] + state["velocity"][1] * timestep,  # y
        position[2]                                     # z
    ]

    # Boundary conditions
    for i, (pos, vel, bound_min, bound_max) in enumerate(zip(new_position[:2], state["velocity"][:2], bounds[::2], bounds[1::2])):
        if pos < bound_min:
            new_position[i] = bound_min 
            state["velocity"][i] = abs(vel) 
        elif pos > bound_max:
            new_position[i] = bound_max
            state["velocity"][i] = -abs(vel) 


    # Update position
    p.resetBasePositionAndOrientation(
        obstacle_id, new_position, [0, 0, 0, 1], physicsClientId=client_id
    )