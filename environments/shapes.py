import pybullet as p

def create_box_shape(size, color=[0.6, 0.4, 0.2, 1], client_id=0):
    """
    Parameters:
    - size: A list or array [x_half, y_half, z_half] representing half-extents of the box.
    - color: RGBA color for the visual shape.
    - client_id: The PyBullet client ID.

    Returns:
    - obstacle_id: The ID of the created multi-body obstacle.
    """
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=size,
        physicsClientId=client_id
    )
    
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=size,
        rgbaColor=color,
        physicsClientId=client_id
    )

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
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        physicsClientId=client_id
    )
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client_id
    )
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
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height,
        physicsClientId=client_id
    )
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color,
        physicsClientId=client_id
    )
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        physicsClientId=client_id
    )
    return obstacle_id

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

    if (new_x < bounds[0] or new_x > bounds[1] or
        new_y < bounds[2] or new_y > bounds[3] or
        new_z < bounds[4] or new_z > bounds[5]):

        p.resetBasePositionAndOrientation(
            obstacle_id,
            reset_position,
            [0, 0, 0, 1],
            physicsClientId=client_id
        )
    else:

        p.resetBasePositionAndOrientation(
            obstacle_id,
            [new_x, new_y, new_z],
            [0, 0, 0, 1],
            physicsClientId=client_id
        )
