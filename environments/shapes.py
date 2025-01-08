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
    # Collision box
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=size,
        physicsClientId=client_id
    )
    
    # Visual box
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=size,
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







