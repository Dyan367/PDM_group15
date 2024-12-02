import math

class Sphere:
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

    def is_colliding(self, other):
        distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        return distance < (self.radius + other.radius)

def main():
    # Define the main sphere
    drone = Sphere(0, 0, 0, 5)

    # Define a list of obstacle spheres
    obstacles = [
        Sphere(10, 0, 0, 5),
        Sphere(0, 10, 0, 5),
        Sphere(0, 0, 10, 5),
        Sphere(3, 3, 3, 2)
    ]

    # Check for collisions
    for i, obstacle in enumerate(obstacles):
        if drone.is_colliding(obstacle):
            print(f"Collision detected with obstacle {i} at ({obstacle.x}, {obstacle.y}, {obstacle.z})")
        else:
            print(f"No collision with obstacle {i} at ({obstacle.x}, {obstacle.y}, {obstacle.z})")

if __name__ == "__main__":
    main()