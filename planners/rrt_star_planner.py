import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.cost = 0.0

class RRTStarPlanner:
    def __init__(self, start, goal, obstacles, x_range, y_range, z_range, max_iter=1000, step_size=0.1, goal_sample_rate=0.1, search_radius=1.0):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles  
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.node_list = [self.start]
        self.edge_list = []

    def plan(self):
        for _ in range(self.max_iter):
            rnd_point = self.sample()
            nearest_node = self.get_nearest_node(self.node_list, rnd_point)
            new_node = self.steer(nearest_node, rnd_point)
            if self.check_collision(nearest_node.position, new_node.position):
                near_nodes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_nodes)
                self.node_list.append(new_node)
                self.edge_list.append((new_node.parent, new_node))
                self.rewire(new_node, near_nodes)
            if np.linalg.norm(new_node.position - self.goal.position) <= self.step_size:
                if self.check_collision(new_node.position, self.goal.position):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + np.linalg.norm(new_node.position - self.goal.position)
                    self.node_list.append(self.goal)
                    return self.extract_path()
        return None  # Failed to find a path

    def sample(self):
        ### Sample random points for exploration with a bias to the goal

        ### Higher goal sample rate means less exploration and maybe less optimal
        if random.random() < self.goal_sample_rate:
            return self.goal.position
        else:
            return np.array([
                random.uniform(self.x_range[0], self.x_range[1]),
                random.uniform(self.y_range[0], self.y_range[1]),
                random.uniform(self.z_range[0], self.z_range[1])
            ])

    def get_nearest_node(self, node_list, point):
        distances = [np.linalg.norm(node.position - point) for node in node_list]
        min_index = distances.index(min(distances))
        return node_list[min_index]

    def steer(self, from_node, to_point):
        ### Creates new node and path in the direction 
        # from the nearest node to the new node limited
        # by the step size
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = (direction / distance) * self.step_size
        new_position = from_node.position + direction
        new_node = Node(new_position)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm(new_node.position - from_node.position)
        return new_node

    def check_collision(self, p1, p2):
        for obs in self.obstacles:
            if self.line_intersects_obs(p1, p2, obs['position'], obs['size']):
                return False  
        return True  

    def line_intersects_obs(self, p1, p2, cube_center, cube_size):
        # AABB collision detection between a line segment and a cube
        dir_vector = p2 - p1
        for i in range(3):
            if dir_vector[i] == 0:
                if p1[i] < cube_center[i] - cube_size[i]/2 or p1[i] > cube_center[i] + cube_size[i]/2:
                    return False
            else:
                t1 = (cube_center[i] - cube_size[i]/2 - p1[i]) / dir_vector[i]
                t2 = (cube_center[i] + cube_size[i]/2 - p1[i]) / dir_vector[i]
                tmin = max(min(t1, t2), 0)
                tmax = min(max(t1, t2), 1)
                if tmin > tmax:
                    return False
        return True

    def find_near_nodes(self, new_node):
        n = len(self.node_list)
        r = self.search_radius * np.sqrt((np.log(n) / n))
        distances = [np.linalg.norm(node.position - new_node.position) for node in self.node_list]
        near_nodes = [self.node_list[i] for i in range(len(self.node_list)) if distances[i] <= r]
        return near_nodes

    def choose_parent(self, new_node, near_nodes):
        min_cost = new_node.cost
        best_parent = new_node.parent
        for near_node in near_nodes:
            if self.check_collision(near_node.position, new_node.position):
                cost = near_node.cost + np.linalg.norm(near_node.position - new_node.position)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = near_node
        new_node.cost = min_cost
        new_node.parent = best_parent
        return new_node

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if self.check_collision(new_node.position, near_node.position):
                cost = new_node.cost + np.linalg.norm(new_node.position - near_node.position)
                if cost < near_node.cost:
                    old_parent = near_node.parent
                    if old_parent is not None:
                        self.edge_list.remove((old_parent, near_node))

                    near_node.parent = new_node
                    near_node.cost = cost

                    self.edge_list.append((new_node, near_node))

    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node.position)
            node = node.parent
        path.reverse()
        return path

    def draw_tree(self, path=None, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the edges of the RRT* tree
        for edge in self.edge_list:
            parent_node, child_node = edge
            x_vals = [parent_node.position[0], child_node.position[0]]
            y_vals = [parent_node.position[1], child_node.position[1]]
            z_vals = [parent_node.position[2], child_node.position[2]]
            ax.plot(x_vals, y_vals, z_vals, color='blue', linewidth=0.5)

        # Plot the obstacles
        for obs in self.obstacles:
            center = obs['position']
            size = obs['size']

            # Plot the obstacle as a cube
            # The obstacle center is at `center`, and its size is `size`
            # The corners of the cube can be computed by extending the size along the 3 axes
            x = [center[0] - size[0] / 2, center[0] + size[0] / 2]
            y = [center[1] - size[1] / 2, center[1] + size[1] / 2]
            z = [center[2] - size[2] / 2, center[2] + size[2] / 2]

            # Draw the lines for the obstacle (cube)
            ax.plot([x[0], x[1]], [y[0], y[0]], [z[0], z[0]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[0], x[1]], [y[0], y[0]], [z[1], z[1]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[0], x[1]], [y[1], y[1]], [z[0], z[0]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[0], x[1]], [y[1], y[1]], [z[1], z[1]], color='red', linewidth=1, alpha=0.7)

            ax.plot([x[0], x[0]], [y[0], y[1]], [z[0], z[0]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[0], x[0]], [y[0], y[1]], [z[1], z[1]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[1], x[1]], [y[0], y[1]], [z[0], z[0]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[1], x[1]], [y[0], y[1]], [z[1], z[1]], color='red', linewidth=1, alpha=0.7)

            ax.plot([x[0], x[0]], [y[0], y[0]], [z[0], z[1]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[1], x[1]], [y[0], y[0]], [z[0], z[1]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[0], x[0]], [y[1], y[1]], [z[0], z[1]], color='red', linewidth=1, alpha=0.7)
            ax.plot([x[1], x[1]], [y[1], y[1]], [z[0], z[1]], color='red', linewidth=1, alpha=0.7)

        # Plot the start and goal nodes
        ax.scatter(self.start.position[0], self.start.position[1], self.start.position[2], color='green', marker='o',
                   s=100, label='Start')
        ax.scatter(self.goal.position[0], self.goal.position[1], self.goal.position[2], color='red', marker='*', s=100,
                   label='Goal')

        # Plot the final path if it exists
        if path is not None:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], color='orange', label='Path')

        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('RRT* Tree and Path')
        ax.legend()

        # Set equal aspect ratio
        ax.set_box_aspect([np.ptp(a) for a in [self.x_range, self.y_range, self.z_range]])

        if show:
            plt.show()
