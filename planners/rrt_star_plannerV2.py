import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.cost = 0.0

class RRTStarPlannerV2:
    def __init__(self, 
                 start, 
                 goal, 
                 bvh_tree,
                 x_range, 
                 y_range, 
                 z_range,
                 max_iter=3000, 
                 step_size=0.3,
                 goal_sample_rate=0.2, 
                 search_radius=0.5):
        
        self.start = Node(start)
        self.goal = Node(goal)
        self.bvh_tree = bvh_tree  # BVH tree for efficient collision checks

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
            nearest_node = self.get_nearest_node(rnd_point)
            new_node = self.steer(nearest_node, rnd_point)

            # Check collision for the line segment
            if self.check_collision(nearest_node.position, new_node.position):
                near_nodes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_nodes)
                self.node_list.append(new_node)
                self.edge_list.append((new_node.parent, new_node))

                self.rewire(new_node, near_nodes)

            # Attempt to connect to the goal
            if np.linalg.norm(new_node.position - self.goal.position) <= self.step_size:
                if self.check_collision(new_node.position, self.goal.position):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + np.linalg.norm(new_node.position - self.goal.position)
                    self.node_list.append(self.goal)
                    return self.extract_path()

        # If we reach max_iter without connecting to the goal
        return None

    def sample(self):
        """Randomly sample space with 'goal_sample_rate' chance to sample the goal directly."""
        if random.random() < self.goal_sample_rate:
            return self.goal.position
        else:
            return np.array([
                random.uniform(self.x_range[0], self.x_range[1]),
                random.uniform(self.y_range[0], self.y_range[1]),
                random.uniform(self.z_range[0], self.z_range[1])
            ])

    def get_nearest_node(self, point):
        dists = [np.linalg.norm(n.position - point) for n in self.node_list]
        min_index = np.argmin(dists)
        return self.node_list[min_index]

    def steer(self, from_node, to_point):
        """Create a new node from 'from_node' toward 'to_point' with max distance 'step_size'."""
        direction = to_point - from_node.position
        dist = np.linalg.norm(direction)
        if dist > self.step_size:
            direction = (direction / dist) * self.step_size
        new_position = from_node.position + direction
        new_node = Node(new_position)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm(new_node.position - from_node.position)
        return new_node

    def check_collision(self, p1, p2):
        """
        Check if the line from p1 to p2 intersects any obstacle using the BVH tree.
        """
        return not self.check_line_intersection_with_bvh(p1, p2, self.bvh_tree)

    def check_line_intersection_with_bvh(self, p1, p2, node):
        """
        Recursively checks if the line segment from p1 to p2 intersects any obstacle in the BVH tree.
        """
        if node is None:
            return False

        # Check if the line segment intersects the current node's AABB
        aabb = node.aabb
        if not self.line_intersects_aabb(p1, p2, aabb):
            return False

        # If this is a leaf node, check the line against its objects
        if node.objects is not None:
            for obs in node.objects:
                if self.line_intersects_aabb(p1, p2, obs):
                    return True
            return False

        # Otherwise, check the children
        return (self.check_line_intersection_with_bvh(p1, p2, node.left) or
                self.check_line_intersection_with_bvh(p1, p2, node.right))

    def line_intersects_aabb(self, p1, p2, aabb):
        """
        Checks if a line segment intersects an AABB using the slab method.
        """
        aabb_min = aabb['aabb_min']
        aabb_max = aabb['aabb_max']

        tmin, tmax = 0.0, 1.0  # Normalized segment parameters
        for i in range(3):  # For x, y, z axes
            if abs(p2[i] - p1[i]) < 1e-6:  # Parallel to the slab
                if p1[i] < aabb_min[i] or p1[i] > aabb_max[i]:
                    return False
            else:
                inv_d = 1.0 / (p2[i] - p1[i])
                t1 = (aabb_min[i] - p1[i]) * inv_d
                t2 = (aabb_max[i] - p1[i]) * inv_d
                if t1 > t2:  # Swap if t1 > t2
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False
        return True

    def find_near_nodes(self, new_node):
        n = len(self.node_list)
        r = self.search_radius * np.sqrt((np.log(n) / n)) if n > 1 else self.search_radius
        dists = [np.linalg.norm(n.position - new_node.position) for n in self.node_list]
        near_nodes = [self.node_list[i] for i in range(len(self.node_list)) if dists[i] <= r]
        return near_nodes

    def choose_parent(self, new_node, near_nodes):
        best_parent = new_node.parent
        min_cost = new_node.cost
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
            if near_node == new_node.parent:
                continue
            if self.check_collision(new_node.position, near_node.position):
                cost = new_node.cost + np.linalg.norm(new_node.position - near_node.position)
                if cost < near_node.cost:
                    if near_node.parent is not None:
                        self.edge_list.remove((near_node.parent, near_node))
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

    def draw_tree(self, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for (parent, child) in self.edge_list:
            x_vals = [parent.position[0], child.position[0]]
            y_vals = [parent.position[1], child.position[1]]
            z_vals = [parent.position[2], child.position[2]]
            ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=0.5)

        ax.scatter(self.start.position[0], self.start.position[1], self.start.position[2],
                   color='green', marker='o', s=100, label='Start')
        ax.scatter(self.goal.position[0], self.goal.position[1], self.goal.position[2],
                   color='red', marker='*', s=100, label='Goal')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('RRT* Tree')
        ax.legend()

        ax.set_box_aspect([
            (self.x_range[1]-self.x_range[0]),
            (self.y_range[1]-self.y_range[0]),
            (self.z_range[1]-self.z_range[0])
        ])

        if show:
            plt.show()