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
    def __init__(self, start, goal, obstacles_info,
                 x_range, y_range, z_range,
                 max_iter=3000, step_size=0.3,
                 goal_sample_rate=0.2, search_radius=2.0,
                 collision_subsamples=50):
        """
        :param obstacles_info: A list of dicts describing each obstacle:
            e.g. [
              {"type": "box", "position": np.array([x,y,z]), "size": np.array([sx,sy,sz]), "phys_id": ...},
              {"type": "cylinder", "position": ..., "radius": ..., "height": ..., "phys_id": ...},
              ...
            ]
        :param collision_subsamples: how many points to sample along each new edge.
        """
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles_info = obstacles_info  # The new list from your environment

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.collision_subsamples = collision_subsamples

        self.node_list = [self.start]
        self.edge_list = []

    def plan(self):
        for _ in range(self.max_iter):
            rnd_point = self.sample()
            nearest_node = self.get_nearest_node(rnd_point)
            new_node = self.steer(nearest_node, rnd_point)

            # Check collision (including sub-sampling) 
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

        # If we reach max_iter without connecting to goal
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
        """Check for collision by sub-sampling the line from p1->p2."""
        for alpha in np.linspace(0, 1, self.collision_subsamples):
            pt = p1 + alpha*(p2 - p1)
            # Check if this pt is inside any obstacle
            for obs in self.obstacles_info:
                if self.point_in_obstacle(pt, obs):
                    return False
        return True

    def point_in_obstacle(self, pt, obs):
        """Returns True if 'pt' is inside the given obstacle dict from obstacles_info."""
        obs_type = obs["type"]
        center   = obs["position"]

        if obs_type == "box":
            # For axis-aligned box: check if pt is within [center - size/2, center + size/2]
            size = obs["size"]  # e.g. [sx, sy, sz]
            half = 2*size / 2.0
            if (center[0] - half[0] <= pt[0] <= center[0] + half[0] and
                center[1] - half[1] <= pt[1] <= center[1] + half[1] and
                center[2] - half[2] <= pt[2] <= center[2] + half[2]):
                return True
            return False

        elif obs_type == "cylinder":
            # Cylinder aligned along Z-axis
            radius = obs["radius"]
            height = obs["height"]
            dist_xy = np.linalg.norm(pt[:2] - center[:2])
            z_min = center[2] - height/2.0
            z_max = center[2] + height/2.0
            if dist_xy <= radius and z_min <= pt[2] <= z_max:
                return True
            return False

        elif obs_type == "sphere":
            # If you have spherical obstacles
            r = obs["radius"]
            dist_to_center = np.linalg.norm(pt - center)
            return (dist_to_center <= r)

        else:
            # If an unknown shape, treat it as no collision
            return False

    def find_near_nodes(self, new_node):
        n = len(self.node_list)
        # Typical RRT* radius factor
        r = self.search_radius * np.sqrt((np.log(n) / n)) if n>1 else self.search_radius
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
                    # Remove old edge
                    if near_node.parent is not None:
                        self.edge_list.remove((near_node.parent, near_node))
                    # Rewire
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

    #### For Debugging / Visualization ####
    def draw_tree(self, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Edges
        for (parent, child) in self.edge_list:
            x_vals = [parent.position[0], child.position[0]]
            y_vals = [parent.position[1], child.position[1]]
            z_vals = [parent.position[2], child.position[2]]
            ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=0.5)

        # Start + Goal
        ax.scatter(self.start.position[0], self.start.position[1], self.start.position[2],
                   color='green', marker='o', s=100, label='Start')
        ax.scatter(self.goal.position[0], self.goal.position[1], self.goal.position[2],
                   color='red', marker='*', s=100, label='Goal')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('RRT* Tree')
        ax.legend()

        # Match aspect ratio to search range
        ax.set_box_aspect([
            (self.x_range[1]-self.x_range[0]),
            (self.y_range[1]-self.y_range[0]),
            (self.z_range[1]-self.z_range[0])
        ])

        if show:
            plt.show()
