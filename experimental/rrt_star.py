import numpy as np

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, map_bounds, step_size=0.5, goal_sample_rate=0.3, max_iter=500):
        self.start = Node(start)
        self.goal = Node(goal)
        self.map_bounds = map_bounds
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]

    def plan(self):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node.position)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node)

            self.node_list.append(new_node)

            if self.calc_dist_to_goal(new_node.position) <= self.step_size:
                final_node = self.steer(new_node, self.goal)
                final_node.parent = new_node
                self.node_list.append(final_node)
                return self.generate_final_course(len(self.node_list) - 1)
        return None

    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            rnd = [
                np.random.uniform(self.map_bounds[0], self.map_bounds[1]),
                np.random.uniform(self.map_bounds[2], self.map_bounds[3])#,
                # np.random.uniform(self.map_bounds[4], self.map_bounds[5])
            ]
        else:
            rnd = self.goal.position
        return Node(rnd)

    def steer(self, from_node, to_node):
        new_node = Node(list(from_node.position))
        delta = np.array(to_node.position) - np.array(from_node.position)
        dist = np.linalg.norm(delta)
        step = min(self.step_size, dist)
        new_node.position += step * (delta / dist)
        new_node.parent = from_node
        return new_node

    def get_nearest_node_index(self, node_list, rnd_position):
        dlist = [np.linalg.norm(np.array(node.position) - np.array(rnd_position)) for node in node_list]
        return dlist.index(min(dlist))

    def calc_dist_to_goal(self, position):
        return np.linalg.norm(np.array(position) - np.array(self.goal.position))

    def generate_final_course(self, goal_ind):
        path = [self.goal.position]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.position)
            node = node.parent
        path.append(node.position)
        return path[::-1]