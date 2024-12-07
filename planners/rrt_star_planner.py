import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import BSpline, splprep, splev
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
import pybullet as p
from bvh.bvh import BVHNode, build_bvh



class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.cost = 0.0

class RRTStarPlanner:
    def __init__(self, start, goal, obstacles, x_range, y_range, z_range,bvh,use_bvh=False, max_iter=1000, step_size=0.1, goal_sample_rate=0.1, search_radius=1.0):
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
        self.bvh = bvh
        self.use_bvh = use_bvh

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

    ## BASIC STEER FUNCTION
    # def steer(self, from_node, to_point):
    #     ### Creates new node and path in the direction 
    #     # from the nearest node to the new node limited
    #     # by the step size
    #     direction = to_point - from_node.position
    #     distance = np.linalg.norm(direction)
    #     if distance > self.step_size:
    #         direction = (direction / distance) * self.step_size
    #     new_position = from_node.position + direction
    #     new_node = Node(new_position)
    #     new_node.parent = from_node
    #     new_node.cost = from_node.cost + np.linalg.norm(new_node.position - from_node.position)
    #     return new_node

    ## APPROXIMATE STEER USING SIMULATION MODEL
    def steer(self, from_node, to_point):
        """
        Steers the quadrotor using DSLPIDControl for position and attitude control.

        Parameters
        ----------
        from_node : Node
            The starting node of the motion.
        to_point : ndarray
            The target position to steer towards.

        Returns
        -------
        Node
            The new node reached using the steering action.
        """
        # Initialize PID controller
        pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X, g=9.81)

        # Simulation parameters
        control_timestep = 0.01  # Time step for the control loop (s)
        max_time = 1.0  # Maximum time allowed for steering (s)
        time_elapsed = 0

        # Initialize the quadrotor state
        cur_pos = np.array(from_node.position)
        cur_vel = np.zeros(3)
        cur_quat = np.array([1, 0, 0, 0])  # Neutral orientation (w, x, y, z)
        cur_ang_vel = np.zeros(3)

        # Initialize target state
        target_pos = np.array(to_point)
        target_rpy = np.zeros(3)  # Assuming flat orientation

        path = [cur_pos]

        while time_elapsed < max_time:
            # Compute control action
            rpm, pos_e, yaw_error = pid_controller.computeControl(
                control_timestep=control_timestep,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=target_pos,
                target_rpy=target_rpy,
            )

            # Simulate the dynamics (simplified for illustration)
            # Update position based on velocity
            cur_vel += pos_e * control_timestep
            cur_pos += cur_vel * control_timestep

            # Update orientation (assuming no rotation for simplicity)
            cur_quat = np.array([1, 0, 0, 0])

            # Record the path
            path.append(cur_pos)

            # Check if we are close enough to the target
            if np.linalg.norm(cur_pos - target_pos) < self.step_size:
                break

            time_elapsed += control_timestep

        # Create a new node at the final position
        new_position = np.array(path[-1])
        new_node = Node(new_position)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm(new_position - from_node.position)

        return new_node

    ## BASIC CHECK COLLISION FUNCTION
    def basic_check_collision(self, p1, p2):
        for obs in self.obstacles:
            if self.line_intersects_obs(p1, p2, obs['aabb_min'], obs['aabb_max']):
                return False  # Collision detected
        return True  # No collision
    

    ## BOUNDING VOLUME HIEARCHY TREE COLLISION CHECK
    def check_collision(self, p1, p2):
        if not self.use_bvh:
            return self.basic_check_collision(p1,p2)
        else:
            aabb_min = np.minimum(p1, p2) - self.step_size
            aabb_max = np.maximum(p1, p2) + self.step_size

            potential_collisions = self.bvh.query_bvh(self.bvh, aabb_min, aabb_max)
            for obs in potential_collisions:
                if self.line_intersects_obs(p1, p2, obs['aabb_min'], obs['aabb_max']):
                    return False  # Collision detected
            return True  # No collision

    def line_intersects_obs(self, p1, p2, aabb_min, aabb_max):
        dir_vector = p2 - p1
        tmin = 0.0
        tmax = 1.0

        for i in range(3):
            if dir_vector[i] != 0.0:
                t1 = (aabb_min[i] - p1[i]) / dir_vector[i]
                t2 = (aabb_max[i] - p1[i]) / dir_vector[i]

                tmin_axis = min(t1, t2)
                tmax_axis = max(t1, t2)

                tmin = max(tmin, tmin_axis)
                tmax = min(tmax, tmax_axis)

                if tmin > tmax:
                    return False
            else:
                if p1[i] < aabb_min[i] or p1[i] > aabb_max[i]:
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
    
    def compute_bspline_path(self, path, degree=5, num_points=200):
        path = np.array(path)
        tck, u = splprep([path[:, 0], path[:, 1], path[:, 2]], s=0, k=degree)
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine, z_fine = splev(u_fine, tck)
        bspline_path = np.vstack((x_fine, y_fine, z_fine)).T
        return bspline_path



    #### Needs debugging ####
    def draw_tree(self, show=True):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the edges
            for edge in self.edge_list:
                parent_node, child_node = edge
                x_vals = [parent_node.position[0], child_node.position[0]]
                y_vals = [parent_node.position[1], child_node.position[1]]
                z_vals = [parent_node.position[2], child_node.position[2]]
                ax.plot(x_vals, y_vals, z_vals, color='blue', linewidth=0.5)


            # Plot the start and goal nodes
            ax.scatter(self.start.position[0], self.start.position[1], self.start.position[2], color='green', marker='o', s=100, label='Start')
            ax.scatter(self.goal.position[0], self.goal.position[1], self.goal.position[2], color='red', marker='*', s=100, label='Goal')

            # Set labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('RRT* Tree')
            ax.legend()

            # Set equal aspect ratio
            ax.set_box_aspect([np.ptp(a) for a in [self.x_range, self.y_range, self.z_range]])


            if show:
                plt.show()


