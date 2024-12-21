import numpy as np

class BVHNode:
    def __init__(self, aabb_min, aabb_max, left=None, right=None, objects=None):
        self.aabb_min = aabb_min  # Minimum point of the AABB
        self.aabb_max = aabb_max  # Maximum point of the AABB
        self.left = left          # Left child
        self.right = right        # Right child
        self.objects = objects    # List of objects in this node (for leaf nodes)
        
    def query_bvh(self, node, aabb_min, aabb_max):
        """
        Query the BVH for obstacles potentially intersecting a given AABB.

        Parameters
        ----------
        node : BVHNode
            The root node of the BVH to query.
        aabb_min : ndarray
            Minimum corner of the query AABB.
        aabb_max : ndarray
            Maximum corner of the query AABB.

        Returns
        -------
        list
            List of obstacle dictionaries potentially colliding with the query AABB.
        """
        if not self.is_overlap(node.aabb_min, node.aabb_max, aabb_min, aabb_max):
            return []  # No potential collisions at this node

        # If this is a leaf node, return its objects
        if node.objects is not None:
            return node.objects

        # Otherwise, query the child nodes
        results = []
        if node.left:
            results.extend(self.query_bvh(node.left, aabb_min, aabb_max))
        if node.right:
            results.extend(self.query_bvh(node.right, aabb_min, aabb_max))
        return results

    def is_overlap(self, aabb1_min, aabb1_max, aabb2_min, aabb2_max):
        """
        Check if two AABBs overlap.

        Parameters
        ----------
        aabb1_min : ndarray
            Minimum corner of the first AABB.
        aabb1_max : ndarray
            Maximum corner of the first AABB.
        aabb2_min : ndarray
            Minimum corner of the second AABB.
        aabb2_max : ndarray
            Maximum corner of the second AABB.

        Returns
        -------
        bool
            True if the AABBs overlap, False otherwise.
        """
        return np.all(aabb1_max >= aabb2_min) and np.all(aabb2_max >= aabb1_min)


def compute_aabb(obstacles):
    """Compute the AABB that encloses all given obstacles."""
    aabb_min = np.min([obs['aabb_min'] for obs in obstacles], axis=0)
    aabb_max = np.max([obs['aabb_max'] for obs in obstacles], axis=0)
    return aabb_min, aabb_max


def build_bvh(obstacles, max_leaf_size=1):
    """Recursively build a BVH from the obstacles."""
    if len(obstacles) <= max_leaf_size:
        # Create a leaf node if the number of objects is below the threshold
        aabb_min, aabb_max = compute_aabb(obstacles)
        return BVHNode(aabb_min, aabb_max, objects=obstacles)

    # Sort obstacles by the midpoint of their AABBs along the longest axis
    extents = [obs['aabb_max'] - obs['aabb_min'] for obs in obstacles]
    overall_min, overall_max = compute_aabb(obstacles)
    lengths = overall_max - overall_min
    longest_axis = np.argmax(lengths)

    obstacles.sort(key=lambda obs: (obs['aabb_min'][longest_axis] + obs['aabb_max'][longest_axis]) / 2)

    # Split the obstacles into two groups
    mid = len(obstacles) // 2
    left_obstacles = obstacles[:mid]
    right_obstacles = obstacles[mid:]

    # Build child nodes
    left_node = build_bvh(left_obstacles, max_leaf_size)
    right_node = build_bvh(right_obstacles, max_leaf_size)

    # Create an internal node
    aabb_min, aabb_max = compute_aabb(obstacles)
    return BVHNode(aabb_min, aabb_max, left=left_node, right=right_node)
