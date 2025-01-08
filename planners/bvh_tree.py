import numpy as np

class BVHNode:
    def __init__(self, aabb=None, left=None, right=None, objects=None):
        self.aabb = aabb  # The bounding box for this node
        self.left = left  # Left child
        self.right = right  # Right child
        self.objects = objects  # List of objects if this is a leaf

def compute_union_aabb(aabb_list):
    """
    Compute the union of a list of AABBs.
    """
    mins = np.min([aabb['aabb_min'] for aabb in aabb_list], axis=0)
    maxs = np.max([aabb['aabb_max'] for aabb in aabb_list], axis=0)
    return {'aabb_min': mins, 'aabb_max': maxs}

def build_bvh(aabbs, depth=0, max_objects_per_leaf=1):
    """
    Build a bounding volume hierarchy (BVH) tree from a list of AABBs.
    """
    if len(aabbs) <= max_objects_per_leaf:
        # Create a leaf node
        return BVHNode(aabb=compute_union_aabb(aabbs), objects=aabbs)

    # Choose the axis to split along (alternates by depth)
    axis = depth % 3

    # Sort AABBs by their center along the chosen axis
    aabbs.sort(key=lambda aabb: np.mean([aabb['aabb_min'][axis], aabb['aabb_max'][axis]]))

    # Split the list into two halves
    mid = len(aabbs) // 2
    left_aabbs = aabbs[:mid]
    right_aabbs = aabbs[mid:]

    # Recursively build left and right subtrees
    left_child = build_bvh(left_aabbs, depth + 1, max_objects_per_leaf)
    right_child = build_bvh(right_aabbs, depth + 1, max_objects_per_leaf)

    # Create an internal node
    union_aabb = compute_union_aabb(aabbs)
    return BVHNode(aabb=union_aabb, left=left_child, right=right_child)


