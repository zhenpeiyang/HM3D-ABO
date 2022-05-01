import numpy as np
try:
    import open3d as o3d
except:
    pass


def write_ply(fn, point, normal=None, color=None):

    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)
    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)
    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)
    o3d.io.write_point_cloud(fn, ply)


def transform4x4(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (np.matmul(T[:3, :3], pc.T) + T[:3, 3:4]).T


def backproject_depth(depth, inv_K, mask=False):
    h, w = depth.shape
    idu, idv = np.meshgrid(range(w), range(h))
    grid = np.stack((idu.flatten(), idv.flatten(), np.ones([w * h])))
    x = np.matmul(inv_K[:3, :3], grid)
    x = x * depth.flatten()[None, :]
    x = x.T
    if mask:
        x = x[depth.flatten() > 0]
    return x


def get_local_split(items: list, world_size: int, rank: int, seed: int):
    """ The local rank only loads a split of the dataset. """
    n_items = len(items)
    items_permute = np.random.RandomState(seed).permutation(items)
    (n_items // world_size)
    if n_items % world_size == 0:
        padded_items = items_permute
    else:
        padding = np.random.RandomState(seed).choice(items,
                                                     world_size -
                                                     (n_items % world_size),
                                                     replace=True)
        padded_items = np.concatenate([items_permute, padding])
        assert len(padded_items) % world_size == 0, \
            f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank = len(padded_items) // world_size
    if n_per_rank * rank >= n_items:
        return []
    else:
        return padded_items[n_per_rank * rank:min(n_per_rank *
                                                  (rank + 1), n_items)]

class Timer:
    """
    Simple wrapper for time.clock().
    """
    def __init__(self):
        """
        Initialize and start timer.
        """

        self.start = time.perf_counter()
        """ (float) Seconds. """

    def reset(self):
        """
        Reset timer.
        """

        self.start = time.perf_counter()

    def elapsed(self):
        """
        Get elapsed time in seconds

        :return: elapsed time in seconds
        :rtype: float
        """

        return (time.perf_counter() - self.start)
