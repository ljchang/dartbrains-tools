"""Optional compiled kernel; imported only when acceleration is requested."""
import numpy as np
from numba import njit


@njit(cache=True)
def maximum_cluster_sizes(maps, width, forming):
    result = np.zeros(len(maps), dtype=np.int64)
    voxels = width * width
    for p in range(len(maps)):
        visited = np.zeros(voxels, dtype=np.bool_)
        queue = np.empty(voxels, dtype=np.int64)
        for v in range(voxels):
            if visited[v] or abs(maps[p, v]) <= forming:
                continue
            queue[0] = v
            visited[v] = True
            head, tail = 0, 1
            while head < tail:
                u = queue[head]
                head += 1
                row, col = u // width, u % width
                for direction in range(4):
                    valid = ((direction == 0 and row > 0)
                             or (direction == 1 and row < width - 1)
                             or (direction == 2 and col > 0)
                             or (direction == 3 and col < width - 1))
                    if not valid:
                        continue
                    neighbor = u + (-width if direction == 0 else width if direction == 1 else -1 if direction == 2 else 1)
                    if not visited[neighbor] and abs(maps[p, neighbor]) > forming:
                        visited[neighbor] = True
                        queue[tail] = neighbor
                        tail += 1
            result[p] = max(result[p], tail)
    return result
