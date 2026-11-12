import heapq
import numpy as np

def point_in_rects(point, blocks, margin=0.0):
    if blocks is None or len(blocks) == 0:
        return False
    x, y = float(point[0]), float(point[1])
    for x0, y0, x1, y1 in blocks:
        if (x0 - margin) <= x <= (x1 + margin) and (y0 - margin) <= y <= (y1 + margin):
            return True
    return False

def _rect_clear(rect, start, target, clearance):
    for pt in (start, target):
        if point_in_rects(pt, rect.reshape(1, 4), margin=clearance):
            return False
    return True

def _rects_overlap(a, b, gap=0.02):
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])

def sample_blocks(rng, start, target, n_blocks=(2, 4), region=(0.05, 0.95), clearance=0.07):
    lo, hi = region
    n = int(rng.integers(n_blocks[0], n_blocks[1] + 1))
    blocks = []
    for _ in range(80):
        if len(blocks) >= n:
            break
        w = float(rng.uniform(0.09, 0.22))
        h = float(rng.uniform(0.09, 0.22))
        x0 = float(rng.uniform(lo, hi - w))
        y0 = float(rng.uniform(lo, hi - h))
        rect = np.array([x0, y0, x0 + w, y0 + h], dtype=float)
        if not _rect_clear(rect, start, target, clearance):
            continue
        if any(_rects_overlap(rect, other) for other in blocks):
            continue
        blocks.append(rect)
    return np.asarray(blocks, dtype=float)

def _grid_index(point, grid_size):
    g = grid_size - 1
    ij = np.clip(np.round(point * g), 0, g).astype(int)
    return int(ij[0]), int(ij[1])

def _cell_center(i, j, grid_size):
    g = grid_size - 1
    return np.array([i / g, j / g], dtype=float)

def _build_blocked(blocks, grid_size):
    blocked = np.zeros((grid_size, grid_size), dtype=bool)
    if blocks is None or len(blocks) == 0:
        return blocked
    for i in range(grid_size):
        for j in range(grid_size):
            if point_in_rects(_cell_center(i, j, grid_size), blocks, margin=0.0):
                blocked[i, j] = True
    return blocked

def astar_path(start, target, blocks, grid_size=32):
    blocked = _build_blocked(blocks, grid_size)
    si, sj = _grid_index(start, grid_size)
    ti, tj = _grid_index(target, grid_size)
    if blocked[si, sj] or blocked[ti, tj]:
        return None

    def h(i, j):
        return abs(i - ti) + abs(j - tj)

    open_heap = [(h(si, sj), 0, si, sj)]
    came_from = {}
    g_score = {(si, sj): 0}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _, cost, i, j = heapq.heappop(open_heap)
        if (i, j) == (ti, tj):
            path = [_cell_center(ti, tj, grid_size)]
            cur = (ti, tj)
            while cur in came_from:
                cur = came_from[cur]
                path.append(_cell_center(cur[0], cur[1], grid_size))
            path.reverse()
            path[0] = np.asarray(start, dtype=float)
            path[-1] = np.asarray(target, dtype=float)
            return path

        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= grid_size or nj >= grid_size:
                continue
            if blocked[ni, nj]:
                continue
            step = 1.414 if di != 0 and dj != 0 else 1.0
            new_g = cost + step
            key = (ni, nj)
            if key not in g_score or new_g < g_score[key]:
                g_score[key] = new_g
                came_from[key] = (i, j)
                heapq.heappush(open_heap, (new_g + h(ni, nj), new_g, ni, nj))
    return None

def expert_direction(pos, target, blocks, grid_size=32):
    path = astar_path(pos, target, blocks, grid_size=grid_size)
    if path is None or len(path) < 2:
        delta = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    else:
        for waypoint in path[1:]:
            delta = waypoint - np.asarray(pos, dtype=float)
            if np.linalg.norm(delta) > 1e-5:
                break
    norm = np.linalg.norm(delta)
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return (delta / norm).astype(np.float32)
