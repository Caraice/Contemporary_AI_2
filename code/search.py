"""四方向 A* 寻路，按进入格子的代价计费。"""
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import inf


@dataclass
class SearchResult:
    path: list
    cost: int
    expanded: int = 0  # 有效弹出次数，含终点
    generated: int = 0  # 入堆次数，含起点
    trace: list = field(default_factory=list)


def astar(grid, start, goal, *, weighted=False, goal_cost=0,
          heuristic="manhattan", tie="fifo", record=False):
    """S 代价为 1，X 为障碍，终点默认免费。

    goal_cost=None 时保留地形权值；启发下界兼容免费终点。
    """
    if heuristic not in ("zero", "manhattan") or tie not in ("fifo", "larger_g"):
        raise ValueError("unknown heuristic or tie policy")
    if not grid or not grid[0] or any(len(r) != len(grid[0]) for r in grid):
        raise ValueError("grid must be nonempty and rectangular")
    start, goal = tuple(start), tuple(goal)
    rows, cols = len(grid), len(grid[0])

    def inside(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols

    if not inside(start) or not inside(goal):
        raise ValueError("endpoint out of bounds")

    costs = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "X":
                continue
            if not weighted:
                v = 1
            elif (r, c) == goal and goal_cost is not None:
                v = goal_cost
            elif cell == "S":
                v = 1
            else:
                if isinstance(cell, bool):
                    raise ValueError("boolean is not a terrain cost")
                v = int(cell)
                if v != float(cell):
                    raise ValueError("terrain costs must be integers")
            if v < 0:
                raise ValueError("A* requires nonnegative costs")
            costs[r, c] = v
    if start not in costs or goal not in costs:
        return SearchResult([], -1)
    m = min((v for p, v in costs.items() if p != goal), default=0)

    def h(p):
        d = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        if heuristic == "zero" or d == 0:
            return 0
        return m * (d - 1) + min(m, costs[goal])

    serial = count()
    queue = [(h(start), 0, next(serial), 0, start)]
    distance, parent = {start: 0}, {}
    result = SearchResult([], -1, generated=1)
    while queue:
        f, _, _, g, p = heappop(queue)
        if g != distance[p]:
            continue
        result.expanded += 1
        if record:
            result.trace.append({"node": p, "g": g, "h": h(p), "f": f})
        if p == goal:
            path = [p]
            while p in parent:
                p = parent[p]
                path.append(p)
            result.path, result.cost = path[::-1], g
            return result
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            q = p[0] + dr, p[1] + dc
            if q not in costs:
                continue
            ng = g + costs[q]
            if ng < distance.get(q, inf):
                distance[q], parent[q] = ng, p
                heappush(queue, (ng + h(q), -ng if tie == "larger_g" else 0,
                                 next(serial), ng, q))
                result.generated += 1
    return result


def maze_path(start, goal, grid):
    return astar(grid, start, goal).path


def weighted_cost(start, goal, grid):
    return astar(grid, start, goal, weighted=True).cost
