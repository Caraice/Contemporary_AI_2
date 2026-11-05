"""独立校验最短路与博弈结果，种子为 20261105。"""
import json
import random
from pathlib import Path
from cases import MAZE, WEIGHTED, TREASURE_GRID, POSITIONS, VALUES
from search import astar, maze_path, weighted_cost
from game import (TreeNode, sample_tree, tree_search, distance_matrix,
                  treasure_search)


def oracle_distance(grid, start, goal, weighted=False, free_goal=False):
    # 反复松弛求最短路，不复用 A* 代码。
    cells = [(r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x != "X"]
    if start not in cells or goal not in cells:
        return -1
    d = {p: float("inf") for p in cells}
    d[start] = 0
    for _ in range(len(cells) - 1):
        changed = False
        for r, c in cells:
            for q in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if q not in d:
                    continue
                cell = grid[q[0]][q[1]]
                cost = 1 if not weighted or cell == "S" else int(cell) if cell != "E" else 0
                if free_goal and q == goal:
                    cost = 0
                if d[(r,c)] + cost < d[q]:
                    d[q] = d[(r,c)] + cost
                    changed = True
        if not changed:
            break
    return -1 if d[goal] == float("inf") else d[goal]


def check_path(result, grid, start, goal, weighted=False, free_goal=False):
    if result.cost == -1:
        assert not result.path
        return
    assert result.path[0] == start and result.path[-1] == goal
    total = 0
    for p, q in zip(result.path, result.path[1:]):
        assert abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1
        assert grid[q[0]][q[1]] != "X"
        total += (0 if q == goal and free_goal else
                  1 if not weighted or grid[q[0]][q[1]] == "S" else int(grid[q[0]][q[1]]))
    assert total == result.cost


def exhaustive_game(d, values, depth):
    # 独立递归枚举，不复用状态类和转移函数。
    def walk(a, b, remaining, u, turn, h):
        if not remaining or not h:
            return u
        scores = []
        for t in remaining:
            gain = values[t] - d[a if turn == 0 else b][t]
            scores.append(walk(t if turn == 0 else a, b if turn == 0 else t,
                               tuple(x for x in remaining if x != t),
                               u + gain if turn == 0 else u - gain, 1-turn, h-1))
        return (max if turn == 0 else min)(scores)
    root = {t: walk(t, "B", tuple(x for x in values if x != t),
                    values[t]-d["A"][t], 1, depth-1) for t in values}
    return max(root.values()), root


def run_verification(output=None):
    rng = random.Random(20261105)
    counts = {"sample_checks": 0, "random_grid_searches": 0,
              "random_tree_searches": 0, "random_games": 0}
    assert len(maze_path((0,0), (3,4), MAZE))-1 == 7
    assert weighted_cost((0,0), (4,4), WEIGHTED) == 7
    assert tree_search(sample_tree())[0] == 5
    counts["sample_checks"] += 3
    for weighted in (False, True):
        for _ in range(120):
            rows, cols = rng.randint(2, 8), rng.randint(2, 8)
            grid = [["X" if rng.random() < .25 else rng.randint(0, 9) if weighted else "."
                     for c in range(cols)] for r in range(rows)]
            start, goal = (0,0), (rows-1,cols-1)
            grid[0][0], grid[-1][-1] = ("S", "E") if weighted else ("S", "E")
            expected = oracle_distance(grid, start, goal, weighted, weighted)
            for h in ("zero", "manhattan"):
                for tie in ("fifo", "larger_g"):
                    result = astar(grid, start, goal, weighted=weighted, heuristic=h, tie=tie)
                    assert result.cost == expected, (grid, expected, result)
                    check_path(result, grid, start, goal, weighted, weighted)
                    counts["random_grid_searches"] += 1
    # 检查起终点重合、无路、阻塞和免费终点。
    assert weighted_cost((0,0), (0,0), [["E"]]) == 0
    assert maze_path((0,0), (0,0), [["S"]]) == [(0,0)]
    assert weighted_cost((0,0), (0,1), [["S", "E"]]) == 0
    assert maze_path((0,0), (0,2), [["S", "X", "E"]]) == []
    assert weighted_cost((0,0), (0,2), [["S", "X", "E"]]) == -1
    assert maze_path((0,0), (0,1), [["X", "E"]]) == []
    for bad in ([], [[1],[1,2]]):
        try:
            astar(bad, (0,0), (0,0))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid grid accepted")
    def make_tree(depth):
        if depth == 0 or rng.random() < .15:
            return TreeNode(rng.randint(-30, 30))
        return TreeNode(children=[make_tree(depth-1) for _ in range(rng.randint(1,4))])
    def reference_tree(node, maximizing=True):
        if not node.children:
            return node.value
        return (max if maximizing else min)(reference_tree(c, not maximizing) for c in node.children)
    for _ in range(100):
        root = make_tree(5)
        expected = reference_tree(root)
        for order in ("left", "reverse"):
            for prune in (False, True):
                assert tree_search(root, pruning=prune, order=order)[0] == expected
                counts["random_tree_searches"] += 1
    d, paths = distance_matrix(TREASURE_GRID, POSITIONS)
    for row in paths:
        assert row["cost"] == oracle_distance(TREASURE_GRID, POSITIONS[row["from"]],
                                              POSITIONS[row["to"]], True)
    for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
        for pruning in (False, True):
            result, _ = treasure_search(d, VALUES, depth, pruning=pruning)
            assert result == {"best_target": target, "best_value": value}, result
            counts["sample_checks"] += 1
    # 用随机有向距离检查距离方向和位置更新。
    for _ in range(50):
        names = list(POSITIONS)
        random_d = {a: {b: 0 if a == b else rng.randint(1,15) for b in names} for a in names}
        values = {t: rng.randint(-3,15) for t in VALUES}
        for depth in (1,2,3,4):
            expected, roots = exhaustive_game(random_d, values, depth)
            for order in ("natural", "reverse", "gain"):
                for pruning in (False, True):
                    result, _ = treasure_search(random_d, values, depth, order=order, pruning=pruning)
                    assert result["best_value"] == expected
                    assert roots[result["best_target"]] == expected
                    counts["random_games"] += 1
    result = {"passed": True, "seed": 20261105, **counts,
              "ordered_distance_pairs_checked": len(paths), "boundary_cases": 8}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_verification(Path(__file__).parent / "results/verification.json"), indent=2))
