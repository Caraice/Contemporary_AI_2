"""Minimax 与 Alpha-beta 搜索，记录位置和累计收益。"""
from dataclasses import dataclass, field, replace
from math import inf
from search import astar


@dataclass
class TreeNode:
    value: int | None = None
    children: list = field(default_factory=list)


@dataclass
class GameStats:
    visited: int = 0  # 递归调用次数，含叶子
    leaves: int = 0
    cutoffs: int = 0  # 实际跳过剩余孩子的截断次数
    skipped_children: int = 0  # 仅统计直接跳过的孩子
    trace: list = field(default_factory=list)


def tree_search(root, *, pruning=True, order="left", record=False):
    if order not in ("left", "reverse"):
        raise ValueError("unknown order")
    stats = GameStats()

    def visit(node, maximizing, alpha, beta, address):
        stats.visited += 1
        if not node.children:
            if node.value is None:
                raise ValueError("leaf must have a value")
            stats.leaves += 1
            return node.value
        children = list(enumerate(node.children))
        if order == "reverse":
            children.reverse()
        value = -inf if maximizing else inf
        for i, (index, child) in enumerate(children):
            v = visit(child, not maximizing, alpha, beta, address + (index,))
            value = max(value, v) if maximizing else min(value, v)
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if pruning and alpha >= beta and i + 1 < len(children):
                stats.cutoffs += 1
                stats.skipped_children += len(children) - i - 1
                if record:
                    stats.trace.append({"node": address, "alpha": alpha,
                                        "beta": beta, "skipped": len(children)-i-1})
                break
        return value

    return visit(root, True, -inf, inf, ()), stats


def alpha_beta(root):
    return tree_search(root)[0]


def sample_tree():
    def branch(a, b):
        return TreeNode(children=[TreeNode(a), TreeNode(b)])
    return TreeNode(children=[TreeNode(children=[branch(3, 5), branch(6, 9)]),
                              TreeNode(children=[branch(6, 9), branch(1, 2)])])


def tree_from_dict(data):
    data = data.get("tree_structure", data)
    return TreeNode(data.get("value"), [tree_from_dict(c) for c in data.get("children", [])])


def distance_matrix(grid, positions):
    """用 A* 计算全部有向距离，不假设对称。"""
    distances = {}
    searches = []
    for name, start in positions.items():
        distances[name] = {}
        for target, goal in positions.items():
            result = astar(grid, start, goal, weighted=True, goal_cost=None)
            distances[name][target] = result.cost
            searches.append({"from": name, "to": target, "cost": result.cost,
                             "expanded": result.expanded, "path": result.path})
    return distances, searches


@dataclass(frozen=True)
class State:
    pos_a: str = "A"
    pos_b: str = "B"
    remaining: tuple = ("T1", "T2", "T3", "T4")
    score_a: int = 0
    score_b: int = 0
    player: str = "A"

    @property
    def utility(self):
        return self.score_a - self.score_b


def transition(state, target, distances, values):
    pos = state.pos_a if state.player == "A" else state.pos_b
    cost = distances[pos][target]
    if cost < 0 or target not in state.remaining:
        raise ValueError("unreachable or already collected target")
    changes = {"remaining": tuple(t for t in state.remaining if t != target)}
    if state.player == "A":
        changes.update(pos_a=target, score_a=state.score_a+values[target]-cost, player="B")
    else:
        changes.update(pos_b=target, score_b=state.score_b+values[target]-cost, player="A")
    return replace(state, **changes)


def treasure_search(distances, values, depth, *, pruning=True, order="natural", state=None):
    """同分取先访问者，有可达目标时必须行动。

    排序只用即时收益；截断值可能为界，根节点仅在严格改优时换目标。
    """
    if depth < 0 or order not in ("natural", "reverse", "gain"):
        raise ValueError("invalid depth/order")
    state = state or State(remaining=tuple(values))
    stats = GameStats()

    def visit(s, h, alpha, beta):
        stats.visited += 1
        pos = s.pos_a if s.player == "A" else s.pos_b
        moves = [t for t in s.remaining if distances[pos][t] >= 0]
        if h == 0 or not moves:
            stats.leaves += 1
            return s.utility, None
        if order == "reverse":
            moves.reverse()
        elif order == "gain":
            moves.sort(key=lambda t: -(values[t] - distances[pos][t]))
        maximizing = s.player == "A"
        best, target = (-inf if maximizing else inf), None
        for i, t in enumerate(moves):
            value, _ = visit(transition(s, t, distances, values), h-1, alpha, beta)
            if (maximizing and value > best) or (not maximizing and value < best):
                best, target = value, t
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if pruning and alpha >= beta and i + 1 < len(moves):
                stats.cutoffs += 1
                stats.skipped_children += len(moves) - i - 1
                break
        return best, target

    value, target = visit(state, depth, -inf, inf)
    return {"best_target": target, "best_value": value}, stats
