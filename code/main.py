"""运行入口：python main.py [1|2|3|4] [--input input.json]。"""
import argparse
import json
from cases import MAZE, WEIGHTED, TREASURE_GRID, POSITIONS, VALUES
from search import maze_path, weighted_cost
from game import alpha_beta, sample_tree, tree_from_dict, distance_matrix, treasure_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=["1", "2", "3", "4"])
    parser.add_argument("--input", help="JSON file; omit to run the assignment sample")
    args = parser.parse_args()
    data = {}
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    if args.problem == "1":
        result = maze_path(data.get("start", (0, 0)), data.get("goal", (3, 4)), data.get("grid", MAZE))
    elif args.problem == "2":
        result = weighted_cost(data.get("start", (0, 0)), data.get("goal", (4, 4)), data.get("grid", WEIGHTED))
    elif args.problem == "3":
        result = alpha_beta(tree_from_dict(data) if data else sample_tree())
    else:
        d, _ = distance_matrix(data.get("grid", TREASURE_GRID), data.get("positions", POSITIONS))
        values = data.get("values", VALUES)
        result = {"labels": list(d), "distance_matrix": [list(row.values()) for row in d.values()]}
        for h in (2, 4):
            result[f"H={h}"] = {name: treasure_search(d, values, h, pruning=prune)[0]
                                for name, prune in (("Minimax", False), ("Alpha-beta", True))}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
