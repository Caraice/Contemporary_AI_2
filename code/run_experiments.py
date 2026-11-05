"""复现实验，保存数据、搜索轨迹和图表。"""
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from cases import MAZE, WEIGHTED, TREASURE_GRID, POSITIONS, VALUES
from search import astar
from game import (TreeNode, sample_tree, tree_search, distance_matrix, treasure_search,
                  State, transition)
from verify import run_verification, exhaustive_game

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(fn, repeats=31):
    fn()  # 预热，不计时
    times = []
    for _ in range(repeats):
        start = perf_counter_ns()
        result = fn()
        times.append((perf_counter_ns()-start)/1e6)
    return result, statistics.median(times), times


def principal_variation(d, depth):
    # 用完整搜索还原最优路线，不计入性能测量。
    state = State()
    rows = []
    for ply in range(depth):
        decision, _ = treasure_search(d, VALUES, depth-ply, pruning=False, state=state)
        target = decision["best_target"]
        old = state
        state = transition(state, target, d, VALUES)
        pos = old.pos_a if old.player == "A" else old.pos_b
        rows.append({"ply": ply+1, "player": old.player, "from": pos, "target": target,
                     "cost": d[pos][target], "gain": VALUES[target]-d[pos][target],
                     "score_a": state.score_a, "score_b": state.score_b, "utility": state.utility})
    return rows


def make_grid(kind, size, seed):
    rng = random.Random(seed)
    grid = [[1 for _ in range(size)] for _ in range(size)]
    if kind == "wall_detour":
        for r in range(size-1):
            grid[r][size//2] = "X"
    elif kind == "random_weighted":
        grid = [["X" if rng.random()<.2 else rng.choice([1,1,1,3,8])
                 for _ in range(size)] for _ in range(size)]
        # 打通通道，保留随机地形代价。
        for r in range(size):
            if grid[r][0] == "X":
                grid[r][0] = rng.choice([1,3,8])
        for c in range(size):
            if grid[-1][c] == "X":
                grid[-1][c] = rng.choice([1,3,8])
    return grid


def figures(search_rows, tree_rows, game_rows, d, extensions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(10,4), layout="constrained")
    for ax, name in zip(axes, ("maze", "weighted")):
        rows = [r for r in search_rows if r["problem"] == name]
        ax.bar(range(len(rows)), [r["expanded"] for r in rows], color=["#94a3b8", "#3b82f6", "#0f766e"])
        ax.set_xticks(range(len(rows)), [r["strategy"] for r in rows], rotation=15)
        ax.set(title=name, ylabel="Expanded nodes (goal included)")
        for i,r in enumerate(rows):
            ax.text(i,r["expanded"],str(r["expanded"]),ha="center",va="bottom")
    fig.savefig(figdir/"astar_samples.png"); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax, rows, title in [(axes[0],tree_rows,"Problem 3"),
                             (axes[1],[r for r in game_rows if r["depth"]==2],"Treasure H=2"),
                             (axes[2],[r for r in game_rows if r["depth"]==4],"Treasure H=4")]:
        ax.bar(range(len(rows)),[r["visited"] for r in rows],color=["#94a3b8" if r["algorithm"]=="Minimax" else "#0f766e" for r in rows])
        ax.set_xticks(range(len(rows)),[r["algorithm"]+"\n"+r["order"] for r in rows],rotation=45,ha="right")
        ax.set(title=title,ylabel="Recursive calls")
    fig.savefig(figdir/"game_comparison.png"); plt.close(fig)
    labels=list(d)
    matrix=[list(row.values()) for row in d.values()]
    fig,ax=plt.subplots(figsize=(6,5),layout="constrained")
    im=ax.imshow(matrix,cmap="Blues")
    ax.set_xticks(range(6),labels); ax.set_yticks(range(6),labels)
    ax.set(title="Minimum entry-cost distance",xlabel="To",ylabel="From")
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):
            ax.text(j,i,str(v),ha="center",va="center",color="white" if v>7 else "black")
    fig.colorbar(im,ax=ax,label="Cost"); fig.savefig(figdir/"distance_matrix.png");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout="constrained")
    for ax,kind in zip(axes,("open","wall_detour","random_weighted")):
        for strategy in ("zero/fifo","manhattan/fifo","manhattan/larger_g"):
            subset=[r for r in extensions if r["kind"]==kind and r["strategy"]==strategy]
            sizes=sorted(set(r["size"] for r in subset))
            y=[statistics.mean(r["expanded"] for r in subset if r["size"]==n) for n in sizes]
            ax.plot(sizes,y,"o-",label=strategy)
        ax.set(title=kind,xlabel="Grid side",ylabel="Mean expanded nodes")
    axes[-1].legend(fontsize=7)
    fig.savefig(figdir/"extension_grids.png");plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    run_verification(OUT/"verification.json")
    timing_raw=[]
    samples={}
    search_rows=[]
    for name, grid, start, goal, weighted in [("maze",MAZE,(0,0),(3,4),False),
                                               ("weighted",WEIGHTED,(0,0),(4,4),True)]:
        for h,tie in (("zero","fifo"),("manhattan","fifo"),("manhattan","larger_g")):
            fn=lambda: astar(grid,start,goal,weighted=weighted,heuristic=h,tie=tie)
            result,ms,times=measure(fn)
            search_rows.append({"problem":name,"strategy":h+"/"+tie,"cost":result.cost,
                                "expanded":result.expanded,"generated":result.generated,"median_ms":ms})
            timing_raw.append({"experiment":name+"/"+h+"/"+tie,"times_ms":times})
        result=astar(grid,start,goal,weighted=weighted,record=True)
        samples[name]=asdict(result)
        cumulative=0
        step_rows=[]
        for i,p in enumerate(result.path):
            step=0 if i==0 or weighted and p==goal else int(grid[p[0]][p[1]]) if weighted else 1
            cumulative+=step
            step_rows.append({"step":i,"row":p[0],"col":p[1],"entry_cost":step,"cumulative":cumulative})
        save_csv(name+"_path.csv",step_rows)
    save_csv("astar_comparison.csv",search_rows)
    tree_rows=[]
    for pruning in (False,True):
        for order in ("left","reverse"):
            result,ms,times=measure(lambda:tree_search(sample_tree(),pruning=pruning,order=order))
            value,stats=result
            row={"algorithm":"Alpha-beta" if pruning else "Minimax","order":order,
                 "value":value,**{k:v for k,v in asdict(stats).items() if k!="trace"},"median_ms":ms}
            tree_rows.append(row)
            timing_raw.append({"experiment":"tree/"+row["algorithm"]+"/"+order,"times_ms":times})
    samples["tree_cutoffs"]=asdict(tree_search(sample_tree(),record=True)[1])
    save_csv("tree_comparison.csv",tree_rows)
    # 构造顺序敏感的树，反序先访问较优分支。
    extra_tree = TreeNode(children=[TreeNode(children=[TreeNode(1),TreeNode(2),TreeNode(3)]),
                                    TreeNode(children=[TreeNode(9),TreeNode(8),TreeNode(7)])])
    extra_rows=[]
    for pruning in (False,True):
        for order in ("left","reverse"):
            value,stats=tree_search(extra_tree,pruning=pruning,order=order)
            extra_rows.append({"algorithm":"Alpha-beta" if pruning else "Minimax","order":order,
                               "value":value,**{k:v for k,v in asdict(stats).items() if k!="trace"}})
    save_csv("extension_tree_order.csv",extra_rows)
    save_json("extension_tree_input.json",{"children":[{"children":[{"value":x} for x in [1,2,3]]},
                                                                     {"children":[{"value":x} for x in [9,8,7]]}]})
    (d,paths),matrix_ms,matrix_times=measure(lambda:distance_matrix(TREASURE_GRID,POSITIONS))
    save_csv("distance_matrix.csv",[{"from":name,**row} for name,row in d.items()])
    save_json("distance_paths.json",paths)
    timing_raw.append({"experiment":"all_36_astar_pairs","times_ms":matrix_times})
    game_rows=[]
    root_rows=[]
    for depth in (2,4):
        _,roots=exhaustive_game(d,VALUES,depth)
        for t,value in roots.items():
            root_rows.append({"depth":depth,"first_target":t,"exact_value":value})
        save_csv(f"optimal_line_H{depth}.csv",principal_variation(d,depth))
        for pruning in (False,True):
            for order in ("natural","reverse","gain"):
                result,ms,times=measure(lambda:treasure_search(d,VALUES,depth,pruning=pruning,order=order))
                decision,stats=result
                game_rows.append({"depth":depth,"algorithm":"Alpha-beta" if pruning else "Minimax",
                                  "order":order,**decision,**{k:v for k,v in asdict(stats).items() if k!="trace"},
                                  "median_ms":ms})
                timing_raw.append({"experiment":f"treasure/{depth}/{pruning}/{order}","times_ms":times})
    save_csv("treasure_comparison.csv",game_rows)
    save_csv("root_action_values.csv",root_rows)
    extensions=[]
    scenarios=[]
    for kind in ("open","wall_detour","random_weighted"):
        for size in (20,40):
            for seed in ((101,202,303) if kind=="random_weighted" else (101,)):
                grid=make_grid(kind,size,seed)
                goal=(0,size-1) if kind=="wall_detour" else (size-1,size-1)
                scenarios.append({"kind":kind,"size":size,"seed":seed,"grid":grid,
                                  "start":(0,0),"goal":goal})
                costs=[]
                for h,tie in (("zero","fifo"),("manhattan","fifo"),("manhattan","larger_g")):
                    fn=lambda:astar(grid,(0,0),goal,weighted=True,goal_cost=None,heuristic=h,tie=tie)
                    result,ms,times=measure(fn,repeats=7)
                    costs.append(result.cost)
                    extensions.append({"kind":kind,"size":size,"seed":seed,"strategy":h+"/"+tie,
                                       "cost":result.cost,"expanded":result.expanded,"generated":result.generated,
                                       "median_ms":ms})
                    timing_raw.append({"experiment":f"extension/{kind}/{size}/{seed}/{h}/{tie}","times_ms":times})
                assert len(set(costs))==1
    save_csv("extension_grids.csv",extensions)
    save_json("extension_inputs.json",scenarios)
    save_json("sample_results.json",samples)
    save_json("timing_raw.json",timing_raw)
    save_json("environment.json",{
        "python":sys.version,"platform":platform.platform(),"executable":sys.executable,
        "random_seeds":[20261105,101,202,303],"sample_timing_repeats":31,"extension_timing_repeats":7,
        "timing":"one warmup excluded; median wall time; parsing, file IO and plotting excluded; matrix precomputation excluded from game timing",
        "distance_matrix_median_ms":matrix_ms,
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
        "inputs_sha256":hashlib.sha256(json.dumps({"maze":MAZE,"weighted":WEIGHTED,"treasure":TREASURE_GRID,"positions":POSITIONS,"values":VALUES},sort_keys=True).encode()).hexdigest()})
    figures(search_rows,tree_rows,game_rows,d,extensions)
    print("All samples and independent checks passed; results saved to",OUT)
    for row in game_rows:
        if row["order"] in ("natural","gain"):
            print(row)


if __name__=="__main__":
    main()
