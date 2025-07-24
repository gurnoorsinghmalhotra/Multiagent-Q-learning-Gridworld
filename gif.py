"""
gif.py - Create a GIF animation of the GridWorld agents' paths.
This script builds a GridWorld environment, initializes agents, and simulates their actions
to create a visual representation of their paths towards a goal.
It saves the resulting frames as a GIF file.
Usage:
    python gif.py --ckpt <checkpoint_path> --cfg <config_path> --out <output_gif_path> [--clean] [--tmp <temp_dir>] [--manual "bx,by a0x,a0y ..."]
Arguments:
    --ckpt   : Path to the DQN model checkpoint.
    --cfg    : Path to the configuration file for the environment.
    --out    : Output path for the generated GIF.
    --clean  : If set, cleans up temporary frames after GIF creation.
    --tmp    : Temporary directory for storing individual frames.
    --manual : Manually specify the initial positions of agents and the goal.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
import numpy as np
import argparse, json, random, shutil, pathlib
from environment import GridWorld
from dqn import DQN
from agent import DQNAgent
from collections import deque

STATE, ACTION = 7, 4

# ---------- colormap helper ----------
def cmap_resampled(name, n):
    base = cm.get_cmap(name)
    try:                    return base.resampled(n)
    except AttributeError:  return ListedColormap(base(np.linspace(0,1,n)))

# ---------- plotting helpers ----------
def plot_initial(env):
    """Return fig, ax, agent_colors (like your notebook version)."""
    amap = env.get_map_array()
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = cmap_resampled("viridis", env.n_agents + 2)
    im   = ax.imshow(amap, cmap=cmap, interpolation="none")

    agent_colors = [im.cmap(im.norm(i + 1)) for i in range(env.n_agents)]

    ax.grid(which="minor", lw=.8, color="black")
    ax.set_xticks(np.arange(amap.shape[1]) + .5, minor=True)
    ax.set_yticks(np.arange(amap.shape[0]) + .5, minor=True)
    ax.set_xticks([]); ax.set_yticks([])

    pos_dict = {}
    for i, pos in enumerate(env.agent_positions):
        lbl = f'a{i+1}' if env.agent_types[i]==0 else f'A{i+1}'
        pos_dict.setdefault(pos, []).append(lbl)
    for pos, labs in pos_dict.items():
        ax.text(pos[0], pos[1], ",".join(labs),
                ha="center", va="center",
                fontsize=12, color="black", fontweight="bold")

    ax.text(env.b_location[0], env.b_location[1], 'B',
            ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")
    return fig, ax, agent_colors

def plot_paths(ax, env, agent_colors, off=.15):
    used={}
    for i, path in enumerate(env.paths):
        col = agent_colors[i]
        for (x0,y0),(x1,y1) in zip(path[:-1], path[1:]):
            k=((x0,y0),(x1,y1)); rk=((x1,y1),(x0,y0))
            prev = used.get(k, used.get(rk))
            dx,dy = (x1-x0+off, y1-y0+off) if prev and prev!=i else (x1-x0, y1-y0)
            ax.arrow(x0+off/2, y0+off/2, dx, dy,
                     head_width=.2, head_length=.2,
                     fc=col, ec=col, lw=1.5,
                     length_includes_head=True)
            used[k]=i

# ---------- env + agents ----------
def build_env(ckpt,cfg):
    meta=json.load(open(cfg))
    grid,types=meta["grid_size"],meta["agents_per_type"]
    n=sum(types)

    dqn=DQN(STATE,ACTION); dqn.load(ckpt)
    buf={t:deque(maxlen=1) for t in range(len(types))}
    agents,tlist,aid=[],[],0
    for t,c in enumerate(types):
        for _ in range(c):
            agents.append(DQNAgent(aid,t,dqn,buf[t],ACTION,epsilon=0))
            tlist.append(t); aid+=1

    zeros=dict(move=0,invalid_move=0,secret_exchange=0,
               reach_b_with_full_secret=0,reach_b_without_secret=0)
    env=GridWorld(size=grid,n_agents=n,agent_types=tlist,reward=zeros)
    return env,agents

def random_config(env):
    cells=[(x,y) for x in range(env.size) for y in range(env.size)]
    random.shuffle(cells)
    return cells.pop(), [cells.pop() for _ in range(env.n_agents)]

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--ckpt", default="runs/checkpoint_final.pt")
    ap.add_argument("--cfg",  default="runs/run_config.json")
    ap.add_argument("--out",  default="assets/demo.gif")
    ap.add_argument("--tmp",  default="frames_tmp")
    ap.add_argument("--manual", help='"bx,by a0x,a0y ..."')
    args=ap.parse_args()

    env,agents=build_env(args.ckpt,args.cfg)
    if args.manual:
        parts=args.manual.split()
        B=tuple(map(int,parts[0].split(",")))
        A=[tuple(map(int,p.split(","))) for p in parts[1:]]
    else:
        B,A=random_config(env)

    env.reset(); env.b_location=B; env.agent_positions=list(A)
    env.paths=[[] for _ in range(env.n_agents)]
    for i in range(env.n_agents):
        env.paths[i].append(env.agent_positions[i])

    tmp=pathlib.Path(args.tmp)
    if args.clean and tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(exist_ok=True)
    frames=[]

    task="?"
    done=False; step=0
    while not done and step<100:
        fig,ax,cols=plot_initial(env)
        ax.set_title(f"Step {step}   |   Task completed by -  {task}")
        plot_paths(ax,env,cols)
        fname=tmp/f"f{step:03d}.png"
        fig.savefig(fname,dpi=120,bbox_inches="tight"); plt.close(fig)
        frames.append(str(fname))

        for ag in agents:
            a=ag.act(env.get_state(ag.agent_id))
            _,_,done=env.agent_step(ag.agent_id,a)
            env.paths[ag.agent_id].append(env.agent_positions[ag.agent_id])
            if done:
                task=('a' if env.agent_types[ag.agent_id]==0 else 'A')+str(ag.agent_id+1)
                break
        step+=1

    fig,ax,cols=plot_initial(env)
    ax.set_title(f"Step {step}   |   Task completed by -  {task}")
    plot_paths(ax,env,cols)
    fname=tmp/f"f{step:03d}.png"
    fig.savefig(fname,dpi=120,bbox_inches="tight"); plt.close(fig)
    frames.append(str(fname))

    pathlib.Path(args.out).parent.mkdir(exist_ok=True)
    with imageio.get_writer(args.out, mode="I", fps=1, loop=0) as w:
        for f in frames:
            w.append_data(imageio.imread(f))
    print("GIF saved to", args.out)

    if input("Keep individual PNG frames? (y/N) ").lower()!="y":
        shutil.rmtree(tmp)

if __name__=="__main__":
    main()
