"""
evaluate.py    Gridsize agnostic evaluation for any GridWorld run.

Reads:
    • checkpoint .pt
    • runs/run_config.json      (grid size & agent layout)

Outputs success rate, average steps, % under a threshold, and AEP.
"""

import argparse, json, itertools, pathlib
from environment import GridWorld
from dqn import DQN
from agent import DQNAgent
from replay_buffer import ReplayBuffer
import math
 


STATE_SIZE  = 7
ACTION_SIZE = 4

# ---------------------------------------------------------------------
# Symmetry‑based representative positions
# ---------------------------------------------------------------------
def unique_positions(size: int):
    pos = set()
    mid = size // 2
    if size % 2 == 1:
        pos.add((mid, mid))
    pos.update([(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)])
    pos.update([(mid, 0), (mid, size - 1), (0, mid), (size - 1, mid)])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0: continue
            x, y = mid + dx, mid + dy
            if 0 <= x < size and 0 <= y < size:
                pos.add((x, y))
    return sorted(pos)

def estimate_config_count(env):
    """
    Return the total number of configurations that
    generate_configurations(env) will yield.
    """
    n_pos = len(unique_positions(env.size))
    n_ent = env.n_agents + 1          # B  +  agents
    combos = math.comb(n_pos, n_ent)  # position sets
    perms  = math.factorial(n_ent)    # B vs. agents ordering
    return combos * perms


def configs(env):
    upos = unique_positions(env.size)
    n_ent = env.n_agents + 1
    for combo in itertools.combinations(upos, n_ent):
        for perm in itertools.permutations(combo):
            yield {'b': perm[0], 'agents': perm[1:]}

# ---------------------------------------------------------------------
# Optimal Manhattan steps 
# ---------------------------------------------------------------------
def optimal_steps(cfg, types):
    agents = cfg['agents']
    b = cfg['b']
    best = float("inf")
    for i, a in enumerate(agents):
        for j, c in enumerate(agents):
            if types[i] != types[j]:
                D_ab = abs(a[0]-c[0]) + abs(a[1]-c[1])
                meet = (D_ab + 1)//2
                m = ((a[0]+c[0])//2, (a[1]+c[1])//2)
                D_mb = abs(m[0]-b[0]) + abs(m[1]-b[1])
                best = min(best, meet + D_mb)
    return best

# ---------------------------------------------------------------------
# Build objects from checkpoint + json
# ---------------------------------------------------------------------
def build(checkpoint, run_json):
    with open(run_json, "r") as f:
        meta = json.load(f)
    grid, types = meta["grid_size"], meta["agents_per_type"]
    n_agents = sum(types)

    dqn = DQN(STATE_SIZE, ACTION_SIZE)
    dqn.load(checkpoint)

    bufs = {t: ReplayBuffer(maxlen=1) for t in range(len(types))}
    agents, t_list, aid = [], [], 0
    for t, cnt in enumerate(types):
        for _ in range(cnt):
            agents.append(DQNAgent(aid, t, dqn, bufs[t], ACTION_SIZE, epsilon=0.0))
            t_list.append(t); aid += 1

    env = GridWorld(size=grid, n_agents=n_agents, agent_types=t_list, reward=dict(move=0, invalid_move=0, secret_exchange=0, reach_b_with_full_secret=0, reach_b_without_secret=0))
    return env, agents

# ---------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------
def evaluate(env, agents, step_thr=15, max_steps=100):
    cfg_total = succ = steps_sum = under = exc_sum = 0

    for cfg in configs(env):
        cfg_total += 1
        env.reset()
        env.b_location     = cfg['b']
        env.agent_positions = list(cfg['agents'])
        env.has_full_secret = [False] * env.n_agents
        env.paths           = [[] for _ in range(env.n_agents)]

        opt = optimal_steps(cfg, env.agent_types)
        done = False
        steps = 0
        while not done and steps < max_steps:
            for ag in agents:
                s = env.get_state(ag.agent_id)
                a = ag.act(s)
                _, _, done = env.agent_step(ag.agent_id, a)
                if done: break
            steps += 1
            if done or steps > step_thr:
                break

        steps_sum += steps
        if done:
            succ += 1
            exc_sum += max(0, steps - opt)
            if steps <= step_thr:
                under += 1

        if cfg_total % 5000 == 0:
            print(f"Processed {cfg_total} configs …")

    print(f"\nSuccess:  {succ/cfg_total*100:.2f}%")
    print(f"AvgStep:  {steps_sum/cfg_total:.2f}")
    print(f"≤{step_thr}: {under/cfg_total*100:.2f}%")
    print("AEP:      {:.4f}".format(exc_sum/succ) if succ else "AEP: n/a")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/checkpoint_final.pt")
    p.add_argument("--run-config", default="runs/run_config.json")
    p.add_argument("--step-threshold", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=100)
    args = p.parse_args()

    if not pathlib.Path(args.checkpoint).exists():
        raise SystemExit("Checkpoint not found.")
    if not pathlib.Path(args.run_config).exists():
        raise SystemExit("run_config.json not found.")

    env, agents = build(args.checkpoint, args.run_config)

    total_cfgs = estimate_config_count(env)
    print(f"\nThis evaluation will enumerate {total_cfgs:,} configurations "
        f"on a {env.size}x{env.size} grid with {env.n_agents} agents.\n")

    evaluate(env, agents, args.step_threshold, args.max_steps)


if __name__ == "__main__":
    main()
