"""
train.py  –  DQN training script for GridWorld (size 3 to 7)

Features
--------
• Hyper‑parameters via YAML (--cfg) or CLI flags.
• Hard guard: grid size must be between 3 and 7.
• Per‑episode console logging (Reward dict, Loss, ε, Steps).
• Graceful KeyboardInterrupt → saves everything before exit.
• Artifacts written to runs/ :
      checkpoint_final.pt
      episode_rewards.npy
      losses.npy
      run_config.json
"""

import argparse, yaml, json, os
from collections import deque
import numpy as np

from dqn import DQN
from environment import GridWorld
from agent import DQNAgent

STATE_SIZE = 7
ACTION_SIZE = 4

# ---------------------------------------------------------------------
# Default hyper‑params (overridable)
# ---------------------------------------------------------------------
DEFAULTS = dict(
    grid_size=5,
    agents_per_type=[2, 2],
    episodes=20000,
    initial_experience=100,
    reward=dict(
        move=-1,
        invalid_move=-5,
        secret_exchange=20,
        reach_b_with_full_secret=100,
        reach_b_without_secret=-5,
    ),
)

# ---------------------------------------------------------------------
# CLI / YAML helpers
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, help="Path to YAML config.")
    p.add_argument("--grid-size", type=int)
    p.add_argument("--agents-per-type", nargs="+", type=int)
    p.add_argument("--episodes", type=int)
    return p.parse_args()

def load_yaml(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------
# Warm‑up experience
# ---------------------------------------------------------------------
def collect_initial_experiences(agents, env, initial_experience):
    print(f"Collecting {initial_experience} warm‑up steps …")
    steps = 0
    while steps < initial_experience:
        states = env.reset()
        done   = False
        while not done and steps < initial_experience:
            for ag in agents:
                a = ag.act(states[ag.agent_id])
                s2, r, done = env.agent_step(ag.agent_id, a)
                ag.remember(states[ag.agent_id], a, r, s2, done)
                states[ag.agent_id] = s2
            steps += 1
            if done:
                break
    print("Warm‑up done.")

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_agents(agents, env, episodes):
    os.makedirs("runs", exist_ok=True)
    reward_hist, loss_hist = [], []

    try:
        for ep in range(1, episodes + 1):
            states = env.reset()
            done   = False
            episode_rewards = {ag.agent_id: 0 for ag in agents}
            episode_loss    = []
            steps_in_episode = 0

            while not done and steps_in_episode < 100:   # max steps
                for ag in agents:
                    a = ag.act(states[ag.agent_id])
                    s2, r, done = env.agent_step(ag.agent_id, a)
                    ag.remember(states[ag.agent_id], a, r, s2, done)
                    states[ag.agent_id] = s2

                    loss = ag.replay()
                    if loss is not None:
                        episode_loss.append(loss)

                    ag.increment_step()
                    episode_rewards[ag.agent_id] += r
                    if done:
                        break
                steps_in_episode += 1

            # metrics
            total_reward = sum(episode_rewards.values())
            avg_loss     = np.mean(episode_loss) if episode_loss else float("nan")
            reward_hist.append(total_reward)
            loss_hist.append(avg_loss)

            # ε decay
            for ag in agents:
                ag.update_epsilon()

            # logging
            if ep == 1 or ep % 100 == 0:
                print(f"Episode {ep:5d}, Rewards {episode_rewards}, "
                      f"Loss {avg_loss:.4f}, ε={agents[0].epsilon:.3f}, "
                      f"Steps {steps_in_episode}")

    except KeyboardInterrupt:
        print("\n[train] KeyboardInterrupt received — stopping early.")

    # always save artifacts
    agents[0].shared_dqn.save("runs/checkpoint_final.pt")
    np.save("runs/episode_rewards.npy", np.array(reward_hist))
    np.save("runs/losses.npy",         np.array(loss_hist))
    print("[train] Final checkpoint & curves saved to runs/")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    os.makedirs("runs", exist_ok=True)

    args = parse_args()
    cfg = {**DEFAULTS, **load_yaml(args.cfg)}
    if args.grid_size:
        cfg["grid_size"] = args.grid_size
    if args.agents_per_type:
        cfg["agents_per_type"] = args.agents_per_type
    if args.episodes:
        cfg["episodes"] = args.episodes

    if not (3 <= cfg["grid_size"] <= 7):
        raise SystemExit("grid_size must be between 3 and 7.")

    grid  = cfg["grid_size"]
    types = cfg["agents_per_type"]
    n_agents = sum(types)

    dqns    = {t: DQN(STATE_SIZE, ACTION_SIZE) for t in range(len(types))}
    buffers = {t: deque(maxlen=1000)          for t in range(len(types))}
    agents, t_list, aid = [], [], 0
    for t, count in enumerate(types):
        for _ in range(count):
            agents.append(DQNAgent(aid, t, dqns[t], buffers[t], ACTION_SIZE))
            t_list.append(t); aid += 1

    env = GridWorld(size=grid, n_agents=n_agents,
                    agent_types=t_list, reward=cfg["reward"])

    collect_initial_experiences(agents, env, cfg["initial_experience"])
    train_agents(agents, env, cfg["episodes"])

    with open("runs/run_config.json", "w") as f:
        json.dump(dict(grid_size=grid,
                       agents_per_type=types,
                       state_size=STATE_SIZE,
                       action_size=ACTION_SIZE), f, indent=2)
    print("run_config.json written.")

if __name__ == "__main__":
    main()
