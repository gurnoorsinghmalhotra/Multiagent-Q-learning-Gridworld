# plots.py
"""Plot reward and loss curves saved by train.py."""
import argparse, pathlib, numpy as np, matplotlib.pyplot as plt

def main(run_dir="runs"):
    run = pathlib.Path(run_dir)
    r = np.load(run / "episode_rewards.npy")
    l = np.load(run / "losses.npy")

    plt.figure()
    plt.plot(r); plt.title("Episode Reward"); plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(run / "rewards.png", dpi=150)

    plt.figure()
    plt.plot(l); plt.title("Episode Loss"); plt.xlabel("Episode"); plt.ylabel("Loss")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(run / "loss.png", dpi=150)

    print("Saved rewards.png and loss.png to", run)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs")
    args = p.parse_args()
    main(args.run)
