"""
plots.py
Plot training curves saved by train.py

Expected files inside <run_dir>:
    episode_rewards.npy     shape (E,)
    losses.npy              shape (E,)
    epsilons.npy            optional, shape (E,) or (E, n_agents)

Outputs PNGs under assets/plots/ :
    reward_curve.png
    loss_curve.png
    epsilon_curve.png
"""

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def _save(fig, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(run_dir="runs", out_dir="assets/plots"):
    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(out_dir)

    # ---- load mandatory arrays --------------------------------------
    rewards = np.load(run_dir / "episode_rewards.npy")
    losses  = np.load(run_dir / "losses.npy")

    # ---- reward plot -------------------------------------------------
    fig_r, ax_r = plt.subplots()
    ax_r.plot(rewards)
    ax_r.set(title="Episode Reward", xlabel="Episode", ylabel="Reward")
    ax_r.grid(alpha=.3)
    _save(fig_r, out_dir / "reward_curve.png")

    # ---- loss plot ---------------------------------------------------
    fig_l, ax_l = plt.subplots()
    ax_l.plot(losses)
    ax_l.set(title="Episode Loss", xlabel="Episode", ylabel="Loss")
    ax_l.grid(alpha=.3)
    _save(fig_l, out_dir / "loss_curve.png")

    # ---- epsilon plot (optional) ------------------------------------
    eps_path = run_dir / "epsilons.npy"
    if eps_path.exists():
        eps = np.load(eps_path)

        fig_e, ax_e = plt.subplots()
        if eps.ndim == 1:
            ax_e.plot(eps, label="ε")
        else:  # multiple agents   plot each
            for k in range(eps.shape[1]):
                ax_e.plot(eps[:, k], label=f"agent {k}")
            ax_e.legend()
        ax_e.set(title="Epsilon Decay", xlabel="Episode", ylabel="ε")
        ax_e.grid(alpha=.3)
        _save(fig_e, out_dir / "epsilon_curve.png")
        print("Saved epsilon_curve.png to", out_dir)
    else:
        print("[plots] epsilons.npy not found  skipping epsilon plot.")

    print("Saved reward_curve.png and loss_curve.png to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs",
                        help="directory produced by train.py")
    parser.add_argument("--out", default="assets/plots",
                        help="where PNGs are written")
    args = parser.parse_args()
    main(args.run, args.out)
