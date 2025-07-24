"""
instruct.py
Quick command guide for the Multi-Agent Q-Learning GridWorld project.

Run:
    python instruct.py
"""

import textwrap

GUIDE = textwrap.dedent("""
====================================================================
 Multi-Agent Q-Learning GridWorld  —  Command Reference
====================================================================

SETUP
-----
1) Create/activate venv, install deps:
   python -m venv venv
   venv\\Scripts\\activate        (Windows)   |  source venv/bin/activate (Linux/Mac)
   pip install -r requirements.txt

2) Optional: open VS Code here
   code .

--------------------------------------------------------------------
TRAIN
-----
python train.py [--cfg CONFIG.YAML] [--grid-size N] [--agents-per-type a b ...] [--episodes M]

Outputs to runs/ :
  checkpoint_final.pt
  episode_rewards.npy
  losses.npy
  epsilons.npy
  run_config.json

Ctrl+C during training saves everything gracefully.

--------------------------------------------------------------------
EVALUATE (exhaustive configs)
-----------------------------
python evaluate.py [--step-threshold 15] [--max-steps 100]

Prints success rate, avg steps, AEP, etc.

--------------------------------------------------------------------
PLOTS (reward / loss / epsilon)
-------------------------------
python plots.py [--run runs] [--out assets/plots]

Saves:
  assets/plots/reward_curve.png
  assets/plots/loss_curve.png
  assets/plots/epsilon_curve.png
  assets/plots/loss_curve.png

--------------------------------------------------------------------
GIF / FRAMES (visualise an episode)
-----------------------------------
# Generate GIF directly (random or manual layout):
python gif.py               # defaults (uses runs/checkpoint_final.pt)
python gif.py --manual "bx,by a0x,a0y a1x,a1y ..."  (custom start)
  Options:
    --ckpt PATH    model checkpoint
    --cfg  PATH    run_config.json
    --out  PATH    output gif path (default assets/demo.gif)
    --tmp  DIR     temp frame folder
    --clean        delete temp frames afterward

# Alternative frame-by-frame export + stitch (PNG -> GIF):
python make_gif_frames.py [--clean] [--manual "..."]
  (Same flags as gif.py, saves frames in frames_tmp/ then demo.gif)

# Optional frame viewer (if you added one, e.g. frames.py):
python frames.py frames_tmp

--------------------------------------------------------------------
UTILITY
-------
View this help again:
  python instruct.py

Clean temp frames manually:
  python -c "import shutil; shutil.rmtree('frames_tmp', True)"

Check defaults (copy into a YAML to edit):
  # see config/default.yaml or the block inside train.py DEFAULTS

====================================================================
""").strip()


def main():
    print(GUIDE)


if __name__ == "__main__":
    main()
