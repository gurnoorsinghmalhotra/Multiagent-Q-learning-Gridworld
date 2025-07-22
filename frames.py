# frame_viewer.py – scrub through PNG frames with a slider
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob, sys, os

folder = sys.argv[1] if len(sys.argv) > 1 else "frames_tmp"
pngs   = sorted(glob.glob(os.path.join(folder, "f*.png")))
if not pngs:
    raise SystemExit(f"No PNG files found in {folder}")

img = plt.imread(pngs[0])
fig, ax = plt.subplots()
im_obj = ax.imshow(img)
ax.axis("off")

slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(slider_ax, "frame", 0, len(pngs)-1,
                valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    im_obj.set_data(plt.imread(pngs[idx]))
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
