import zarr
import numpy as np
import imageio
from pathlib import Path
import matplotlib.pyplot as plt

ZARR_PATH = "datasets/infer/p2igan_test.zarr"
OUT_GIF = "preview_first23.gif"
NUM_FRAMES = 241
FPS = 4

z = zarr.open(ZARR_PATH, mode="r")

# 取第一個 event
event_key = sorted(z.array_keys())[0]
data = z[event_key][:NUM_FRAMES]  # shape: (T, C, H, W)

frames = []
for t in range(data.shape[0]):
    frame = data[t]
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]

    vmin = float(frame.min())
    vmax = float(frame.max())
    vmean = float(frame.mean())

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"t={t}\nmin={vmin:.3f} max={vmax:.3f} mean={vmean:.3f}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(img)

    plt.close(fig)

imageio.mimsave(OUT_GIF, frames, fps=FPS)

print(f"Saved GIF to {OUT_GIF}")
