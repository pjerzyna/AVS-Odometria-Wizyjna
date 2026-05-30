"""
Visualization Module
---------------------
Helpers for displaying:
  - Feature matches between two images (side-by-side + connecting lines)
  - Accumulated visual odometry trajectory with direction arrows
  - Per-frame keypoint overlays
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import cv2


# =============================================================
#  Side-by-side match display
# =============================================================

def plot_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    matches: list,
    title: str = "Feature Matches After RANSAC",
    max_display: int = 80,
    figsize: tuple = (16, 7),
) -> plt.Figure:
    """
    Draw two images side-by-side with lines connecting matched keypoints.

    Parameters
    ----------
    img1, img2   : grayscale uint8 images
    matches      : list of ((r1, c1), (r2, c2)) or ((r1,c1),(r2,c2), dist)
    title        : figure title
    max_display  : cap the number of drawn lines to avoid clutter
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas_rgb)
    ax.axis("off")

    cmap = plt.cm.hsv
    n = min(len(matches), max_display)
    for k, match in enumerate(matches[:n]):
        pt1 = match[0]   # (row, col)
        pt2 = match[1]
        x1, y1 = float(pt1[1]), float(pt1[0])
        x2, y2 = float(pt2[1]) + w1, float(pt2[0])
        color = cmap(k / max(n - 1, 1))
        ax.plot([x1, x2], [y1, y2], "-", color=color, linewidth=0.6, alpha=0.7)
        ax.plot(x1, y1, "o", color=color, markersize=3)
        ax.plot(x2, y2, "o", color=color, markersize=3)

    ax.set_title(f"{title}  ({len(matches)} inliers shown: {n})", fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================
#  Keypoint overlay on single frame
# =============================================================

def draw_keypoints(
    img: np.ndarray,
    keypoints: list,
    color: tuple = (0, 255, 0),
    radius: int = 4,
) -> np.ndarray:
    """
    Return a colour copy of `img` with circles at each keypoint.
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        r, c = int(round(kp["pt"][0])), int(round(kp["pt"][1]))
        cv2.circle(vis, (c, r), radius, color, 1, cv2.LINE_AA)
    return vis


# =============================================================
#  Trajectory plot
# =============================================================

def plot_trajectory(
    trajectory: list,
    title: str = "Visual Odometry — Estimated Trajectory",
    figsize: tuple = (10, 8),
    arrow_every: int = 5,
) -> plt.Figure:
    """
    Plot the accumulated 2-D camera trajectory with direction arrows.

    Parameters
    ----------
    trajectory  : list of (tx, ty) translation tuples (cumulative)
    arrow_every : draw a directional arrow every N steps
    """
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    # Gradient colour along the path (blue → yellow)
    n = len(xs)
    cmap = plt.cm.plasma
    for i in range(n - 1):
        c = cmap(i / max(n - 2, 1))
        ax.plot(xs[i:i+2], ys[i:i+2], "-", color=c, linewidth=1.5, alpha=0.85)

    # Direction arrows
    for i in range(0, n - 1, arrow_every):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        length = np.hypot(dx, dy)
        if length < 1e-3:
            continue
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(
                arrowstyle="->",
                color="white",
                lw=1.0,
            ),
        )

    # Start / end markers
    ax.scatter([xs[0]], [ys[0]], s=80, color="lime", zorder=5, label="Start")
    ax.scatter([xs[-1]], [ys[-1]], s=80, color="red",  zorder=5, label="End")

    # Frame index labels at every 10th frame
    for i in range(0, n, 10):
        ax.text(
            xs[i], ys[i],
            str(i + 1),
            color="white",
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.set_xlabel("X  (pixels)", color="white")
    ax.set_ylabel("Y  (pixels)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


# =============================================================
#  Per-frame strip  (used in the main loop for live feedback)
# =============================================================

def plot_frame_matches(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    good_matches: list,
    frame_idx: int,
    tx: float,
    ty: float,
    n_inliers: int,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Compact 2-panel figure: matched pair (left) + translation info (right).
    """
    fig, (ax_match, ax_info) = plt.subplots(1, 2, figsize=figsize,
                                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: side-by-side match strip
    h1, w1 = img_prev.shape
    h2, w2 = img_curr.shape
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2), dtype=np.uint8)
    canvas[:h1, :w1] = img_prev
    canvas[:h2, w1:] = img_curr
    ax_match.imshow(canvas, cmap="gray")

    cmap = plt.cm.spring
    n = min(len(good_matches), 60)
    for k, m in enumerate(good_matches[:n]):
        pt1, pt2 = m[0], m[1]
        x1, y1 = float(pt1[1]), float(pt1[0])
        x2, y2 = float(pt2[1]) + w1, float(pt2[0])
        col = cmap(k / max(n - 1, 1))
        ax_match.plot([x1, x2], [y1, y2], "-", color=col, lw=0.5, alpha=0.6)
    ax_match.axis("off")
    ax_match.set_title(f"Frame {frame_idx - 1} → {frame_idx}", fontsize=9)

    # Right: info panel
    ax_info.set_facecolor("#111")
    info_text = (
        f"Frame pair\n{frame_idx-1} → {frame_idx}\n\n"
        f"Inliers: {n_inliers}\n\n"
        f"ΔX = {tx:+.1f} px\n"
        f"ΔY = {ty:+.1f} px"
    )
    ax_info.text(
        0.5, 0.5, info_text,
        ha="center", va="center",
        transform=ax_info.transAxes,
        fontsize=10, color="white",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#222", ec="#555"),
    )
    ax_info.axis("off")

    plt.tight_layout()
    return fig
