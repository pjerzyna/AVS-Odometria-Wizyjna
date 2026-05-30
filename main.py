"""
Visual Odometry Pipeline  —  MPC (Motion Positioning Component)
================================================================
Processes a numbered sequence of aerial images (frame_001.jpg …
frame_062.jpg) and estimates the camera trajectory by:

  1. Loading consecutive frame pairs
  2. Detecting ORB keypoints (FAST + Harris + NMS + orientation)
  3. Computing rotation-invariant BRIEF descriptors
     (using the pre-optimised 256-pair file)
  4. Brute-force Hamming matching with Lowe's ratio test
  5. Homography estimation + RANSAC outlier rejection
  6. Extracting the (tx, ty) translation from the homography
  7. Accumulating the camera position over all frames
  8. Saving visualisations:
       - Per-pair match strip (output/matches/)
       - Final trajectory plot (output/trajectory.png)

Usage:
------
  python main.py --images_dir images/sequence \
                 --pairs_file orb_descriptor_positions.txt \
                 --output_dir output \
                 --threshold 20 --n_best 500
                 --save_strips   # optional: save per-pair images

"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend for saving figures
import matplotlib.pyplot as plt

from orb          import get_orb_keypoints, descriptor_BRIEF, load_brief_pairs
from matching     import matching_descriptors_hamming
from visualization import plot_matches, plot_trajectory, plot_frame_matches


# =============================================================
#  Argument parsing
# =============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Visual Odometry — MPC Module")
    p.add_argument("--images_dir",  default="images/sequence",
                   help="Folder containing frame_001.jpg … frame_062.jpg")
    p.add_argument("--pairs_file",  default="orb_descriptor_positions.txt",
                   help="Path to the 256 BRIEF pair offsets file")
    p.add_argument("--output_dir",  default="output",
                   help="Directory for saved figures")
    p.add_argument("--threshold",   type=int,   default=20,
                   help="FAST detection threshold (lower = more points)")
    p.add_argument("--n_best",      type=int,   default=500,
                   help="Max keypoints per frame after NMS + Harris scoring")
    p.add_argument("--n_matches",   type=int,   default=150,
                   help="Number of candidates before RANSAC")
    p.add_argument("--ransac_thr",  type=float, default=5.0,
                   help="RANSAC reprojection error threshold (pixels)")
    p.add_argument("--save_strips", action="store_true",
                   help="Save per-pair match strip images")
    return p.parse_args()


# =============================================================
#  Image loading helper
# =============================================================

def load_image_sequence(images_dir: str) -> list:
    """
    Discover and sort all .jpg / .png images in `images_dir`.
    Supports names like frame_001.jpg, 001.jpg, 1.jpg etc.
    """
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(images_dir, pat)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return paths


# =============================================================
#  Single-pair odometry step
# =============================================================

def process_pair(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    pairs: np.ndarray,
    threshold: int,
    n_best: int,
    n_matches: int,
    ransac_thr: float,
) -> tuple:
    """
    Estimate the 2-D translation between two consecutive frames.

    Returns
    -------
    (tx, ty, n_inliers, good_matches, H)
      tx, ty      : pixel translation estimate
      n_inliers   : number of RANSAC inliers
      good_matches: filtered inlier matches
      H           : 3×3 homography matrix (or None if estimation failed)
    """
    # --- Feature detection & description ---
    kp_prev = get_orb_keypoints(img_prev, threshold=threshold, n_best=n_best)
    kp_curr = get_orb_keypoints(img_curr, threshold=threshold, n_best=n_best)

    if len(kp_prev) < 8 or len(kp_curr) < 8:
        return 0.0, 0.0, 0, [], None

    desc_prev = descriptor_BRIEF(img_prev, kp_prev, pairs)
    desc_curr = descriptor_BRIEF(img_curr, kp_curr, pairs)

    if desc_prev.shape[0] == 0 or desc_curr.shape[0] == 0:
        return 0.0, 0.0, 0, [], None

    # --- Matching ---
    coords_prev = [kp["pt"] for kp in kp_prev]
    coords_curr = [kp["pt"] for kp in kp_curr]

    raw_matches = matching_descriptors_hamming(
        desc_prev, coords_prev,
        desc_curr, coords_curr,
        n=n_matches,
    )

    if len(raw_matches) < 4:
        return 0.0, 0.0, 0, [], None

    # --- RANSAC Homography estimation ---
    # Convert (row, col) → (x, y) for OpenCV
    src_pts = np.float32([[m[0][1], m[0][0]] for m in raw_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([[m[1][1], m[1][0]] for m in raw_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thr)

    if H is None or mask is None:
        return 0.0, 0.0, 0, [], None

    mask_flat = mask.ravel().tolist()
    good_matches = [m for i, m in enumerate(raw_matches) if mask_flat[i] == 1]
    n_inliers = len(good_matches)

    # --- Extract translation from homography ---
    # For mostly-translating aerial cameras the (tx, ty) lives in H[0,2], H[1,2]
    tx = float(H[0, 2])
    ty = float(H[1, 2])

    return tx, ty, n_inliers, good_matches, H


# =============================================================
#  Main pipeline
# =============================================================

def main():
    args = parse_args()

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    matches_dir = os.path.join(args.output_dir, "matches")
    if args.save_strips:
        os.makedirs(matches_dir, exist_ok=True)

    # Load BRIEF pairs
    print(f"Loading BRIEF pairs from: {args.pairs_file}")
    pairs = load_brief_pairs(args.pairs_file)
    print(f"  → {pairs.shape[0]} pairs loaded.")

    # Discover image sequence
    print(f"\nScanning images in: {args.images_dir}")
    image_paths = load_image_sequence(args.images_dir)
    n_frames = len(image_paths)
    print(f"  → {n_frames} frames found.")

    if n_frames < 2:
        print("Need at least 2 frames. Exiting.")
        sys.exit(1)

    # Trajectory accumulator
    trajectory = [(0.0, 0.0)]   # start at origin
    cx, cy = 0.0, 0.0

    # Per-frame statistics
    stats = []

    # Load first frame
    prev_bgr = cv2.imread(image_paths[0])
    if prev_bgr is None:
        raise IOError(f"Cannot read: {image_paths[0]}")
    img_prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

    print("\n─── Processing frame pairs ─────────────────────────────────")
    for frame_idx in range(1, n_frames):
        curr_bgr = cv2.imread(image_paths[frame_idx])
        if curr_bgr is None:
            print(f"  [SKIP] Cannot read: {image_paths[frame_idx]}")
            trajectory.append((cx, cy))
            stats.append({"frame": frame_idx, "tx": 0, "ty": 0, "inliers": 0})
            img_prev = img_prev   # keep previous
            continue

        img_curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        tx, ty, n_inliers, good_matches, H = process_pair(
            img_prev, img_curr,
            pairs,
            threshold=args.threshold,
            n_best=args.n_best,
            n_matches=args.n_matches,
            ransac_thr=args.ransac_thr,
        )

        # Accumulate position
        cx += tx
        cy += ty
        trajectory.append((cx, cy))

        stats.append({
            "frame": frame_idx,
            "tx": tx,
            "ty": ty,
            "inliers": n_inliers,
        })

        print(
            f"  Frame {frame_idx:02d}/{n_frames-1:02d} | "
            f"Inliers: {n_inliers:3d} | "
            f"ΔX={tx:+7.2f}  ΔY={ty:+7.2f} | "
            f"Pos ({cx:+8.2f}, {cy:+8.2f})"
        )

        # Save match strip
        if args.save_strips and good_matches:
            fig = plot_frame_matches(
                img_prev, img_curr, good_matches,
                frame_idx=frame_idx, tx=tx, ty=ty, n_inliers=n_inliers,
            )
            strip_path = os.path.join(matches_dir, f"pair_{frame_idx:03d}.png")
            fig.savefig(strip_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        img_prev = img_curr   # advance frame

    # =============================================================
    #  Trajectory plot
    # =============================================================
    print("\n─── Saving trajectory ──────────────────────────────────────")
    traj_fig = plot_trajectory(
        trajectory,
        title="Visual Odometry — Estimated Camera Trajectory (62 frames)",
        arrow_every=4,
    )
    traj_path = os.path.join(args.output_dir, "trajectory.png")
    traj_fig.savefig(traj_path, dpi=130, bbox_inches="tight")
    plt.close(traj_fig)
    print(f"  Trajectory saved → {traj_path}")

    # =============================================================
    #  Summary stats
    # =============================================================
    print("\n─── Summary ─────────────────────────────────────────────────")
    inlier_counts = [s["inliers"] for s in stats]
    print(f"  Total frames processed : {n_frames}")
    print(f"  Mean inliers / pair    : {np.mean(inlier_counts):.1f}")
    print(f"  Min  inliers           : {min(inlier_counts)}")
    print(f"  Max  inliers           : {max(inlier_counts)}")
    total_dist = sum(
        np.hypot(trajectory[i+1][0] - trajectory[i][0],
                 trajectory[i+1][1] - trajectory[i][1])
        for i in range(len(trajectory) - 1)
    )
    print(f"  Cumulative path length : {total_dist:.1f} px")
    print(f"  Final position         : ({cx:+.1f}, {cy:+.1f}) px")
    print("\nDone.")


if __name__ == "__main__":
    main()
