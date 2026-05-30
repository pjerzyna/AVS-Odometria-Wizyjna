"""
Two-Image Feature-Match Demo
------------------------------
Load two images → detect keypoints → match → RANSAC → display side-by-side.

Usage
-----
  python demo_two_images.py --img1 images/eiffel1.jpg \
                             --img2 images/eiffel2.jpg \
                             --pairs orb_descriptor_positions.txt
"""

import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orb          import get_orb_keypoints, descriptor_BRIEF, load_brief_pairs
from matching     import matching_descriptors_hamming
from visualization import plot_matches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img1",   default="images/eiffel1.jpg")
    p.add_argument("--img2",   default="images/eiffel2.jpg")
    p.add_argument("--pairs",  default="orb_descriptor_positions.txt")
    p.add_argument("--out",    default="output/two_image_demo.png")
    p.add_argument("--threshold", type=int, default=20)
    p.add_argument("--n_best",    type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()

    # Load
    img1_color = cv2.imread(args.img1)
    img2_color = cv2.imread(args.img2)
    if img1_color is None:
        raise IOError(f"Cannot read {args.img1}")
    if img2_color is None:
        raise IOError(f"Cannot read {args.img2}")

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    pairs = load_brief_pairs(args.pairs)

    # Detect
    kp1 = get_orb_keypoints(img1, threshold=args.threshold, n_best=args.n_best)
    kp2 = get_orb_keypoints(img2, threshold=args.threshold, n_best=args.n_best)
    print(f"Keypoints  img1={len(kp1)}  img2={len(kp2)}")

    # Describe
    desc1 = descriptor_BRIEF(img1, kp1, pairs)
    desc2 = descriptor_BRIEF(img2, kp2, pairs)

    # Match
    coords1 = [k["pt"] for k in kp1]
    coords2 = [k["pt"] for k in kp2]
    raw = matching_descriptors_hamming(desc1, coords1, desc2, coords2, n=150)
    print(f"Raw matches (ratio test): {len(raw)}")

    # RANSAC
    src_pts = np.float32([[m[0][1], m[0][0]] for m in raw]).reshape(-1, 1, 2)
    dst_pts = np.float32([[m[1][1], m[1][0]] for m in raw]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is not None:
        good = [m for i, m in enumerate(raw) if mask.ravel()[i] == 1]
    else:
        good = []

    print(f"Inliers after RANSAC: {len(good)}")

    # Visualise
    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig = plot_matches(img1, img2, good, title="Two-Image Demo — RANSAC Inliers")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
