"""
Visual Odometry Pipeline: Feature Detection, Description, Matching, and RANSAC Filtering.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.orb import get_orb_keypoints, descriptor_BRIEF, load_brief_pairs
from src.matching import matching_descriptors_hamming
from src.visualization import plot_matches


def main():
    # --- 1. Data Loading & Preprocessing ---
    # Load input images and convert them to grayscale for feature extraction
    img1_color = cv2.imread('images/eiffel1.jpg')
    img2_color = cv2.imread('images/eiffel2.jpg')

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Load 256 pixel pair offsets used for binary intensity tests in BRIEF
    pairs = load_brief_pairs('orb_descriptor_positions.txt')

    # --- 2. Feature Detection & Description ---
    # Detect keypoints using FAST, score them via Harris, apply NMS, and compute orientation
    kp1 = get_orb_keypoints(img1, threshold=20, n_best=500)
    kp2 = get_orb_keypoints(img2, threshold=20, n_best=500)

    # Generate rotation-invariant BRIEF descriptors based on keypoint orientation
    desc1 = descriptor_BRIEF(img1, kp1, pairs)
    desc2 = descriptor_BRIEF(img2, kp2, pairs)

    # --- 3. Feature Matching ---
    # Extract raw coordinates (y, x) from keypoint dictionaries
    coords1 = [k['pt'] for k in kp1]
    coords2 = [k['pt'] for k in kp2]

    # Perform Brute-Force matching using Hamming distance for binary patterns
    # Retain a larger pool of matches (n=150) to provide a sufficient consensus set for RANSAC
    matches_with_dist = matching_descriptors_hamming(desc1, coords1, desc2, coords2, n=150)

    # Extract coordinate pairs, discarding the distance metric for visualization/estimation
    # Mapping structure: final_matches = [((y1, x1), (y2, x2)), ...]
    final_matches = [(m[0], m[1]) for m in matches_with_dist]

    # --- 4. Transformation Estimation & Outlier Rejection (RANSAC) ---
    # Convert points from NumPy layout (row, col) to OpenCV spatial layout (x, y)
    src_pts = np.float32([[m[0][1], m[0][0]] for m in final_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([[m[1][1], m[1][0]] for m in final_matches]).reshape(-1, 1, 2)

    # Estimate the Homography matrix and identify geometric outliers using RANSAC
    # A maximum reprojection error of 5.0 pixels is allowed for a point to be considered an inlier
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Filter out mismatching outliers, keeping only verified inliers
    if mask is not None:
        matches_mask = mask.ravel().tolist()
        good_matches = [m for i, m in enumerate(final_matches) if matches_mask[i] == 1]
    else:
        good_matches = []

    # --- 5. Evaluation & Visualization ---
    print(f"Total matches before RANSAC: {len(final_matches)}")
    print(f"Valid inliers after RANSAC:  {len(good_matches)}")

    # Display the filtered geometric matches side by side
    plot_matches(img1, img2, good_matches)
    plt.title("Feature Matches After RANSAC Filtering")
    plt.show()


if __name__ == "__main__":
    main()