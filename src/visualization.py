"""
Visualization Utilities for Feature Matching and Image Alignment.
"""

import numpy as np
import matplotlib.pyplot as plt


def appendimages(im1, im2):
    """
    Concatenates two images horizontally side-by-side. 
    If the images have different heights, the smaller one is padded with zeros 
    (black pixels) at the bottom to ensure shape alignment before concatenation.
    """
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    # Adjust vertical dimensions if heights do not match
    if rows1 < rows2:
        # Pad the first image with zero-rows to match the height of the second image
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        # Pad the second image with zero-rows to match the height of the first image
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)

    # Combined canvas generation via horizontal concatenation
    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, matches):
    """
    Renders a side-by-side feature matching visualization.
    Draws lines connecting corresponding keypoints between the two input images.

    Expected format for 'matches': [((y1, x1), (y2, x2)), ...]
    """
    # Color cyclic palette for alternating line colors to enhance readability
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Create a unified canvas containing both images
    im3 = appendimages(im1, im2)

    plt.figure()
    plt.imshow(im3, cmap='gray')

    # Get the horizontal offset (width of the first image)
    # This is required to shift the X-coordinates of the second image points to the right
    cols1 = im1.shape[1]

    # Render connection lines between corresponding features
    for i, m in enumerate(matches):
        # Coordinates mapping: 
        # m[0] -> (y1, x1) from the first image
        # m[1] -> (y2, x2) from the second image
        # Note: plt.plot expects coordinates in [x1, x2], [y1, y2] spatial layout
        plt.plot([m[0][1], m[1][1] + cols1], [m[0][0], m[1][0]], colors[i % 6], linewidth=0.5)

    plt.axis('off')