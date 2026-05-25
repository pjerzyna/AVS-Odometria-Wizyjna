# ===== ORB - (Oriented FAST and Rotated BRIEF) ====
import numpy as np
import cv2
from .harris import harris

# FAST (Features from Accelerated Segment Test)
def detector_FAST(I, x, y, threshold=20, n=9):
    # Intensity of central point
    Ic = int(I[y, x])

    # 16 pixels in a circle around the keypoint (coordinates are read from the image)
    circle = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
              (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]
    vals = [int(I[y + dy, x + dx]) for dx, dy in circle]

    # Adding the beginning of the list to its end (e.g., first 8 elements for n=9)
    # Allows the for loop to check sequences that wrap around the circle's starting point
    vals = vals + vals[:n - 1]

    # Continuity check: at least n pixels in a row must be brighter or darker than the central pixel by the threshold
    brighter_count = 0
    darker_count = 0

    for v in vals:
        if v > Ic + threshold:
            brighter_count += 1
            darker_count = 0
        elif v < Ic - threshold:
            darker_count += 1
            brighter_count = 0
        else:
            brighter_count = 0
            darker_count = 0

        if brighter_count >= n or darker_count >= n:
            return True

    return False


def rotate(p, angle):
    dx, dy = p
    nx = np.cos(angle) * dx - np.sin(angle) * dy
    ny = np.sin(angle) * dx + np.cos(angle) * dy
    return int(round(nx)), int(round(ny))


# BRIEF (Binary Robust Independent Elementary Features)
def descriptor_BRIEF(I, keypoints, pairs):
    # To reduce noise, we can apply a Gaussian blur to the image before sampling the pixel intensities for the descriptor
    I_blurred = cv2.GaussianBlur(I, (5, 5), 0)
    descriptors = []

    for kp in keypoints:
        y, x = kp['pt']
        theta = kp['theta']
        bit_vector = []

        for p1, p2 in pairs:
            # Rotation of test point pairs by the keypoint's orientation angle theta, p = (dx, dy)

            p1_rot = rotate(p1, theta)
            p2_rot = rotate(p2, theta)

            # Sampling the pixel intensities at the rotated positions and comparing them to create the binary descriptor
            if I_blurred[y + p1_rot[1], x + p1_rot[0]] < I_blurred[y + p2_rot[1], x + p2_rot[0]]:
                bit_vector.append(1)
            else:
                bit_vector.append(0)

        # The resulting bit vector is the BRIEF descriptor for the keypoint
        descriptors.append(np.array(bit_vector, dtype=np.uint8))

    return descriptors


def get_orientation(I, x, y, r=15):
    m01 = 0
    m10 = 0
    # Iterating in a circle of radius r around the point (x, y) and calculating the moments m10 and m01
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                val = int(I[y + dy, x + dx])
                m10 += dx * val  # moment p=1, q=0
                m01 += dy * val  # moment p=0, q=1

    # theta is the angle of the vector (m10, m01) with respect to the x-axis
    return np.arctan2(m01, m10)


def get_orb_keypoints(img_gray, threshold=20, n_best=500):
    Y, X = img_gray.shape
    # Harris map for the entire image
    harris_map = harris(img_gray)

    fast_points = []
    # Margin 31 px --> to ensure that BRIEF descriptor can be computed
    for y in range(31, Y - 31):
        for x in range(31, X - 31):
            if detector_FAST(img_gray, x, y, threshold):
                fast_points.append((y, x))

    # Non-Maximum Suppression (NMS) 3x3 
    nms_points = []
    for y, x in fast_points:
        patch = harris_map[y - 1:y + 2, x - 1:x + 2]
        if harris_map[y, x] == np.max(patch):
            nms_points.append((y, x, harris_map[y, x]))

    # Picking the n_best points with the highest Harris response after NMS
    nms_points.sort(key=lambda x: x[2], reverse=True)
    final_pts = nms_points[:n_best]

    # Calculation of an orientation
    keypoints = []
    for y, x, h_val in final_pts:
        theta = get_orientation(img_gray, x, y)
        keypoints.append({'pt': (y, x), 'theta': theta})

    return keypoints

def load_brief_pairs(filename):
    # Loading pairs of dx1, dy1, dx2, dy2 from the file
    data = np.loadtxt(filename, dtype=float)
    # int conversion for pixel coordinates, and taking only the first 256 pairs
    data = data[:256].astype(int)

    pairs = []
    for row in data:
        # row to [x1, y1, x2, y2]
        p1 = (row[0], row[1])
        p2 = (row[2], row[3])
        pairs.append((p1, p2))
    return pairs
