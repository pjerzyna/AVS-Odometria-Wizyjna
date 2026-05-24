# Implementacja Odometrii Wizyjnej - naraize szkic (Harris + ORB) + RANSAC

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pm import plot_matches


# Data loading and transformation to grayscale
IG_EIFFEL1 = cv2.imread('images/eiffel1.jpg')
IG_GRAY_EIFFEL1 = cv2.cvtColor(IG_EIFFEL1, cv2.COLOR_BGR2GRAY)
IG_EIFFEL2 = cv2.imread('images/eiffel2.jpg')
IG_GRAY_EIFFEL2 = cv2.cvtColor(IG_EIFFEL2, cv2.COLOR_BGR2GRAY)



# ==== Harris corner detection (horizontal and vertical gradients) ====
def harris(I: cv2.typing.MatLike, mask_Sobel=7, mask_Gaussian=7, k=0.05):
    
    IG_SOBEL_X = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=mask_Sobel)
    IG_SOBEL_Y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=mask_Sobel)
    
    IG_SOBEL_X2 = IG_SOBEL_X * IG_SOBEL_X
    IG_SOBEL_Y2 = IG_SOBEL_Y * IG_SOBEL_Y
    IG_SOBEL_XY = IG_SOBEL_X * IG_SOBEL_Y

    IG_SOBEL_X2_GAUSS = cv2.GaussianBlur(IG_SOBEL_X2, (mask_Gaussian, mask_Gaussian), 0)
    IG_SOBEL_Y2_GAUSS = cv2.GaussianBlur(IG_SOBEL_Y2, (mask_Gaussian, mask_Gaussian), 0)
    IG_SOBEL_XY_GAUSS = cv2.GaussianBlur(IG_SOBEL_XY, (mask_Gaussian, mask_Gaussian), 0)

    # autocorrelation matrix
    M = np.array([[IG_SOBEL_X2_GAUSS, IG_SOBEL_XY_GAUSS], [IG_SOBEL_XY_GAUSS, IG_SOBEL_Y2_GAUSS]])
    
    # determinant and trace
    M_det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    M_tr = M[0][0] + M[1][1]

    H = M_det - k*M_tr*M_tr
    H_NORMALIZED = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX)

    return H_NORMALIZED



# ===== ORB - (Oriented FAST and Rotated BRIEF) ====

# FAST (Features from Accelerated Segment Test)
def detector_FAST(I, x, y, threshold=20, n=9):
    # Intensity of central point
    Ic = int(I[y,x])

    # 16 pixels in a circle around the keypoint (coordinates are read from the image)
    circle = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
              (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]   
    vals = [int(I[y + dy, x + dx]) for dx, dy in circle]

    # Adding the beginning of the list to its end (e.g., first 8 elements for n=9) 
    # Allows the for loop to check sequences that wrap around the circle's starting point
    vals = vals + vals[:n-1] 

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
            if dx*dx + dy*dy <= r*r:
                val = int(I[y + dy, x + dx])
                m10 += dx * val # moment p=1, q=0
                m01 += dy * val # moment p=0, q=1 
    
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
    
    # Non-Maximum Suppression (NMS) 3x3  [Curcial moment]
    nms_points = []
    for y, x in fast_points:
        patch = harris_map[y-1:y+2, x-1:x+2]
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


def matching_descriptors_hamming(desc1, coords1, desc2, coords2, n=20):
    all_matches = []

    for i in range(len(desc1)):
        best_dist = float('inf')
        best_idx_in_2 = -1
        
        for j in range(len(desc2)):
            # Hamming distance for binary descriptors is the number of positions at which the corresponding bits are different 
            dist = np.count_nonzero(desc1[i] != desc2[j])
            
            if dist < best_dist:
                best_dist = dist
                best_idx_in_2 = j
        
        if best_idx_in_2 != -1:
            # coordinates of the pair + distance between descriptors
            all_matches.append((coords1[i], coords2[best_idx_in_2], best_dist))

    # Sorting by distance (the best matches have the smallest distance)
    all_matches.sort(key=lambda x: x[2])
    return all_matches[:n]


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

if __name__ == "__main__":
    img1 = IG_GRAY_EIFFEL1
    img2 = IG_GRAY_EIFFEL2

    # Load 256 pairs of test points from the file (dx1, dy1, dx2, dy2)
    pairs = load_brief_pairs('orb_descriptor_positions.txt')

    # Points detection (FAST + Harris + NMS + Orientation) 
    kp1 = get_orb_keypoints(img1, threshold=20, n_best=500)
    kp2 = get_orb_keypoints(img2, threshold=20, n_best=500)

    # BRIEF descriptor generation with rotation invariance 
    desc1 = descriptor_BRIEF(img1, kp1, pairs)
    desc2 = descriptor_BRIEF(img2, kp2, pairs)

    # Matching descriptors with Hamming distance (for binary descriptors) 
    coords1 = [k['pt'] for k in kp1]
    coords2 = [k['pt'] for k in kp2]
    
    # ZWIĘKSZONO wartość 'n', aby RANSAC miał wystarczająco dużo punktów do przetestowania modeli
    matches_with_dist = matching_descriptors_hamming(desc1, coords1, desc2, coords2, n=150)

    # Formatting for visualization, we take only coordinates, not distance
    # m[0] to (y1, x1), m[1] to (y2, x2)
    final_matches = [(m[0], m[1]) for m in matches_with_dist]

    # =====================================================================
    # KROK 4: Estymacja transformacji i filtrowanie (RANSAC) wg rozprawy
    # =====================================================================
    
    # Przekształcenie punktów z formatu (row, col) do formatu (x, y) dla OpenCV
    src_pts = np.float32([[m[0][1], m[0][0]] for m in final_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([[m[1][1], m[1][0]] for m in final_matches]).reshape(-1, 1, 2)

    # Estymacja macierzy homografii i filtracja za pomocą algorytmu RANSAC
    # Parametr 5.0 to dopuszczalny błąd (w pikselach) dla punktu, aby został uznany za inlier
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Zachowaj tylko poprawne dopasowania (odrzucenie tzw. outliers)
    if mask is not None:
        matchesMask = mask.ravel().tolist()
        good_matches = [m for i, m in enumerate(final_matches) if matchesMask[i] == 1]
    else:
        good_matches = []

    print(f"Liczba dopasowań przed RANSAC: {len(final_matches)}")
    print(f"Liczba poprawnych dopasowań po RANSAC: {len(good_matches)}")

    # Wizualizacja poprawnych (przefiltrowanych) wyników
    plot_matches(img1, img2, good_matches)
    plt.title("Dopasowania po filtracji RANSAC")
    plt.show()