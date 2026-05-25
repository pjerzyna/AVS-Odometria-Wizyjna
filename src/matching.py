# ===== Descriptor Matching =====
import numpy as np

def matching_descriptors_hamming(desc1, coords1, desc2, coords2, n=20):
    all_matches = []

    # Iterate through all descriptors from the first image
    for i in range(len(desc1)):
        best_dist = float('inf')
        best_idx_in_2 = -1

        # Find the best match in the second image
        for j in range(len(desc2)):
            # Hamming distance for binary descriptors is the number of positions at which the corresponding bits are different
            dist = np.count_nonzero(desc1[i] != desc2[j])

            if dist < best_dist:
                best_dist = dist
                best_idx_in_2 = j

        if best_idx_in_2 != -1:
            # Store coordinates of the pair and the distance between descriptors
            all_matches.append((coords1[i], coords2[best_idx_in_2], best_dist))

    # Sorting by distance (the best matches have the smallest distance)
    all_matches.sort(key=lambda x: x[2])
    return all_matches[:n]