"""
Descriptor Matching Module  (vectorised)
-----------------------------------------
Brute-force Hamming matching for 256-bit descriptors stored as (N, 4) uint64.

Key speed-up: the full N x M distance matrix is computed in one vectorised
pass using a 16-bit popcount LUT applied to each 16-bit slice of the XOR.
"""

import numpy as np


# =============================================================
#  16-bit popcount lookup table  (built once at import time)
# =============================================================

def _build_lut16() -> np.ndarray:
    lut = np.zeros(65536, dtype=np.uint8)
    for i in range(1, 65536):
        lut[i] = lut[i >> 1] + (i & 1)
    return lut

_LUT16 = _build_lut16()


def _hamming_matrix(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """
    Vectorised N x M Hamming distance matrix.

    desc1 : (N, 4) uint64
    desc2 : (M, 4) uint64
    returns (N, M) int32
    """
    N, M = desc1.shape[0], desc2.shape[0]
    dists = np.zeros((N, M), dtype=np.int32)

    for word in range(4):
        # (N,1) XOR (1,M) → (N,M) uint64
        xor = np.bitwise_xor(
            desc1[:, word].reshape(N, 1),
            desc2[:, word].reshape(1, M),
        )
        # Count bits in four 16-bit chunks of each 64-bit word
        for shift in (np.uint64(0), np.uint64(16), np.uint64(32), np.uint64(48)):
            chunk = ((xor >> shift) & np.uint64(0xFFFF)).astype(np.uint16)
            dists += _LUT16[chunk].astype(np.int32)

    return dists


# =============================================================
#  Public matching API
# =============================================================

def matching_descriptors_hamming(
    desc1: np.ndarray,
    coords1: list,
    desc2: np.ndarray,
    coords2: list,
    n: int = 150,
    ratio_threshold: float = 0.75,
) -> list:
    """
    Brute-force Hamming matching with Lowe's ratio test.

    Returns list of ((row1,col1), (row2,col2), distance), len ≤ n.
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    D = _hamming_matrix(desc1, desc2)   # (N, M)
    N, M = D.shape

    # Two nearest neighbours for every descriptor in set 1
    if M >= 2:
        # Use partition to find top-2 cheaply
        part = np.argpartition(D, 2, axis=1)[:, :2]   # (N, 2)
        idx1 = part[:, 0]
        idx2 = part[:, 1]
        # Make sure idx1 is truly the closer one
        swap = D[np.arange(N), idx1] > D[np.arange(N), idx2]
        idx1[swap], idx2[swap] = idx2[swap].copy(), idx1[swap].copy()
        d1 = D[np.arange(N), idx1]
        d2 = D[np.arange(N), idx2]
        ratio_ok = d1 < ratio_threshold * d2
    else:
        idx1 = np.zeros(N, dtype=np.int32)
        d1   = D[:, 0]
        ratio_ok = np.ones(N, dtype=bool)

    # Apply ratio test
    valid = np.where(ratio_ok)[0]
    if len(valid) == 0:
        return []

    # Build match list and sort by distance
    matches = [
        (tuple(coords1[i]), tuple(coords2[idx1[i]]), int(d1[i]))
        for i in valid
    ]
    matches.sort(key=lambda m: m[2])
    return matches[:n]
