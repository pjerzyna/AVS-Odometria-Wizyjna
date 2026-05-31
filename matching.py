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
    max_hamming: int = 50,    # Maksymalny akceptowalny dystans Hamminga
    margin: int = 5           # Minimalna różnica dystansu między 1. a 2. sąsiadem
) -> list:
    """
    Brute-force Hamming matching z uwzględnieniem bezpiecznego marginesu 
    oraz twardego progu dla deskryptorów binarnych.
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    D = _hamming_matrix(desc1, desc2)   # (N, M)
    N, M = D.shape

    valid_matches = []

    if M >= 2:
        # POPRAWKA B: Używamy kth=1, ponieważ dla M=2 prawidłowe indeksy to 0 i 1.
        part = np.argpartition(D, 1, axis=1)[:, :2]   # (N, 2)
        idx1 = part[:, 0]
        idx2 = part[:, 1]
        
        # Upewniamy się, że idx1 wskazuje na najbliższego sąsiada
        swap = D[np.arange(N), idx1] > D[np.arange(N), idx2]
        idx1[swap], idx2[swap] = idx2[swap].copy(), idx1[swap].copy()
        
        d1 = D[np.arange(N), idx1]
        d2 = D[np.arange(N), idx2]
        
        # POPRAWKA C: Stosujemy absolutny próg oraz bezpieczny margines różnicy 
        # zamiast ułamkowego testu Lowe'a
        ratio_ok = (d1 <= max_hamming) & (d1 <= d2 - margin)
        
        # Zbieranie wyników
        for i in range(N):
            if ratio_ok[i]:
                valid_matches.append((coords1[i], coords2[idx1[i]], d1[i]))
                
    elif M == 1:
        # Przypadek brzegowy - dostępny tylko jeden punkt w drugim obrazie
        for i in range(N):
            dist = D[i, 0]
            if dist <= max_hamming:
                valid_matches.append((coords1[i], coords2[0], dist))

    # Sortowanie po dystansie i zwrócenie najlepszych `n` wyników
    valid_matches.sort(key=lambda x: x[2])
    return valid_matches[:n]
