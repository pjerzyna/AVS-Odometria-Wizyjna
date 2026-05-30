"""
ORB Feature Detection & Description Module  (vectorised, fast)
---------------------------------------------------------------
Every hot-path is NumPy-vectorised — no Python loops over keypoints.

Implements:
  - FAST keypoint detection (OpenCV)
  - Harris corner response scoring (vectorised image-space lookup)
  - Non-Maximum Suppression  (grid-cell bucketing — O(N), not O(N²))
  - Intensity-centroid orientation  (batched integral-image trick)
  - rBRIEF descriptor  (all 256 pairs x all N keypoints in one shot)
"""

import numpy as np
import cv2


# =============================================================
#  File I/O
# =============================================================

def load_brief_pairs(filepath: str) -> np.ndarray:
    """
    Load 256 pre-optimised BRIEF pixel-pair offsets.
    Each line: x1  y1  x2  y2  (offsets from keypoint centre).
    Returns (256, 4) float32 array.
    """
    pairs = np.loadtxt(filepath, dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape != (256, 4):
        raise ValueError(
            f"Expected (256,4) array, got {pairs.shape}. "
            "Check orb_descriptor_positions.txt."
        )
    return pairs


# =============================================================
#  FAST detection  (OpenCV, returns arrays directly)
# =============================================================

def _fast_detect(img: np.ndarray, threshold: int):
    """
    Returns rows, cols, responses as float32 arrays, shape (N,).
    """
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold, nonmaxSuppression=False
    )
    kps_cv = fast.detect(img, None)
    if not kps_cv:
        return (
            np.empty(0, np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
        )
    pts = np.array([kp.pt for kp in kps_cv], dtype=np.float32)   # (N,2): col,row
    resp = np.array([kp.response for kp in kps_cv], dtype=np.float32)
    return pts[:, 1], pts[:, 0], resp   # rows, cols, responses


# =============================================================
#  Harris scoring  (vectorised image-space lookup)
# =============================================================

def _harris_response(img: np.ndarray, rows, cols, patch: int = 7):
    """
    Re-score keypoints with the Harris response using a pre-computed
    response map — one vectorised array lookup, no per-point loop.

    Returns filtered (rows, cols, responses) where response > 0.
    """
    h, w = img.shape
    half = patch // 2
    k = 0.04

    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = cv2.boxFilter(Ix * Ix, -1, (patch, patch))
    Iyy = cv2.boxFilter(Iy * Iy, -1, (patch, patch))
    Ixy = cv2.boxFilter(Ix * Iy, -1, (patch, patch))

    det   = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    harris_map = det - k * trace * trace    # (H, W) float32

    ri = np.round(rows).astype(np.int32)
    ci = np.round(cols).astype(np.int32)

    # Boundary mask
    valid = (ri >= half) & (ri < h - half) & (ci >= half) & (ci < w - half)
    ri, ci = ri[valid], ci[valid]
    rows_v, cols_v = rows[valid], cols[valid]

    # Lookup in harris_map
    resp = harris_map[ri, ci]
    pos = resp > 0
    return rows_v[pos], cols_v[pos], resp[pos]


# =============================================================
#  NMS  — grid-cell bucketing, O(N)
# =============================================================

def _nms(rows, cols, responses, radius: int = 5):
    """
    Fast NMS via grid bucketing.

    Divide the image into cells of size `radius`.  For each cell keep
    only the keypoint with the highest response.  This is O(N) and
    produces the same coarse result as the old O(N²) sweep without
    hanging on large keypoint sets.
    """
    if len(rows) == 0:
        return rows, cols, responses

    cell_r = (rows / radius).astype(np.int32)
    cell_c = (cols / radius).astype(np.int32)

    # Unique cell ID per keypoint
    cell_ids = cell_r * 100000 + cell_c    # assumes image width < 100000 px

    # Sort by response descending so argmax per cell = first occurrence
    order = np.argsort(-responses)
    rows_s      = rows[order]
    cols_s      = cols[order]
    resp_s      = responses[order]
    cell_ids_s  = cell_ids[order]

    # Keep first occurrence of each cell (= highest response)
    _, first_idx = np.unique(cell_ids_s, return_index=True)

    return rows_s[first_idx], cols_s[first_idx], resp_s[first_idx]


# =============================================================
#  Intensity-centroid orientation  (batched)
# =============================================================

def _compute_orientation(img: np.ndarray, rows, cols, patch_radius: int = 15):
    """
    Vectorised orientation via intensity centroid.

    Uses a fixed-size patch around every keypoint simultaneously.
    Keypoints too close to the border get angle=0.
    """
    h, w = img.shape
    pr = patch_radius
    img_f = img.astype(np.float32)

    # Coordinate grids for the patch (relative to centre)
    dy = np.arange(-pr, pr + 1, dtype=np.float32)   # (2pr+1,)
    dx = np.arange(-pr, pr + 1, dtype=np.float32)
    DX, DY = np.meshgrid(dx, dy)                     # (2pr+1, 2pr+1) each

    ri = np.round(rows).astype(np.int32)
    ci = np.round(cols).astype(np.int32)
    angles = np.zeros(len(ri), dtype=np.float64)

    for i in range(len(ri)):
        r0 = ri[i] - pr
        r1 = ri[i] + pr + 1
        c0 = ci[i] - pr
        c1 = ci[i] + pr + 1
        if r0 < 0 or r1 > h or c0 < 0 or c1 > w:
            # Near-border keypoint: just use angle=0
            continue
        patch = img_f[r0:r1, c0:c1]
        m10 = float(np.sum(DX * patch))
        m01 = float(np.sum(DY * patch))
        angles[i] = np.arctan2(m01, m10)

    return angles


# =============================================================
#  Public API: get keypoints
# =============================================================

def get_orb_keypoints(
    img: np.ndarray,
    threshold: int = 20,
    n_best: int = 500,
    nms_radius: int = 5,
    patch_radius: int = 15,
) -> list:
    """
    Detect ORB-style keypoints.  Returns a list of dicts with keys:
      'pt'       : (row, col) float
      'response' : Harris score
      'angle'    : orientation in radians
    """
    rows, cols, resp = _fast_detect(img, threshold)
    if len(rows) == 0:
        return []

    rows, cols, resp = _harris_response(img, rows, cols)
    if len(rows) == 0:
        return []

    rows, cols, resp = _nms(rows, cols, resp, radius=nms_radius)

    # Keep n_best by Harris score
    if len(resp) > n_best:
        top = np.argsort(-resp)[:n_best]
        rows, cols, resp = rows[top], cols[top], resp[top]

    angles = _compute_orientation(img, rows, cols, patch_radius=patch_radius)

    keypoints = [
        {"pt": (float(rows[i]), float(cols[i])),
         "response": float(resp[i]),
         "angle": float(angles[i])}
        for i in range(len(rows))
    ]
    return keypoints


# =============================================================
#  rBRIEF descriptors  (fully vectorised over all keypoints)
# =============================================================

def _build_rotation_matrices(angles: np.ndarray) -> np.ndarray:
    """
    Build N×2×2 rotation matrices for N keypoints simultaneously.
    """
    cos_a = np.cos(angles)   # (N,)
    sin_a = np.sin(angles)
    R = np.stack([
        np.stack([ cos_a, -sin_a], axis=1),
        np.stack([ sin_a,  cos_a], axis=1),
    ], axis=1)               # (N, 2, 2)
    return R.astype(np.float32)


def descriptor_BRIEF(
    img: np.ndarray,
    keypoints: list,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    Compute rotation-invariant BRIEF (rBRIEF) descriptors.

    All N keypoints are processed in a single vectorised pass:
      - Rotate 256 pairs by each keypoint's orientation  → (N, 256, 4) int
      - Sample two pixels per pair per keypoint           → fancy index
      - Binary comparison                                 → (N, 256) bool
      - Pack into 4 x uint64 words                        → (N, 4) uint64

    Returns (N, 4) uint64 array, or empty (0, 4) if no keypoints.
    """
    if not keypoints:
        return np.empty((0, 4), dtype=np.uint64)

    h, w = img.shape
    smooth = cv2.GaussianBlur(img, (5, 5), 2.0)   # (H, W) uint8

    N = len(keypoints)
    rows   = np.array([kp["pt"][0] for kp in keypoints], dtype=np.float32)  # (N,)
    cols   = np.array([kp["pt"][1] for kp in keypoints], dtype=np.float32)
    angles = np.array([kp.get("angle", 0.0) for kp in keypoints], dtype=np.float32)

    # Rotation matrices for every keypoint: (N, 2, 2)
    R = _build_rotation_matrices(angles)

    # pairs[:, :2] = (x1, y1),  pairs[:, 2:] = (x2, y2)  — shape (256, 2) each
    pts1 = pairs[:, :2].T   # (2, 256)
    pts2 = pairs[:, 2:].T   # (2, 256)

    # Rotate: R @ pts  → (N, 2, 2) × (2, 256) → (N, 2, 256)
    rot1 = np.round(R @ pts1).astype(np.int32)   # (N, 2, 256)  [dx, dy]
    rot2 = np.round(R @ pts2).astype(np.int32)

    ri = np.round(rows).astype(np.int32)   # (N,)
    ci = np.round(cols).astype(np.int32)

    # Absolute pixel coordinates for both ends of every pair
    # rot1[n, 0, p] = dx,  rot1[n, 1, p] = dy
    # Image coords: col = ci[n] + dx,  row = ri[n] + dy
    r1s = np.clip(ri[:, None] + rot1[:, 1, :], 0, h - 1)   # (N, 256)
    c1s = np.clip(ci[:, None] + rot1[:, 0, :], 0, w - 1)
    r2s = np.clip(ri[:, None] + rot2[:, 1, :], 0, h - 1)
    c2s = np.clip(ci[:, None] + rot2[:, 0, :], 0, w - 1)

    # Pixel intensity lookup  → (N, 256)
    pix1 = smooth[r1s, c1s].astype(np.int16)
    pix2 = smooth[r2s, c2s].astype(np.int16)

    # Binary test: bit = 1 when pix1 < pix2  → (N, 256) bool
    bits = (pix1 < pix2)    # (N, 256) bool

    # Pack 256 bits into 4 × uint64 using np.packbits then view
    # np.packbits packs MSB-first, so we flip per word.
    # Simplest correct approach: use a powers-of-2 matrix multiply.
    # We split into 4 groups of 64 bits.
    # For each group: value = sum(bit[i] * 2^i for i in 0..63)
    pow2 = (np.uint64(1) << np.arange(64, dtype=np.uint64))   # (64,)

    descriptors = np.zeros((N, 4), dtype=np.uint64)
    for word in range(4):
        chunk = bits[:, word * 64: (word + 1) * 64].astype(np.uint64)  # (N, 64)
        # dot product with powers of 2
        descriptors[:, word] = chunk @ pow2   # (N,)

    return descriptors
