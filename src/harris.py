# ===== Harris Corner Detection =====
import numpy as np
import cv2


def harris(I: cv2.typing.MatLike, mask_Sobel=7, mask_Gaussian=7, k=0.05):
    # Compute horizontal and vertical image gradients
    IG_SOBEL_X = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=mask_Sobel)
    IG_SOBEL_Y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=mask_Sobel)

    # Calculate products of gradients
    IG_SOBEL_X2 = IG_SOBEL_X * IG_SOBEL_X
    IG_SOBEL_Y2 = IG_SOBEL_Y * IG_SOBEL_Y
    IG_SOBEL_XY = IG_SOBEL_X * IG_SOBEL_Y

    # Blur the gradient products to smooth the results
    IG_SOBEL_X2_GAUSS = cv2.GaussianBlur(IG_SOBEL_X2, (mask_Gaussian, mask_Gaussian), 0)
    IG_SOBEL_Y2_GAUSS = cv2.GaussianBlur(IG_SOBEL_Y2, (mask_Gaussian, mask_Gaussian), 0)
    IG_SOBEL_XY_GAUSS = cv2.GaussianBlur(IG_SOBEL_XY, (mask_Gaussian, mask_Gaussian), 0)

    # Autocorrelation matrix
    M = np.array([[IG_SOBEL_X2_GAUSS, IG_SOBEL_XY_GAUSS], [IG_SOBEL_XY_GAUSS, IG_SOBEL_Y2_GAUSS]])

    # Determinant and trace
    M_det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    M_tr = M[0][0] + M[1][1]

    # Compute Harris response score
    H = M_det - k * M_tr * M_tr

    # Normalize the response map to 0-255 range
    H_NORMALIZED = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX)

    return H_NORMALIZED