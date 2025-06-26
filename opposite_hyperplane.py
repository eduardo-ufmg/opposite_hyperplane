import numpy as np
from numpy.linalg import svd

def opposite_hyperplane(Q: np.ndarray, y: np.ndarray, factor_h: float, factor_k: float) -> float:
    """
    Computes the parallelism between the class centroid hyperplane and the
    hyperplane defined by sum(axis) = 0.

    Parallelism is measured as the absolute cosine similarity between the
    normal vectors of the two hyperplanes. A value of 1.0 indicates perfect
    parallelism, while 0.0 indicates orthogonality.

    This implementation is optimized for performance and memory, using
    vectorized operations to avoid Python loops and intermediate copies of
    the input matrix Q.

    Parameters:
        Q: A 2D numpy array of shape (n_samples, n_classes) representing the
           similarity matrix, where Q[i, c] is the similarity of sample i
           to class c.
        y: A 1D numpy array of shape (n_samples,) containing the integer
           class label for each sample. It's assumed that labels are
           in the range [0, n_classes-1].
        factor_h: A scaled factor from the RBF kernel bandwidth parameter.
        factor_k: A scaled factor from the number of nearest neighbors used
                  in the sparse RBF kernel.

    Returns:
        A float value between 1.0 and 0.0 representing the opposite of the absolute
        cosine similarity between the two hyperplane normals. Returns np.nan
        if a unique hyperplane cannot be formed (e.g., fewer than 2 classes
        have samples).

    Raises:
        ValueError: If any class in the range [0, n_classes-1] has no
                    associated samples in y, as this prevents the formation
                    of a unique n_classes-dimensional hyperplane.
    """
    # --- Input Validation and Dimension Extraction ---
    if Q.ndim != 2 or y.ndim != 1:
        raise ValueError("Q must be a 2D array and y must be a 1D array.")
    if Q.shape[0] != y.shape[0]:
        raise ValueError("Q and y must have the same number of samples (first dimension).")

    _n_samples, n_classes = Q.shape

    # A hyperplane requires at least 2 points to be defined in this context.
    if n_classes < 2:
        return np.nan

    # --- Centroid Calculation (Optimized) ---
    # Use np.bincount to get the number of samples per class efficiently.
    class_counts = np.bincount(y, minlength=n_classes)

    # For a unique (n_classes-1)-dimensional hyperplane to be defined by the
    # centroids, every class must be represented by at least one sample.
    if np.any(class_counts == 0):
        raise ValueError(
            f"All classes from 0 to {n_classes-1} must have at least one sample. "
            f"Classes with 0 samples: {np.where(class_counts == 0)[0]}"
        )

    # Use np.add.at for a highly efficient, in-place grouped sum.
    # This avoids creating large intermediate arrays, crucial for memory efficiency.
    class_sums = np.zeros((n_classes, n_classes), dtype=Q.dtype)
    np.add.at(class_sums, y, Q)
    
    # Calculate centroids via vectorized division.
    centroids = class_sums / class_counts[:, np.newaxis]

    # --- Centroid Hyperplane Normal Vector Calculation ---
    # The normal vector is found by identifying the null space of the matrix
    # formed by vectors connecting the centroids.
    # First, create n_classes-1 vectors lying on the hyperplane.
    # We select the first centroid as the reference point.
    V = centroids[1:] - centroids[0]

    # The normal vector is orthogonal to all rows of V. We can find this vector
    # using Singular Value Decomposition (SVD). The last right singular vector
    # (from Vh) corresponds to the smallest singular value and spans the
    # 1D null space of V.
    try:
        # Vh is the matrix of right singular vectors, with shape (n_classes, n_classes).
        # SVD returns it as unit vectors.
        _u, _s, vh = svd(V)
        n_centroid = vh[-1]
    except np.linalg.LinAlgError:
        # This case is unlikely if the input assumptions hold (affinely
        # independent centroids), but is included for robustness.
        return np.nan

    # --- Opposite Hyperplane Normal Vector Calculation ---
    # The "opposite hyperplane" (e.g., x+y+z=0) has a simple normal vector (1, 1, 1).
    # We create and normalize this vector.
    n_opposite = np.ones(n_classes, dtype=Q.dtype)
    norm_opposite = np.sqrt(n_classes) # Faster than np.linalg.norm()
    n_opposite_norm = n_opposite / norm_opposite

    # --- Final Parallelism Calculation ---
    # With both normal vectors being unit vectors, the cosine of the angle
    # between them is simply their dot product. We take the absolute value
    # as the direction of the normal does not affect the hyperplane's orientation.
    cosine_similarity = np.abs(np.dot(n_centroid, n_opposite_norm))
    
    return (1 - float(cosine_similarity)) * factor_h * factor_k

