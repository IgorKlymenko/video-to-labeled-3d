from q1_affine_factorization_method import get_R_S, plot_reconstruction
import os
import sys
import numpy as np

def a(matrix_path, plot_first_frame=True):
    R, S = get_R_S(matrix_path, plot_first_frame=plot_first_frame)
    r1 = R[::2, :]
    r2 = R[1::2, :]

    # first metric constraint
    A1 = np.vstack((
        r1[:, 0]**2,
        2 * r1[:, 0] * r1[:, 1],
        2 * r1[:, 0] * r1[:, 2],
        r1[:, 1]**2,
        2 * r1[:, 1] * r1[:, 2],
        r1[:, 2]**2
    )).T

    # second metric constraint
    A2 = np.vstack((
        r2[:, 0]**2,
        2 * r2[:, 0] * r2[:, 1],
        2 * r2[:, 0] * r2[:, 2],
        r2[:, 1]**2,
        2 * r2[:, 1] * r2[:, 2],
        r2[:, 2]**2
    )).T

    # third metric constraint
    A3 = np.vstack((
        r1[:, 0] * r2[:, 0],
        r1[:, 0] * r2[:, 1] + r1[:, 1] * r2[:, 0],
        r1[:, 0] * r2[:, 2] + r1[:, 2] * r2[:, 0],
        r1[:, 1] * r2[:, 1],
        r1[:, 1] * r2[:, 2] + r1[:, 2] * r2[:, 1],
        r1[:, 2] * r2[:, 2]
    )).T

    A = np.vstack((A1, A2, A3))
    return A, R, S

def b(A):
    F = A.shape[0] // 3
    b = np.vstack((
        np.ones((F, 1)),
        np.ones((F, 1)),
        np.zeros((F, 1))
    ))
    l, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    L = np.array([
        [l[0, 0], l[1, 0], l[2, 0]],
        [l[1, 0], l[3, 0], l[4, 0]],
        [l[2, 0], l[4, 0], l[5, 0]]
    ])
    return L

def c(L, R, S):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < 0:
            eigenvalues[i] = 1e-10

    L = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    Q = np.linalg.cholesky(L)

    R_eucl = R @ Q
    S_eucl = np.linalg.inv(Q) @ S

    return R_eucl, S_eucl

def d(S_eucl, ref_points=None, ref_distance=None):
    """Fix the global scale of the metric reconstruction.

    Factorization recovers structure only up to scale. If a pair of point
    indices and their real-world distance are known, scale so that pair
    matches (``ref_points=(i, j), ref_distance=d``). Otherwise normalize
    so the RMS distance from the centroid is 1, which keeps downstream
    mesh thresholds (fractions of the bbox diagonal) stable across videos.
    """
    S_eucl = np.asarray(S_eucl, dtype=np.float64)
    n_points = S_eucl.shape[1]

    if ref_points is not None:
        if ref_distance is None or ref_distance <= 0:
            raise ValueError("ref_distance must be > 0 when ref_points is given")
        i, j = int(ref_points[0]), int(ref_points[1])
        if not (0 <= i < n_points and 0 <= j < n_points):
            raise IndexError(
                f"ref_points {(i, j)} out of range for {n_points} reconstructed points"
            )
        d_recon = np.linalg.norm(S_eucl[:, i] - S_eucl[:, j])
        if d_recon <= 1e-12:
            raise ValueError(f"reference points {(i, j)} coincide; cannot set scale")
        return (float(ref_distance) / d_recon) * S_eucl

    centered = S_eucl - S_eucl.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(centered**2, axis=0)))
    if rms <= 1e-12:
        return S_eucl
    return S_eucl / rms

def accuracy_assesment(matrix_path, ref_points, ref_distance, test_points, test_distance):
    """Scale with one known distance, then report the error on a second one."""
    S_eucl = main_2(matrix_path, show_plot=False, plot_first_frame=False)
    S_scaled = d(S_eucl, ref_points, ref_distance)
    i, j = test_points
    dist = np.linalg.norm(S_scaled[:, i] - S_scaled[:, j])
    error = abs(test_distance - dist) / test_distance * 100
    print(f"{matrix_path}: dist={dist:.3f} expected={test_distance} error={error:.2f}%")
    return error

def main_2(matrix_path, show_plot=True, plot_first_frame=True, ref_points=None, ref_distance=None):
    A, R, S = a(matrix_path, plot_first_frame=plot_first_frame)
    L = b(A)
    R_eucl, S_eucl = c(L, R, S)
    S_scaled = d(S_eucl, ref_points=ref_points, ref_distance=ref_distance)
    data_dir = os.path.dirname(matrix_path)
    np.save(os.path.join(data_dir, "q2_S_scaled.npy"), S_scaled)
    plot_reconstruction(
        S_scaled,
        "Metric",
        save_path=os.path.join(data_dir, "q2_S_scaled.png"),
        show=show_plot,
    )
    return S_eucl



if __name__ == "__main__":
    # python q2_metric_upgrade.py <tracks.txt> [ref_i ref_j ref_distance]
    if len(sys.argv) < 2:
        print("usage: python q2_metric_upgrade.py <tracks.txt> [ref_i ref_j ref_distance]")
        sys.exit(1)
    ref_pts, ref_dist = None, None
    if len(sys.argv) >= 5:
        ref_pts = (int(sys.argv[2]), int(sys.argv[3]))
        ref_dist = float(sys.argv[4])
    main_2(sys.argv[1], ref_points=ref_pts, ref_distance=ref_dist)
    
