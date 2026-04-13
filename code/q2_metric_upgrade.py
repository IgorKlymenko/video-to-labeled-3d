from q1_affine_factorization_method import get_R_S, plot_reconstruction
import os
import numpy as np

matrix1_path = "media/hotel/tracks.txt"
matrix2_path = "media/poor/tracks.txt"
matrix3_path = "media/good/tracks.txt"

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

def d(S_eucl):
    d_real = 9
    d_recon = np.linalg.norm(S_eucl[:, 9] - S_eucl[:, 16])

    scale = d_real / d_recon

    S_scaled = scale * S_eucl
    return S_scaled

def accuracy_assesment():
    S_eucl = main_2("media/hotel/tracks.txt")
    d_real = 9
    d_recon = np.linalg.norm(S_eucl[:, 9] - S_eucl[:, 16])
    scale = d_real / d_recon
    S_scaled = scale * S_eucl
    print("hotel dist:", np.linalg.norm(S_scaled[:, 9] - S_scaled[:, 140]))
    hotel_error = (np.abs(5 - np.linalg.norm(S_scaled[:, 9] - S_scaled[:, 140])) / 5 * 100)
    print(f"hotel error: {hotel_error:.2f}%")

    S_eucl = main_2("media/poor/tracks.txt")
    d_real = 20 #mm
    d_recon = np.linalg.norm(S_eucl[:, 479] - S_eucl[:, 924])
    scale = d_real / d_recon
    S_scaled = scale * S_eucl
    print("poor dist:", np.linalg.norm(S_scaled[:, 924] - S_scaled[:, 963]))
    poor_error = (np.abs(6 - np.linalg.norm(S_scaled[:, 924] - S_scaled[:, 963])) / 6 * 100)
    print(f"poor error: {poor_error:.2f}%")

    S_eucl = main_2("media/good/tracks.txt")
    d_real = 20 #mm
    d_recon = np.linalg.norm(S_eucl[:, 919] - S_eucl[:, 998])
    scale = d_real / d_recon
    S_scaled = scale * S_eucl
    print("good dist:", np.linalg.norm(S_scaled[:, 919] - S_scaled[:, 54]))
    good_error = (np.abs(10 - np.linalg.norm(S_scaled[:, 919] - S_scaled[:, 54])) / 10 * 100)
    print(f"good error: {good_error:.2f}%")
    #return S_scaled

def main_2(matrix_path, show_plot=True, plot_first_frame=True):
    A, R, S = a(matrix_path, plot_first_frame=plot_first_frame)
    L = b(A)
    R_eucl, S_eucl = c(L, R, S)
    S_scaled = d(S_eucl)
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
    #accuracy_assesment()

    # test all datasets
    for matrix_path in [matrix1_path, matrix2_path, matrix3_path]:
        main_2(matrix_path)

    # test one dataset
    #main_2("media/poor/tracks.txt")
    
