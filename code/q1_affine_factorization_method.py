import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def map_dots(row, frame_name):
    xs = np.array(row[0], dtype=float)
    ys = np.array(row[1], dtype=float)
    
    frame = mpimg.imread(frame_name)
    plt.figure(figsize=(8, 8))  
    plt.imshow(frame, cmap="gray" if frame.ndim == 2 else None)
    plt.scatter(xs, ys, marker='o', c='r', s=10) 

    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x, y, str(i), color='white', fontsize=6, ha='center', va='center')  

    plt.title(frame_name)
    plt.axis('off')
    plt.show()

def normalize_marix(data_location, normalize=True, plot_first_frame=True, fill_nan=True):
    with open(data_location, 'r') as mat_text:
        lines = mat_text.readlines()

    image_path = os.path.join(os.path.dirname(data_location), "frame000000.jpg")
    if plot_first_frame:
        map_dots(np.vstack((lines[0].split(), lines[1].split())), image_path)

    n_rows, n_points = len(lines), len(lines[0].split())
    W = np.array([[float(x) if x.lower() != 'nan' else np.nan for x in line.split()] for line in lines])
    nan_mask = np.isnan(W)
    if fill_nan and np.any(nan_mask):
        for i in range(W.shape[0]):
            row = W[i, :]
            row_mean = np.nanmean(row)
            W[i, :] = np.where(np.isnan(row), row_mean, row)
    if normalize:
        W[0::2, :] -= np.mean(W[0::2, :], axis=1, keepdims=True)
        W[1::2, :] -= np.mean(W[1::2, :], axis=1, keepdims=True)
    return W

def compute_SVD(W):
    U, E, Vh = np.linalg.svd(W)
    return U[:, :3], np.diag(E[:3]), Vh[:3, :]  

def motion_matrix(U, E):
    return U @ np.sqrt(E)

def shape_matrix(E, Vh):
    return np.sqrt(E) @ Vh

def affone_reconstruction(R, S):
    return R @ S

def plot_reconstruction(S, title, save_path=None, show=True, figsize=(10, 6.5), dpi=140):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(S[0, :], S[1, :], S[2, :], c='b', marker='o', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

def get_R_S(matrix_path, plot_first_frame=True):
    W = normalize_marix(matrix_path, plot_first_frame=plot_first_frame)
    U, E, Vh = compute_SVD(W)
    R = motion_matrix(U, E)
    S = shape_matrix(E, Vh)
    return R, S

def main_1(matrix_path):
    R, S = get_R_S(matrix_path)
    data_dir = os.path.dirname(matrix_path)
    np.save(os.path.join(data_dir, "q1_S.npy"), S)
    plot_reconstruction(S, 'Affine', save_path=os.path.join(data_dir, "q1_affineReconstruction.png"))
    return R, S

def main():
    matrix1_path = "media/hotel/tracks.txt"
    matrix2_path = "media/poor/tracks.txt"
    matrix3_path = "media/good/tracks.txt"

    # test all datasets
    for matrix_path in [matrix1_path, matrix2_path, matrix3_path]:
        main_1(matrix_path)

if __name__ == "__main__":
    main()
