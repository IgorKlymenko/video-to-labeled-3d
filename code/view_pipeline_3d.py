#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d

from q1_affine_factorization_method import get_R_S, plot_reconstruction
from reconstruction_pipeline import save_metric_scaled_plot_with_class_legend

code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)


def _view_affine(out_dir: str) -> None:
    tracks_txt = os.path.join(out_dir, "tracks.txt")
    if not os.path.isfile(tracks_txt):
        print(f"skip affine: missing {tracks_txt}")
        return

    _, S = get_R_S(tracks_txt, plot_first_frame=False)
    plot_reconstruction(S, "Affine SfM — close window to continue", show=True)


def _view_metric(out_dir: str) -> None:
    pth = os.path.join(out_dir, "q2_S_scaled.npy")
    if not os.path.isfile(pth):
        print(f"skip metric: missing {pth}")
        return

    S = np.load(pth)
    plot_reconstruction(S, "Metric SfM — close window to continue", show=True)


def _view_metric_labeled(out_dir: str) -> None:
    out = save_metric_scaled_plot_with_class_legend(
        out_dir, show=True, write_file=False
    )
    if out is None:
        print(
            "skip metric + labels: need q2_S_scaled.npy and point_labels "
            "(or seg_class_maps.npz + tracks.txt)."
        )


def _view_mesh(mesh_path: str) -> None:
    if not os.path.isfile(mesh_path):
        print(f"skip mesh: missing {mesh_path}")
        return

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        print(f"skip mesh: empty geometry in {mesh_path}")
        return
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Mesh — close window to exit",
        width=1280,
        height=960,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="view reconstruction outputs")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Folder with tracks.txt, q2_S_scaled.npy, mesh, etc.",
    )
    ap.add_argument(
        "--mesh",
        default=None,
        help="Path to mesh .obj (default: <out-dir>/mesh_reconstruction.obj)",
    )
    ap.add_argument(
        "--steps",
        nargs="*",
        default=["affine", "metric", "labeled", "mesh"],
        choices=["affine", "metric", "labeled", "mesh", "all"],
        help="Which viewers to open in order (default: all four)",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    mesh_path = (
        os.path.abspath(os.path.expanduser(args.mesh))
        if args.mesh
        else os.path.join(out_dir, "mesh_reconstruction.obj")
    )

    steps = list(args.steps)
    if not steps or "all" in steps:
        steps = ["affine", "metric", "labeled", "mesh"]

    order = ["affine", "metric", "labeled", "mesh"]
    for name in order:
        if name not in steps:
            continue
        print(name)
        if name == "affine":
            _view_affine(out_dir)
        elif name == "metric":
            _view_metric(out_dir)
        elif name == "labeled":
            _view_metric_labeled(out_dir)
        elif name == "mesh":
            _view_mesh(mesh_path)


if __name__ == "__main__":
    main()
