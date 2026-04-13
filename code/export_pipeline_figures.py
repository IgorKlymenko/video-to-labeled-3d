#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment, misc]

from q1_affine_factorization_method import get_R_S, plot_reconstruction
from reconstruction_pipeline import save_metric_scaled_plot_with_class_legend
from yolo_segmentation import (
    colorize_seg_map,
    rgb_uint8_texture_with_dense_seg_overlay,
)

code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
if os.path.dirname(code_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(code_dir))


def _save_affine_png(out_dir: str, png_path: str) -> bool:
    tracks_txt = os.path.join(out_dir, "tracks.txt")
    if not os.path.isfile(tracks_txt):
        print(f"skip affine: missing {tracks_txt}")
        return False

    _, S = get_R_S(tracks_txt, plot_first_frame=False)
    plot_reconstruction(
        S,
        "Affine SfM (factorization)",
        save_path=png_path,
        show=False,
    )
    print(f"saved {png_path}")
    return True


def _save_metric_png(out_dir: str, png_path: str) -> bool:
    pth = os.path.join(out_dir, "q2_S_scaled.npy")
    if not os.path.isfile(pth):
        print(f"skip metric: missing {pth}")
        return False

    S = np.load(pth)
    plot_reconstruction(
        S,
        "Metric SfM (scaled)",
        save_path=png_path,
        show=False,
    )
    print(f"saved {png_path}")
    return True


def _save_yolo_frame0_png(
    out_dir: str, png_path: str, seg_alpha: float
) -> bool:
    npz_path = os.path.join(out_dir, "seg_class_maps.npz")
    if not os.path.isfile(npz_path):
        print(f"skip yolo frame0: missing {npz_path}")
        return False

    data = np.load(npz_path)
    maps = data["maps"]
    seg0 = maps[0]

    frame_path = os.path.join(out_dir, "frame000000.jpg")
    if os.path.isfile(frame_path):
        base_bgr = cv2.imread(frame_path)
        if base_bgr is None:
            print(f"skip photo overlay: can't read {frame_path}")
            bgr = colorize_seg_map(seg0)
        else:
            base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb_uint8_texture_with_dense_seg_overlay(
                base_rgb, seg0, overlay_alpha=seg_alpha
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        print(
            f"no frame000000.jpg; saving flat class map"
        )
        bgr = colorize_seg_map(seg0)

    cv2.imwrite(png_path, bgr)
    print(f"saved {png_path}")
    return True


def _save_metric_labeled_png(out_dir: str, png_path: str) -> bool:
    out = save_metric_scaled_plot_with_class_legend(
        out_dir, show=False, save_path=png_path
    )
    if out is None:
        print(
            "skip metric labeled: need q2_S_scaled.npy and point_labels "
            "(or seg_class_maps.npz + tracks.txt to rebuild labels)."
        )
        return False
    return True


def _save_mesh_png(mesh_path: str, png_path: str, width: int, height: int) -> bool:
    if not os.path.isfile(mesh_path):
        print(f"skip mesh: missing {mesh_path}")
        return False
    if o3d is None:
        print("skip mesh: open3d not installed.")
        return False

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        print(f"skip mesh: empty geometry in {mesh_path}")
        return False
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(width=width, height=height, visible=False)
    if not ok:
        print("skip mesh: could not create offscreen window")
        return False
    try:
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.light_on = True
        ctr = vis.get_view_control()
        ctr.set_zoom(0.62)
        for _ in range(3):
            vis.poll_events()
            vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
    finally:
        vis.destroy_window()

    img_u8 = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(png_path, bgr)
    print(f"saved {png_path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export pipeline PNGs from an existing reconstruction folder."
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Folder with tracks.txt, q2_S_scaled.npy, seg_class_maps.npz, etc.",
    )
    ap.add_argument(
        "--export-dir",
        default=None,
        help="Where to write pipeline_*.png (default: same as --out-dir)",
    )
    ap.add_argument(
        "--seg-alpha",
        type=float,
        default=0.4,
        help="YOLO class layer strength on frame000000.jpg (0=photo only, 1=flat classes). Default: 0.4",
    )
    ap.add_argument(
        "--write-3d-pngs",
        action="store_true",
        help="Also save pipeline_01_affine … pipeline_05_mesh (offscreen). Omit for YOLO PNG only.",
    )
    ap.add_argument(
        "--mesh",
        default=None,
        help="Path to mesh_reconstruction.obj (default: <out-dir>/mesh_reconstruction.obj)",
    )
    ap.add_argument("--width", type=int, default=1280, help="Offscreen mesh render width")
    ap.add_argument("--height", type=int, default=960, help="Offscreen mesh render height")
    ap.add_argument(
        "--skip-mesh",
        action="store_true",
        help="With --write-3d-pngs: do not render pipeline_05_mesh.png",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    export_dir = os.path.abspath(os.path.expanduser(args.export_dir or out_dir))
    os.makedirs(export_dir, exist_ok=True)

    mesh_path = (
        os.path.abspath(os.path.expanduser(args.mesh))
        if args.mesh
        else os.path.join(out_dir, "mesh_reconstruction.obj")
    )

    prefix = os.path.join(export_dir, "pipeline_")
    seg_alpha = float(max(0.0, min(1.0, args.seg_alpha)))

    if args.write_3d_pngs:
        _save_affine_png(out_dir, f"{prefix}01_affine.png")
        _save_metric_png(out_dir, f"{prefix}02_metric.png")
    _save_yolo_frame0_png(out_dir, f"{prefix}03_yolo_seg_frame0.png", seg_alpha=seg_alpha)
    if args.write_3d_pngs:
        _save_metric_labeled_png(out_dir, f"{prefix}04_metric_labeled.png")
        if not args.skip_mesh:
            _save_mesh_png(mesh_path, f"{prefix}05_mesh.png", args.width, args.height)
        else:
            print("skip mesh png")


if __name__ == "__main__":
    main()
