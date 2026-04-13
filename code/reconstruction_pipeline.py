from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment, misc]

from q1_affine_factorization_method import normalize_marix
from q2_metric_upgrade import main_2
from q3_texturing import (
    save_class_colored_mesh,
    save_textured_mesh,
    save_textured_meshes_per_class,
)
from yolo_segmentation import (
    bgr_colors_for_point_labels,
    label_tracks_from_seg_maps_xy,
    overlay_alpha_default,
    rgb01_seg_overlay_on_image,
)


def track_arrays_from_dict(
    refined_tracks: Dict[int, List], num_frames: int
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    column_ids = sorted(refined_tracks.keys())
    n_features = len(column_ids)
    track_x = np.full((n_features, num_frames), np.nan)
    track_y = np.full((n_features, num_frames), np.nan)
    for i, track_id in enumerate(column_ids):
        track = refined_tracks[track_id]
        padded = [track[j] if j < len(track) else None for j in range(num_frames)]
        arr = np.array([(p[0], p[1]) if p is not None else (np.nan, np.nan) for p in padded])
        track_x[i, :] = arr[:, 0]
        track_y[i, :] = arr[:, 1]
    return track_x, track_y, column_ids


def write_tracks_txt(path: str, track_x: np.ndarray, track_y: np.ndarray) -> None:
    num_frames = track_x.shape[1]
    with open(path, "w") as f:
        for frame_idx in range(num_frames):
            x_row = " ".join(str(v) if not np.isnan(v) else "nan" for v in track_x[:, frame_idx])
            f.write(x_row + "\n")
            y_row = " ".join(str(v) if not np.isnan(v) else "nan" for v in track_y[:, frame_idx])
            f.write(y_row + "\n")
    print(f"saved tracks: {path}")


def save_point_labels(path: str, labels: np.ndarray) -> None:
    np.save(path, labels)
    print(f"saved labels: {path}")


def load_point_labels_if_present(data_dir: str) -> Optional[np.ndarray]:
    p = os.path.join(data_dir, "point_labels.npy")
    if os.path.isfile(p):
        return np.load(p)
    return None


def load_tracks_xy_nan(tracks_txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(tracks_txt_path) as f:
        lines = f.readlines()
    W = np.array(
        [[float(x) if x.lower() != "nan" else np.nan for x in line.split()] for line in lines]
    )
    n_rows, n_points = W.shape
    if n_rows < 2 or n_rows % 2 != 0:
        raise ValueError(f"Unexpected tracks.txt shape: {W.shape}")
    n_frames = n_rows // 2
    track_x = np.zeros((n_points, n_frames), dtype=np.float64)
    track_y = np.zeros((n_points, n_frames), dtype=np.float64)
    for t in range(n_frames):
        track_x[:, t] = W[2 * t, :]
        track_y[:, t] = W[2 * t + 1, :]
    return track_x, track_y


def rebuild_point_labels_from_seg_bundle(out_dir: str) -> Optional[np.ndarray]:
    npz_path = os.path.join(out_dir, "seg_class_maps.npz")
    tracks_txt = os.path.join(out_dir, "tracks.txt")
    if not os.path.isfile(npz_path) or not os.path.isfile(tracks_txt):
        return None

    data = np.load(npz_path)
    seg_maps = data["maps"]
    track_x, track_y = load_tracks_xy_nan(tracks_txt)
    pl = label_tracks_from_seg_maps_xy(seg_maps, track_x, track_y)
    save_point_labels(os.path.join(out_dir, "point_labels.npy"), pl)
    print(
        f"rebuilt point_labels.npy ({pl.shape[0]} labels) from seg_class_maps.npz + tracks.txt"
    )
    return pl


def _align_point_labels_to_S(
    out_dir: str, n_columns: int, context: str
) -> Optional[np.ndarray]:
    labels = load_point_labels_if_present(out_dir)
    if labels is not None and labels.shape[0] == n_columns:
        return np.asarray(labels).astype(np.int32).reshape(-1)
    if labels is not None and labels.shape[0] != n_columns:
        print(
            f"{context}: point_labels length {labels.shape[0]} != {n_columns} ({context}); "
            "rebuilding from seg maps + tracks..."
        )
    else:
        print(f"{context}: no point_labels.npy")
    rebuilt = rebuild_point_labels_from_seg_bundle(out_dir)
    if rebuilt is not None and rebuilt.shape[0] == n_columns:
        return np.asarray(rebuilt).astype(np.int32).reshape(-1)
    if labels is not None and labels.shape[0] != n_columns:
        print(
            f"Still mismatched or missing seg bundle. Re-run the full pipeline with --yolo, "
            f"or delete stale point_labels.npy in {out_dir!r}."
        )
    return None


# Single frame for UV/triangulation matches baked texture (reduces 2D/3D inconsistency).
mesh_uv_frame_span = 1

# Drop long 2D edges (spikes) when caller does not override — looser default to avoid “Swiss cheese”.
mesh_edge_frac_fallback = 0.058

# Drop triangles whose longest 3D edge exceeds this fraction of the 3D bbox diagonal.
mesh_edge_3d_frac_fallback = 0.10


def effective_mesh_max_edge_frac(mesh_max_edge_frac: Optional[float]) -> Optional[float]:
    if mesh_max_edge_frac is None:
        return mesh_edge_frac_fallback
    if mesh_max_edge_frac <= 0:
        return None
    return mesh_max_edge_frac


def effective_mesh_max_edge_3d_frac(mesh_max_edge_3d_frac: Optional[float]) -> float:
    if mesh_max_edge_3d_frac is None:
        return float(mesh_edge_3d_frac_fallback)
    if mesh_max_edge_3d_frac <= 0:
        return 0.0
    return float(mesh_max_edge_3d_frac)


def points_2d_for_mesh_uv(
    W: np.ndarray,
    *,
    frame_start: int = 0,
    mesh_uv_num_frames: Optional[int] = None,
) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    n_frames = W.shape[0] // 2
    p = W.shape[1]
    f0 = max(0, int(frame_start))
    if f0 >= n_frames:
        f0 = 0
    if mesh_uv_num_frames is not None and int(mesh_uv_num_frames) <= 0:
        f1 = n_frames
    elif mesh_uv_num_frames is None:
        f1 = min(n_frames, f0 + mesh_uv_frame_span)
    else:
        f1 = min(n_frames, f0 + int(mesh_uv_num_frames))
    if f1 <= f0:
        f1 = min(n_frames, f0 + 1)

    xs = np.stack([W[2 * f, :] for f in range(f0, f1)], axis=0)
    ys = np.stack([W[2 * f + 1, :] for f in range(f0, f1)], axis=0)
    with np.errstate(invalid="ignore"):
        u = np.nanmean(xs, axis=0)
        v = np.nanmean(ys, axis=0)
    bad = np.isnan(u) | np.isnan(v)
    if np.any(bad):
        all_x = np.stack([W[2 * f, :] for f in range(n_frames)], axis=0)
        all_y = np.stack([W[2 * f + 1, :] for f in range(n_frames)], axis=0)
        with np.errstate(invalid="ignore"):
            uf = np.nanmean(all_x, axis=0)
            vf = np.nanmean(all_y, axis=0)
        u = np.where(bad, uf, u)
        v = np.where(bad, vf, v)
        bad2 = np.isnan(u) | np.isnan(v)
        if np.any(bad2):
            for j in np.where(bad2)[0]:
                for f in range(n_frames):
                    x, y = W[2 * f, j], W[2 * f + 1, j]
                    if not (np.isnan(x) or np.isnan(y)):
                        u[j], v[j] = x, y
                        break
    return np.column_stack((u, v))


def print_class_legend_concise(out_dir: str) -> None:
    labels = load_point_labels_if_present(out_dir)
    if labels is None:
        return
    names = load_seg_class_names_if_present(out_dir)
    uniq = sorted(set(np.asarray(labels).astype(int).tolist()))
    lines = ["Mesh class legend (map_id → label):"]
    for v in uniq:
        if v == 0:
            lines.append("  0  unknown / background")
        elif names:
            nm = names.get(str(v - 1), f"class {v - 1}")
            lines.append(f"  {v}  {nm}")
        else:
            lines.append(f"  {v}  yolo_cls_index={v - 1}")
    print("\n".join(lines))


def load_seg_class_names_if_present(data_dir: str) -> Optional[Dict[str, str]]:
    p = os.path.join(data_dir, "seg_class_names.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def class_legend_patches(
    uniq_ids: List[int],
    names: Optional[Dict[str, str]],
):
    handles = []
    for v in uniq_ids:
        bgr = bgr_colors_for_point_labels(np.array([v], dtype=np.int32))[0]
        rgb = tuple((bgr[::-1] / 255.0).tolist())
        if v == 0:
            label = "unknown / background"
        elif names:
            nm = names.get(str(v - 1), f"class {v - 1}")
            label = nm
        else:
            label = f"map id {v}"
        handles.append(
            Patch(facecolor=rgb, edgecolor="0.35", linewidth=0.6, label=label)
        )
    return handles


def save_class_legend_png(
    out_dir: str,
    uniq_ids: List[int],
    names: Optional[Dict[str, str]],
) -> str:
    handles = class_legend_patches(uniq_ids, names)
    fig, ax = plt.subplots(figsize=(5.5, max(1.2, 0.38 * len(uniq_ids) + 0.4)))
    ax.axis("off")
    fig.legend(handles=handles, loc="center", fontsize=10, frameon=True)
    path = os.path.join(out_dir, "class_legend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def save_metric_scaled_plot_with_class_legend(
    out_dir: str,
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    write_file: bool = True,
) -> Optional[str]:
    pth = os.path.join(out_dir, "q2_S_scaled.npy")
    if not os.path.isfile(pth):
        return None
    S_scaled = np.load(pth)
    n = S_scaled.shape[1]
    labels = _align_point_labels_to_S(out_dir, n, "q2_S_scaled_labeled")
    if labels is None:
        return None

    bgr = bgr_colors_for_point_labels(labels)
    rgb = bgr[:, ::-1].astype(np.float64) / 255.0
    uniq = sorted(set(labels.tolist()))
    names = load_seg_class_names_if_present(out_dir)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(S_scaled[0, :], S_scaled[1, :], S_scaled[2, :], c=rgb, marker="o", s=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Metric-scaled reconstruction (class colors + YOLO legend)")
    handles = class_legend_patches(uniq, names)
    fig.subplots_adjust(left=0.02, right=0.72, top=0.92, bottom=0.06)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=9,
    )
    out_png = save_path or os.path.join(out_dir, "q2_S_scaled_labeled.png")
    if write_file:
        plt.savefig(out_png, dpi=140)
        print(f"saved {out_png}")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def save_labeled_reconstruction_viz(
    out_dir: str,
    *,
    show: bool = False,
    S_scaled: Optional[np.ndarray] = None,
    write_ply: bool = True,
    overlay_alpha: Optional[float] = None,
    save_path: Optional[str] = None,
) -> Optional[str]:
    if S_scaled is None:
        pth = os.path.join(out_dir, "q2_S_scaled.npy")
        if not os.path.isfile(pth):
            print("missing q2_S_scaled.npy")
            return None
        S_scaled = np.load(pth)
    else:
        S_scaled = np.asarray(S_scaled)
    n = S_scaled.shape[1]
    labels = _align_point_labels_to_S(out_dir, n, "Labeled 3D viz")
    if labels is None:
        return None

    if overlay_alpha is None:
        overlay_alpha = overlay_alpha_default

    image_path = os.path.join(out_dir, "frame000000.jpg")
    if os.path.isfile(image_path):
        tracks_txt = os.path.join(out_dir, "tracks.txt")
        W = normalize_marix(tracks_txt, normalize=False, plot_first_frame=False)
        xs, ys = W[0, :], W[1, :]
        img = mpimg.imread(image_path)
        rgb = rgb01_seg_overlay_on_image(
            img, xs, ys, labels, overlay_alpha=overlay_alpha
        )
    else:
        bgr = bgr_colors_for_point_labels(labels)
        rgb = bgr[:, ::-1].astype(np.float64) / 255.0

    uniq = sorted(set(labels.tolist()))
    names = load_seg_class_names_if_present(out_dir)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(S_scaled[0, :], S_scaled[1, :], S_scaled[2, :], c=rgb, marker="o", s=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Reconstruction (class tint over image where available)")
    handles = class_legend_patches(uniq, names)
    fig.subplots_adjust(left=0.02, right=0.72, top=0.92, bottom=0.06)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=9,
    )
    out_png = save_path or os.path.join(out_dir, "q2_reconstruction_labeled.png")
    plt.savefig(out_png, dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    print(f"saved {out_png}")

    save_class_legend_png(out_dir, uniq, names)

    legend_path = os.path.join(out_dir, "reconstruction_label_legend.txt")
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write(
            "# Map id 0 = background/unknown; id k>0 is YOLO class index (k-1) in composite seg.\n"
        )
        if names:
            f.write("# Class names from seg_class_names.json (model output).\n")
        f.write("\n")
        for v in uniq:
            line = f"map_id={v}"
            if v > 0 and names is not None:
                cls_ix = v - 1
                nm = names.get(str(cls_ix), "?")
                line += f"  yolo_cls_index={cls_ix}  name={nm}"
            elif v > 0:
                line += f"  yolo_cls_index={v - 1}"
            f.write(line + "\n")
    print(f"saved {legend_path}")

    if write_ply:
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(S_scaled.T.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            ply = os.path.join(out_dir, "reconstruction_labeled.ply")
            o3d.io.write_point_cloud(ply, pcd)
            print(f"saved {ply}")

    return out_png


def run_metric_reconstruction(
    tracks_txt_path: str,
    show_plot: bool = True,
    plot_first_frame: bool = True,
):
    return main_2(
        tracks_txt_path,
        show_plot=show_plot,
        plot_first_frame=plot_first_frame,
    )


def try_save_metrics(
    out_dir: str,
    *,
    mesh_max_edge_frac: Optional[float] = None,
    mesh_max_edge_3d_frac: Optional[float] = None,
    mesh_uv_frame_start: int = 0,
    mesh_uv_num_frames: Optional[int] = None,
    mesh_drop_cross_class_triangles: bool = True,
) -> Optional[str]:
    from evaluation_metrics import save_metrics

    try:
        return save_metrics(
            out_dir,
            mesh_max_edge_frac=mesh_max_edge_frac,
            mesh_max_edge_3d_frac=mesh_max_edge_3d_frac,
            mesh_uv_frame_start=mesh_uv_frame_start,
            mesh_uv_num_frames=mesh_uv_num_frames,
            mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
        )
    except Exception as ex:
        print(f"metrics failed: {ex}")
        return None


def class_mesh_from_track_folder(
    out_dir: str,
    S_eucl: np.ndarray,
    *,
    show: bool = False,
    save_ply: bool = True,
    mesh_basename: str = "mesh_class_colored",
    overlay_alpha: Optional[float] = None,
    mesh_max_edge_frac: Optional[float] = None,
    mesh_max_edge_3d_frac: Optional[float] = None,
    mesh_uv_frame_start: int = 0,
    mesh_uv_num_frames: Optional[int] = None,
    mesh_drop_cross_class_triangles: bool = True,
):
    labels = load_point_labels_if_present(out_dir)
    if labels is None:
        return None

    tracks_txt = os.path.join(out_dir, "tracks.txt")
    W = normalize_marix(
        tracks_txt, normalize=False, plot_first_frame=False, fill_nan=False
    )
    points_2d = points_2d_for_mesh_uv(
        W, frame_start=mesh_uv_frame_start, mesh_uv_num_frames=mesh_uv_num_frames
    )
    image_path = os.path.join(out_dir, "frame000000.jpg")
    img = mpimg.imread(image_path) if os.path.isfile(image_path) else None
    if show:
        print_class_legend_concise(out_dir)
    save_class_colored_mesh(
        S_eucl,
        points_2d,
        labels,
        output_name=os.path.join(out_dir, mesh_basename),
        show=show,
        save_ply=save_ply,
        save_vertex_obj=save_ply,
        image_rgb=img,
        overlay_alpha=overlay_alpha,
        max_triangle_edge_frac=effective_mesh_max_edge_frac(mesh_max_edge_frac),
        max_triangle_edge_3d_frac=effective_mesh_max_edge_3d_frac(mesh_max_edge_3d_frac),
        mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
    )
    return os.path.join(out_dir, f"{mesh_basename}.ply")


def textured_mesh_from_track_folder(
    out_dir: str,
    S_eucl: np.ndarray,
    *,
    show: bool = True,
    save_obj: bool = True,
    mesh_basename: str = "mesh_reconstruction",
    class_overlay_alpha: Optional[float] = None,
    mesh_max_edge_frac: Optional[float] = None,
    mesh_cluster_eps_frac: Optional[float] = None,
    mesh_uv_frame_start: int = 0,
    mesh_uv_num_frames: Optional[int] = None,
    mesh_max_edge_3d_frac: Optional[float] = None,
    mesh_keep_largest_component: bool = False,
    mesh_drop_cross_class_triangles: bool = True,
):
    tracks_txt = os.path.join(out_dir, "tracks.txt")
    image_path = os.path.join(out_dir, "frame000000.jpg")
    W = normalize_marix(
        tracks_txt, normalize=False, plot_first_frame=False, fill_nan=False
    )
    points_2d = points_2d_for_mesh_uv(
        W, frame_start=mesh_uv_frame_start, mesh_uv_num_frames=mesh_uv_num_frames
    )
    img = mpimg.imread(image_path)
    labels = load_point_labels_if_present(out_dir)
    base = os.path.join(out_dir, mesh_basename)

    seg_map_hw = None
    npz_path = os.path.join(out_dir, "seg_class_maps.npz")
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        maps = data["maps"]
        if maps.ndim == 3 and maps.shape[0] >= 1:
            seg_map_hw = np.asarray(maps[0])

    edge = effective_mesh_max_edge_frac(mesh_max_edge_frac)

    if labels is not None:
        print_class_legend_concise(out_dir)
        uniq = sorted(set(np.asarray(labels).astype(int).tolist()))
        names = load_seg_class_names_if_present(out_dir)
        save_class_legend_png(out_dir, uniq, names)
        if seg_map_hw is None:
            print(
                "Note: point_labels.npy present but seg_class_maps.npz missing — "
                "mesh uses photo texture only (no class tint). Re-run with --yolo to export seg maps."
            )

    save_textured_mesh(
        S_eucl,
        points_2d,
        img,
        output_name=base,
        show=show,
        save_obj=save_obj,
        seg_map_hw=seg_map_hw,
        overlay_alpha=class_overlay_alpha,
        max_triangle_edge_frac=edge,
        mesh_cluster_eps_frac=mesh_cluster_eps_frac,
        max_triangle_edge_3d_frac=effective_mesh_max_edge_3d_frac(mesh_max_edge_3d_frac),
        mesh_keep_largest_component=mesh_keep_largest_component,
        point_labels=labels,
        mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
    )


def textured_meshes_per_class_from_track_folder(
    out_dir: str,
    S_eucl: np.ndarray,
    *,
    class_overlay_alpha: Optional[float] = None,
    mesh_max_edge_frac: Optional[float] = None,
    mesh_cluster_eps_frac: Optional[float] = None,
    mesh_uv_frame_start: int = 0,
    mesh_uv_num_frames: Optional[int] = None,
    mesh_max_edge_3d_frac: Optional[float] = None,
    mesh_keep_largest_component: bool = False,
):
    labels = load_point_labels_if_present(out_dir)
    if labels is None:
        return

    tracks_txt = os.path.join(out_dir, "tracks.txt")
    image_path = os.path.join(out_dir, "frame000000.jpg")
    W = normalize_marix(
        tracks_txt, normalize=False, plot_first_frame=False, fill_nan=False
    )
    points_2d = points_2d_for_mesh_uv(
        W, frame_start=mesh_uv_frame_start, mesh_uv_num_frames=mesh_uv_num_frames
    )
    img = mpimg.imread(image_path)

    seg_map_hw = None
    npz_path = os.path.join(out_dir, "seg_class_maps.npz")
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        maps = data["maps"]
        if maps.ndim == 3 and maps.shape[0] >= 1:
            seg_map_hw = np.asarray(maps[0])

    save_textured_meshes_per_class(
        S_eucl,
        points_2d,
        img,
        labels,
        out_dir,
        seg_map_hw=seg_map_hw,
        overlay_alpha=class_overlay_alpha,
        max_triangle_edge_frac=effective_mesh_max_edge_frac(mesh_max_edge_frac),
        mesh_cluster_eps_frac=mesh_cluster_eps_frac,
        max_triangle_edge_3d_frac=effective_mesh_max_edge_3d_frac(mesh_max_edge_3d_frac),
        mesh_keep_largest_component=mesh_keep_largest_component,
    )


def video_to_metric_structure(
    video_path: str,
    out_dir: str,
    *,
    label_fn=None,
    show_plot: bool = False,
    write_preview: bool = True,
    preview_fps: float = 15.0,
    redetect_interval: int = 15,
    max_redetect: int = 20,
    show_mesh: bool = False,
    save_mesh_obj: bool = True,
    yolo_seg_weights: Optional[str] = None,
    yolo_device: str = "cpu",
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.25,
    yolo_save_point_labels: bool = True,
    show_labeled_3d: bool = False,
    show_class_mesh: bool = False,
    max_frames: Optional[int] = 120,
    class_overlay_alpha: Optional[float] = None,
    center_roi_frac: Optional[float] = None,
    yolo_mask_features: bool = False,
    yolo_mask_union_frames: int = 1,
    redetect_full_frame: Optional[bool] = None,
    feature_quality: float = 0.02,
    min_feature_distance: int = 12,
    feature_max_corners: int = 1500,
    feature_grid: Tuple[int, int] = (4, 4),
    mesh_max_edge_frac: Optional[float] = None,
    mesh_cluster_eps_frac: Optional[float] = None,
    mesh_uv_frame_start: int = 0,
    mesh_uv_num_frames: Optional[int] = None,
    mesh_max_edge_3d_frac: Optional[float] = None,
    mesh_keep_largest_component: bool = False,
    track_scale: float = 1.0,
    fast_lk: bool = True,
    lk_preset: str = "fast",
    lk_max_jump_frac: Optional[float] = 0.12,
    feature_distribute_experimental: bool = False,
    feature_spread_min_dist: Optional[float] = None,
    split_meshes_by_class: bool = False,
    mesh_drop_cross_class_triangles: bool = True,
    feature_use_clahe: bool = False,
    dump_metrics: bool = True,
):
    import feature_tracker as ft

    os.makedirs(out_dir, exist_ok=True)
    bundle = ft.track_video_to_disk(
        video_path,
        out_dir,
        label_fn=label_fn,
        redetect_interval=redetect_interval,
        max_redetect=max_redetect,
        write_preview=write_preview,
        preview_fps=preview_fps,
        center_roi_frac=center_roi_frac,
        yolo_mask_features=yolo_mask_features,
        yolo_mask_union_frames=yolo_mask_union_frames,
        redetect_full_frame=redetect_full_frame,
        feature_quality=feature_quality,
        min_feature_distance=min_feature_distance,
        feature_max_corners=feature_max_corners,
        feature_grid=feature_grid,
        yolo_seg_weights=yolo_seg_weights,
        yolo_device=yolo_device,
        yolo_imgsz=yolo_imgsz,
        yolo_conf=yolo_conf,
        yolo_save_point_labels=yolo_save_point_labels,
        max_frames=max_frames,
        track_scale=track_scale,
        fast_lk=fast_lk,
        lk_preset=lk_preset,
        lk_max_jump_frac=lk_max_jump_frac,
        feature_distribute_experimental=feature_distribute_experimental,
        feature_spread_min_dist=feature_spread_min_dist,
        feature_use_clahe=feature_use_clahe,
    )
    if bundle is None:
        return None
    tracks_txt = os.path.join(out_dir, "tracks.txt")
    S_eucl = run_metric_reconstruction(
        tracks_txt, show_plot=show_plot, plot_first_frame=False
    )
    if show_mesh or save_mesh_obj:
        textured_mesh_from_track_folder(
            out_dir,
            S_eucl,
            show=show_mesh,
            save_obj=save_mesh_obj,
            class_overlay_alpha=class_overlay_alpha,
            mesh_max_edge_frac=mesh_max_edge_frac,
            mesh_cluster_eps_frac=mesh_cluster_eps_frac,
            mesh_uv_frame_start=mesh_uv_frame_start,
            mesh_uv_num_frames=mesh_uv_num_frames,
            mesh_max_edge_3d_frac=mesh_max_edge_3d_frac,
            mesh_keep_largest_component=mesh_keep_largest_component,
            mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
        )
    if split_meshes_by_class and load_point_labels_if_present(out_dir) is not None:
        textured_meshes_per_class_from_track_folder(
            out_dir,
            S_eucl,
            class_overlay_alpha=class_overlay_alpha,
            mesh_max_edge_frac=mesh_max_edge_frac,
            mesh_cluster_eps_frac=mesh_cluster_eps_frac,
            mesh_uv_frame_start=mesh_uv_frame_start,
            mesh_uv_num_frames=mesh_uv_num_frames,
            mesh_max_edge_3d_frac=mesh_max_edge_3d_frac,
            mesh_keep_largest_component=mesh_keep_largest_component,
        )
    if load_point_labels_if_present(out_dir) is not None:
        save_labeled_reconstruction_viz(
            out_dir, show=show_labeled_3d, overlay_alpha=class_overlay_alpha
        )
        save_metric_scaled_plot_with_class_legend(
            out_dir, show=show_labeled_3d
        )
        if show_class_mesh:
            class_mesh_from_track_folder(
                out_dir,
                S_eucl,
                show=True,
                overlay_alpha=class_overlay_alpha,
                mesh_max_edge_frac=mesh_max_edge_frac,
                mesh_max_edge_3d_frac=mesh_max_edge_3d_frac,
                mesh_uv_frame_start=mesh_uv_frame_start,
                mesh_uv_num_frames=mesh_uv_num_frames,
                mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
            )
    metrics_path: Optional[str] = None
    if dump_metrics:
        metrics_path = try_save_metrics(
            out_dir,
            mesh_max_edge_frac=mesh_max_edge_frac,
            mesh_max_edge_3d_frac=mesh_max_edge_3d_frac,
            mesh_uv_frame_start=mesh_uv_frame_start,
            mesh_uv_num_frames=mesh_uv_num_frames,
            mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
        )
    return {
        "out_dir": out_dir,
        "tracks_txt": tracks_txt,
        "S_scaled_path": os.path.join(out_dir, "q2_S_scaled.npy"),
        "column_ids": bundle["column_ids"],
        "S_eucl": S_eucl,
        "metrics_path": metrics_path,
    }

