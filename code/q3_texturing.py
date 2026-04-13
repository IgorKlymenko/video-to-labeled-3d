from __future__ import annotations

import os
from typing import Optional

import matplotlib.image as mpimg
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, cKDTree

from q1_affine_factorization_method import normalize_marix
from yolo_segmentation import (
    overlay_alpha_default,
    rgb_uint8_texture_with_dense_seg_overlay,
    rgb01_color_table,
    rgb01_sample_image,
)

# mesh safety (see reconstruction_pipeline matching values)
mesh_mad_k_fallback = 6.5
mesh_edge_3d_frac_fallback = 0.10


def mesh_mask_inlier_points_3d(
    S: np.ndarray,
    *,
    mad_k: float = mesh_mad_k_fallback,
    dist_lo_pct: float = 0.15,
    dist_hi_pct: float = 99.85,
) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    p = S.shape[1]
    finite = np.isfinite(S).all(axis=0)
    if not finite.any():
        return np.ones(p, dtype=bool)

    mk = float(mad_k)
    for _ in range(5):
        med = np.median(S[:, finite], axis=1, keepdims=True)
        mad = np.median(np.abs(S - med), axis=1, keepdims=True)
        mad = np.maximum(mad, 1e-9 * (np.abs(med) + 1.0))
        z = np.abs(S - med) / mad
        m_mad = z.max(axis=0) <= mk

        ctr = med.flatten()
        d = np.linalg.norm(S - ctr.reshape(3, 1), axis=0)
        base = finite & m_mad
        if int(base.sum()) < 8:
            lo, hi = np.percentile(d[finite], [dist_lo_pct, dist_hi_pct])
        else:
            lo, hi = np.percentile(d[base], [dist_lo_pct, dist_hi_pct])
        m_dist = (d >= lo) & (d <= hi)
        mask = finite & m_mad & m_dist
        n = int(mask.sum())
        need = max(30, int(np.ceil(0.045 * p)))
        if n >= max(3, min(need, p - 1)) or mk >= 14.0:
            return mask
        mk *= 1.55
    return mask


def _filter_simplices_by_max_edge_3d(
    S: np.ndarray,
    simplices: np.ndarray,
    max_edge: float,
) -> np.ndarray:
    if max_edge is None or max_edge <= 0 or simplices.size == 0:
        return simplices
    keep = []
    for sim in simplices:
        i, j, k = int(sim[0]), int(sim[1]), int(sim[2])
        p0, p1, p2 = S[:, i], S[:, j], S[:, k]
        e = [
            float(np.linalg.norm(p0 - p1)),
            float(np.linalg.norm(p1 - p2)),
            float(np.linalg.norm(p2 - p0)),
        ]
        if max(e) <= max_edge:
            keep.append(sim)
    if not keep:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(keep, dtype=np.int32)


def _simplices_with_2d_and_3d_edge_limits(
    S: np.ndarray,
    points_2d: np.ndarray,
    tri: Delaunay,
    image_w: int,
    image_h: int,
    max_triangle_edge_frac: Optional[float],
    max_triangle_edge_3d_frac: Optional[float],
) -> np.ndarray:
    simplices = _triangle_simplices_with_edge_limit(
        points_2d, tri, image_w, image_h, max_triangle_edge_frac
    )
    if simplices.size == 0:
        return simplices
    if max_triangle_edge_3d_frac is not None and float(max_triangle_edge_3d_frac) <= 0:
        return simplices
    span = S.max(axis=1) - S.min(axis=1)
    diag3 = float(np.linalg.norm(span))
    if diag3 <= 1e-12:
        return simplices
    frac = (
        float(max_triangle_edge_3d_frac)
        if max_triangle_edge_3d_frac is not None
        else mesh_edge_3d_frac_fallback
    )
    if frac <= 0:
        return simplices
    max3 = frac * diag3
    filt = _filter_simplices_by_max_edge_3d(S, simplices, max3)
    if filt.shape[0] == 0:
        print(
            "Mesh: all triangles removed by 3D edge filter; relaxing 3D limit 1.5× "
            "(or set --mesh-max-edge-3d-frac higher)."
        )
        filt = _filter_simplices_by_max_edge_3d(S, simplices, max3 * 1.5)
    if filt.shape[0] == 0:
        print("mesh: 3d edge filter empty; using 2d-only triangles")
        return simplices
    print(
        f"Mesh: 3D edge filter kept {filt.shape[0]}/{simplices.shape[0]} triangles "
        f"(max edge {frac:.3f}·3D bbox diagonal ≈ {max3:.4f} units)."
    )
    return filt


def _mesh_keep_largest_connected_component(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    cluster_ids, n_tris, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    n_tris = np.asarray(n_tris)
    if n_tris.size == 0:
        return mesh
    largest = int(np.argmax(n_tris))
    remove_mask = cluster_ids != largest
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print(f"Mesh: kept largest connected component ({n_t} triangles, {n_v} vertices).")
    return mesh


def _filter_simplices_by_max_edge_2d(
    points_2d: np.ndarray,
    simplices: np.ndarray,
    max_edge_px: float,
) -> np.ndarray:
    if max_edge_px is None or max_edge_px <= 0 or simplices.size == 0:
        return simplices
    keep = []
    for sim in simplices:
        i, j, k = int(sim[0]), int(sim[1]), int(sim[2])
        p0, p1, p2 = points_2d[i], points_2d[j], points_2d[k]
        e = [
            float(np.linalg.norm(p0 - p1)),
            float(np.linalg.norm(p1 - p2)),
            float(np.linalg.norm(p2 - p0)),
        ]
        if max(e) <= max_edge_px:
            keep.append(sim)
    if not keep:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(keep, dtype=np.int32)


def _filter_simplices_cross_class(
    simplices: np.ndarray,
    vertex_labels: np.ndarray,
) -> np.ndarray:
    if simplices.size == 0:
        return simplices
    lab = np.asarray(vertex_labels, dtype=np.int32).reshape(-1)
    keep: list = []
    for sim in simplices:
        i, j, k = int(sim[0]), int(sim[1]), int(sim[2])
        a, b, c = lab[i], lab[j], lab[k]
        bad = False
        for u, v in ((a, b), (a, c), (b, c)):
            if u > 0 and v > 0 and u != v:
                bad = True
                break
        if not bad:
            keep.append(sim)
    if not keep:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(keep, dtype=np.int32)


def _triangle_simplices_with_edge_limit(
    points_2d: np.ndarray,
    tri: Delaunay,
    image_w: int,
    image_h: int,
    max_triangle_edge_frac: Optional[float],
) -> np.ndarray:
    simplices = np.asarray(tri.simplices, dtype=np.int32)
    if max_triangle_edge_frac is None or max_triangle_edge_frac <= 0:
        return simplices
    diag = float(np.hypot(image_w, image_h))
    max_edge = max_triangle_edge_frac * diag
    filt = _filter_simplices_by_max_edge_2d(points_2d, simplices, max_edge)
    if filt.shape[0] == 0:
        print(
            "Mesh: all triangles removed by edge filter; relaxing max edge 2x "
            "(or increase --mesh-max-edge-frac)."
        )
        filt = _filter_simplices_by_max_edge_2d(points_2d, simplices, max_edge * 2.0)
    if filt.shape[0] == 0:
        print("mesh: 2d edge filter empty; using full delaunay")
        return simplices
    print(
        f"Mesh: kept {filt.shape[0]}/{simplices.shape[0]} triangles "
        f"(max 2D edge {max_triangle_edge_frac:.3f} of image diagonal)."
    )
    return filt


def _2d_connected_components_eps(
    points_2d: np.ndarray, eps: float, min_samples: int = 3
) -> list:
    n = points_2d.shape[0]
    if n < min_samples:
        return [np.arange(n)] if n > 0 else []

    tree = cKDTree(points_2d)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(int(a), int(b))

    roots = np.array([find(i) for i in range(n)])
    clusters: list = []
    for r in np.unique(roots):
        idx = np.where(roots == r)[0]
        if len(idx) >= min_samples:
            clusters.append(idx)
    if not clusters:
        return [np.arange(n)]
    return clusters


def _triangle_face_label(a: int, b: int, c: int) -> int:
    a, b, c = int(a), int(b), int(c)
    for cand in (a, b, c):
        if (a == cand) + (b == cand) + (c == cand) >= 2:
            return cand
    return 0


def save_class_colored_mesh(
    S_euclid,
    points_2d,
    point_labels,
    output_name="mesh_class_colored",
    show=True,
    save_ply=True,
    save_vertex_obj: bool = False,
    image_rgb=None,
    overlay_alpha=None,
    max_triangle_edge_frac: Optional[float] = None,
    max_triangle_edge_3d_frac: Optional[float] = None,
    mesh_mad_k: float = mesh_mad_k_fallback,
    mesh_drop_cross_class_triangles: bool = True,
):
    if overlay_alpha is None:
        overlay_alpha = overlay_alpha_default
    overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))

    print("triangulating class mesh...")

    point_labels = np.asarray(point_labels).astype(np.int32).reshape(-1)
    S = np.asarray(S_euclid, dtype=np.float64)
    p = S.shape[1]
    if points_2d.shape[0] != p or point_labels.shape[0] != p:
        raise ValueError("S columns, points_2d rows, and point_labels must match")

    m3d = mesh_mask_inlier_points_3d(S, mad_k=mesh_mad_k)
    if int(m3d.sum()) < 3:
        print("skip class mesh: too few inliers")
        return
    if int(m3d.sum()) < p:
        print(
            f"Class mesh: 3D inlier filter kept {int(m3d.sum())}/{p} points."
        )
    S = S[:, m3d]
    points_2d = np.asarray(points_2d, dtype=np.float64)[m3d]
    point_labels = point_labels[m3d]
    p = S.shape[1]

    tri = Delaunay(points_2d)
    if image_rgb is not None:
        _h, _w = image_rgb.shape[0], image_rgb.shape[1]
    else:
        _w = max(int(np.ceil(float(points_2d[:, 0].max()))) + 1, 1)
        _h = max(int(np.ceil(float(points_2d[:, 1].max()))) + 1, 1)
    simplices = _simplices_with_2d_and_3d_edge_limits(
        S, points_2d, tri, _w, _h, max_triangle_edge_frac, max_triangle_edge_3d_frac
    )
    if simplices.size == 0:
        print("skip class mesh: no triangles")
        return

    if mesh_drop_cross_class_triangles:
        n_b = int(simplices.shape[0])
        simplices = _filter_simplices_cross_class(simplices, point_labels)
        if simplices.size == 0:
            print("skip class mesh: no triangles after class filter")
            return
        if simplices.shape[0] < n_b:
            print(
                f"Class mesh: cross-class filter kept {simplices.shape[0]}/{n_b} triangles "
                "(removed faces spanning two YOLO classes)."
            )

    max_id = int(point_labels.max()) if point_labels.size else 0
    table = rgb01_color_table(max_id)

    if image_rgb is not None:
        bases = np.zeros((p, 3), dtype=np.float64)
        for vi in range(p):
            bases[vi] = rgb01_sample_image(
                image_rgb, float(points_2d[vi, 0]), float(points_2d[vi, 1])
            )
    else:
        bases = np.full((p, 3), 0.92, dtype=np.float64)

    def corner_rgb(vi: int, face_label: int) -> np.ndarray:
        base = bases[vi]
        fl = int(np.clip(face_label, 0, max_id))
        if fl == 0:
            return base
        cls = table[fl]
        return (1.0 - overlay_alpha) * base + overlay_alpha * cls

    verts, cols, faces = [], [], []
    for simplex in simplices:
        i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
        fl = _triangle_face_label(point_labels[i], point_labels[j], point_labels[k])
        fl = int(np.clip(fl, 0, max_id))
        base = len(verts)
        verts.append(S[:, i])
        verts.append(S[:, j])
        verts.append(S[:, k])
        cols.extend([corner_rgb(i, fl), corner_rgb(j, fl), corner_rgb(k, fl)])
        faces.append([base, base + 1, base + 2])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
    mesh.compute_vertex_normals()

    if save_ply:
        ply_path = f"{output_name}.ply"
        o3d.io.write_triangle_mesh(ply_path, mesh)
        print(f"saved {ply_path}")

    if save_vertex_obj:
        obj_path = f"{output_name}.obj"
        o3d.io.write_triangle_mesh(obj_path, mesh, write_triangle_uvs=False)
        print(f"saved {obj_path}")

    if show:
        o3d.visualization.draw_geometries([mesh])


def save_textured_mesh(
    S_euclid,
    points_2d,
    image_rgb,
    output_name="hotel_reconstruction",
    show=True,
    save_obj=True,
    seg_map_hw=None,
    overlay_alpha=None,
    max_triangle_edge_frac: Optional[float] = None,
    mesh_cluster_eps_frac: Optional[float] = None,
    max_triangle_edge_3d_frac: Optional[float] = None,
    mesh_keep_largest_component: bool = False,
    mesh_mad_k: float = mesh_mad_k_fallback,
    point_labels: Optional[np.ndarray] = None,
    mesh_drop_cross_class_triangles: bool = True,
):
    if overlay_alpha is None:
        overlay_alpha = overlay_alpha_default

    tex_rgb = image_rgb
    if seg_map_hw is not None:
        print("baking class overlay texture...")
        tex_rgb = rgb_uint8_texture_with_dense_seg_overlay(
            image_rgb, seg_map_hw, overlay_alpha=overlay_alpha
        )

    h, w = int(tex_rgb.shape[0]), int(tex_rgb.shape[1])
    diag = float(np.hypot(w, h))

    if mesh_cluster_eps_frac is not None and float(mesh_cluster_eps_frac) > 0:
        eps = float(mesh_cluster_eps_frac) * diag
        clusters = _2d_connected_components_eps(
            np.asarray(points_2d, dtype=np.float64), eps, min_samples=3
        )
        print(
            f"Mesh 2D clustering: eps={eps:.1f}px ({float(mesh_cluster_eps_frac):.3f}·diagonal) "
            f"→ {len(clusters)} component(s) with ≥3 points."
        )
        if len(clusters) == 1:
            idx = clusters[0]
            pl = (
                np.asarray(point_labels, dtype=np.int32).reshape(-1)[idx]
                if point_labels is not None
                and len(point_labels) == S_euclid.shape[1]
                else None
            )
            _save_textured_mesh_impl(
                S_euclid[:, idx],
                points_2d[idx],
                tex_rgb,
                output_name,
                show=show,
                save_obj=save_obj,
                max_triangle_edge_frac=max_triangle_edge_frac,
                max_triangle_edge_3d_frac=max_triangle_edge_3d_frac,
                mesh_keep_largest_component=mesh_keep_largest_component,
                mesh_mad_k=mesh_mad_k,
                point_labels=pl,
                mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
            )
            return
        for ci, idx in enumerate(clusters):
            if idx.size < 3:
                continue
            name = f"{output_name}_c{ci}"
            pl = (
                np.asarray(point_labels, dtype=np.int32).reshape(-1)[idx]
                if point_labels is not None
                and len(point_labels) == S_euclid.shape[1]
                else None
            )
            _save_textured_mesh_impl(
                S_euclid[:, idx],
                points_2d[idx],
                tex_rgb,
                name,
                show=show and ci == 0,
                save_obj=save_obj,
                max_triangle_edge_frac=max_triangle_edge_frac,
                max_triangle_edge_3d_frac=max_triangle_edge_3d_frac,
                mesh_keep_largest_component=mesh_keep_largest_component,
                mesh_mad_k=mesh_mad_k,
                point_labels=pl,
                mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
            )
        return

    _save_textured_mesh_impl(
        S_euclid,
        points_2d,
        tex_rgb,
        output_name,
        show=show,
        save_obj=save_obj,
        max_triangle_edge_frac=max_triangle_edge_frac,
        max_triangle_edge_3d_frac=max_triangle_edge_3d_frac,
        mesh_keep_largest_component=mesh_keep_largest_component,
        mesh_mad_k=mesh_mad_k,
        point_labels=point_labels,
        mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
    )


def _save_textured_mesh_impl(
    S_euclid: np.ndarray,
    points_2d: np.ndarray,
    tex_rgb: np.ndarray,
    output_name: str,
    *,
    show: bool,
    save_obj: bool,
    max_triangle_edge_frac: Optional[float],
    max_triangle_edge_3d_frac: Optional[float] = None,
    mesh_keep_largest_component: bool = False,
    mesh_mad_k: float = mesh_mad_k_fallback,
    point_labels: Optional[np.ndarray] = None,
    mesh_drop_cross_class_triangles: bool = True,
) -> None:
    S = np.asarray(S_euclid, dtype=np.float64)
    p0 = int(S.shape[1])
    if p0 < 3:
        print(f"skip mesh (need ≥3 points): {output_name} ({p0} points).")
        return
    if point_labels is not None:
        pl = np.asarray(point_labels, dtype=np.int32).reshape(-1)
        if pl.shape[0] != p0:
            print(
                "Mesh: point_labels length does not match S; skipping cross-class triangle filter."
            )
            point_labels = None
        else:
            point_labels = pl
    mask = mesh_mask_inlier_points_3d(S, mad_k=mesh_mad_k)
    n_keep = int(mask.sum())
    if n_keep < 3:
        print(f"Mesh: too few 3D inliers ({n_keep}/{p0}); skip {os.path.basename(output_name)}.")
        return
    if n_keep < p0:
        print(
            f"Mesh: 3D inlier filter kept {n_keep}/{p0} points (removed extreme reconstruction outliers)."
        )
    S = S[:, mask]
    points_2d = np.asarray(points_2d, dtype=np.float64)[mask]
    if point_labels is not None:
        point_labels = point_labels[mask]
    p = int(points_2d.shape[0])
    print(f"Triangulating {p} 2D points -> {os.path.basename(output_name)}...")
    try:
        tri = Delaunay(points_2d)
    except Exception as e:
        print(f"delaunay failed: {e}")
        return
    h, w = int(tex_rgb.shape[0]), int(tex_rgb.shape[1])
    simplices = _simplices_with_2d_and_3d_edge_limits(
        S,
        points_2d,
        tri,
        w,
        h,
        max_triangle_edge_frac,
        max_triangle_edge_3d_frac,
    )
    if simplices.size == 0:
        print(f"Mesh: no triangles after 2D/3D filters; skip {os.path.basename(output_name)}.")
        return

    if (
        mesh_drop_cross_class_triangles
        and point_labels is not None
        and point_labels.shape[0] == p
    ):
        n0 = int(simplices.shape[0])
        simplices = _filter_simplices_cross_class(simplices, point_labels)
        if simplices.size == 0:
            print(
                f"Mesh: no triangles after cross-class filter; skip {os.path.basename(output_name)}."
            )
            return
        if simplices.shape[0] < n0:
            print(
                f"Mesh: cross-class filter kept {simplices.shape[0]}/{n0} triangles "
                "(removed faces spanning two YOLO classes)."
            )

    mesh = o3d.geometry.TriangleMesh()
    vertices_np = np.asarray(S.T, dtype=np.float64)
    triangles_np = np.asarray(simplices, dtype=np.int32)
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(triangles_np)

    uvs = np.zeros((len(simplices), 3, 2), dtype=np.float64)
    for i, triangle in enumerate(simplices):
        for j, vertex in enumerate(triangle):
            uvs[i, j, 0] = points_2d[vertex, 0] / w
            uvs[i, j, 1] = 1.0 - (points_2d[vertex, 1] / h)

    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs.reshape(-1, 2))

    mesh.compute_vertex_normals()

    if mesh_keep_largest_component and len(mesh.triangles) > 0:
        mesh = _mesh_keep_largest_connected_component(mesh)

    img = np.ascontiguousarray(np.flipud(tex_rgb))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    image_rgb_o3d = o3d.geometry.Image(img)
    mesh.textures = [image_rgb_o3d]
    mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))

    if show:
        o3d.visualization.draw_geometries([mesh])

    if save_obj:
        obj_filename = f"{output_name}.obj"
        o3d.io.write_triangle_mesh(obj_filename, mesh, write_triangle_uvs=True)
        print("saved mesh")
        print(f"obj: {obj_filename}")


def save_textured_meshes_per_class(
    S_euclid: np.ndarray,
    points_2d: np.ndarray,
    image_rgb: np.ndarray,
    point_labels: np.ndarray,
    out_dir: str,
    *,
    seg_map_hw=None,
    overlay_alpha=None,
    max_triangle_edge_frac: Optional[float] = None,
    mesh_cluster_eps_frac: Optional[float] = None,
    max_triangle_edge_3d_frac: Optional[float] = None,
    mesh_keep_largest_component: bool = False,
    min_points_mesh: int = 3,
):
    point_labels = np.asarray(point_labels).astype(np.int32).reshape(-1)
    p = S_euclid.shape[1]
    if points_2d.shape[0] != p or point_labels.shape[0] != p:
        raise ValueError("S columns, points_2d rows, and point_labels must match")

    uniq = sorted(set(point_labels.tolist()))
    for lid in uniq:
        if lid <= 0:
            continue
        idx = np.where(point_labels == lid)[0]
        n = int(idx.size)
        if n < min_points_mesh:
            print(
                f"Per-class mesh: map_id={lid} has only {n} points (need ≥{min_points_mesh}); skip."
            )
            continue
        S_sub = S_euclid[:, idx]
        p2d_sub = points_2d[idx]
        base = os.path.join(out_dir, f"mesh_reconstruction_class_{lid}")
        try:
            print(
                f"Per-class mesh: map_id={lid} ({n} points) -> {os.path.basename(base)}.obj"
            )
            save_textured_mesh(
                S_sub,
                p2d_sub,
                image_rgb,
                output_name=base,
                show=False,
                save_obj=True,
                seg_map_hw=seg_map_hw,
                overlay_alpha=overlay_alpha,
                max_triangle_edge_frac=max_triangle_edge_frac,
                mesh_cluster_eps_frac=mesh_cluster_eps_frac,
                max_triangle_edge_3d_frac=max_triangle_edge_3d_frac,
                mesh_keep_largest_component=mesh_keep_largest_component,
            )
        except Exception as ex:
            print(f"per-class mesh failed: {ex}")


def main():
    from reconstruction_pipeline import run_metric_reconstruction

    # quick test on sample datasets
    # for data_dir in ["media/hotel", "media/poor", "media/good"]:
    #     S_eucl = run_metric_reconstruction(os.path.join(data_dir, "tracks.txt"))
    #     W = normalize_marix(os.path.join(data_dir, "tracks.txt"), normalize=False)
    #     image_path = os.path.join(data_dir, "frame000000.jpg")
    #     points_2d = np.column_stack((W[0, :], W[1, :]))
    #     save_textured_mesh(S_eucl, points_2d, mpimg.imread(image_path), output_name=os.path.join(data_dir, "3dReconstruction"))

    media_dir = "media/good"
    S_eucl = run_metric_reconstruction(os.path.join(media_dir, "tracks.txt"))
    # point_labels = load_point_labels_if_present(media_dir)  # same column order as tracks.txt when I add a model
    W = normalize_marix(os.path.join(media_dir, "tracks.txt"), normalize=False, plot_first_frame=False)
    image_path = os.path.join(media_dir, "frame000000.jpg")
    points_2d = np.column_stack((W[0, :], W[1, :]))
    save_textured_mesh(S_eucl, points_2d, mpimg.imread(image_path), output_name=os.path.join(media_dir, "3dReconstruction"))


if __name__ == "__main__":
    main()
