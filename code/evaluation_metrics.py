from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import Delaunay

from q1_affine_factorization_method import normalize_marix
from q2_metric_upgrade import a, b, c
from q3_texturing import (
    _filter_simplices_cross_class,
    _simplices_with_2d_and_3d_edge_limits,
    mesh_mask_inlier_points_3d,
)
from reconstruction_pipeline import (
    effective_mesh_max_edge_frac,
    effective_mesh_max_edge_3d_frac,
    load_point_labels_if_present,
    load_tracks_xy_nan,
    points_2d_for_mesh_uv,
    rebuild_point_labels_from_seg_bundle,
)

code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

metrics_json = "metrics.json"
metrics_txt = "metrics.txt"


def _tracks_to_rows(track_x: np.ndarray, track_y: np.ndarray) -> np.ndarray:
    n, F = track_x.shape
    W = np.zeros((2 * F, n), dtype=np.float64)
    for t in range(F):
        W[2 * t, :] = track_x[:, t]
        W[2 * t + 1, :] = track_y[:, t]
    return W


def _fill_nan_rows(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    out = W.copy()
    if np.any(np.isnan(out)):
        for i in range(out.shape[0]):
            row = out[i, :]
            out[i, :] = np.where(np.isnan(row), np.nanmean(row), row)
    return out


def _demean_rows(W: np.ndarray) -> np.ndarray:
    W = W.copy()
    W[0::2, :] -= np.mean(W[0::2, :], axis=1, keepdims=True)
    W[1::2, :] -= np.mean(W[1::2, :], axis=1, keepdims=True)
    return W


def _reproj_stats(
    tracks_txt: str,
    R_eucl: np.ndarray,
    S_eucl: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    track_x, track_y = load_tracks_xy_nan(tracks_txt)
    W_raw = _tracks_to_rows(track_x, track_y)
    W_fill = _fill_nan_rows(W_raw)
    W_norm = _demean_rows(W_fill)
    W_hat = R_eucl @ S_eucl
    F, n_pts = track_x.shape[1], track_x.shape[0]
    assert W_hat.shape == W_norm.shape

    err_norm_list: List[float] = []
    err_px_list: List[float] = []

    for t in range(F):
        mx = float(np.mean(W_fill[2 * t, :]))
        my = float(np.mean(W_fill[2 * t + 1, :]))
        for i in range(n_pts):
            if not np.isfinite(W_raw[2 * t, i]) or not np.isfinite(W_raw[2 * t + 1, i]):
                continue
            dx_n = W_hat[2 * t, i] - W_norm[2 * t, i]
            dy_n = W_hat[2 * t + 1, i] - W_norm[2 * t + 1, i]
            err_norm_list.append(float(math.hypot(dx_n, dy_n)))

            x_hat_px = W_hat[2 * t, i] + mx
            y_hat_px = W_hat[2 * t + 1, i] + my
            err_px_list.append(
                float(
                    math.hypot(x_hat_px - track_x[i, t], y_hat_px - track_y[i, t])
                )
            )

    en = np.asarray(err_norm_list, dtype=np.float64)
    ep = np.asarray(err_px_list, dtype=np.float64)

    def _mmd(a: np.ndarray) -> Dict[str, Optional[float]]:
        if a.size == 0:
            return {"mean": None, "std": None, "median": None}
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
        }

    out: Dict[str, Any] = {
        "n_obs": int(len(en)),
        "in_W": _mmd(en),
        "pixels": _mmd(ep),
    }
    return out, en, ep


def _track_lengths(track_x: np.ndarray, track_y: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(track_x) & np.isfinite(track_y)
    L = np.sum(valid, axis=1).astype(np.int32)
    if L.size == 0:
        return {}
    return {
        "mean": float(np.mean(L)),
        "median": float(np.median(L)),
        "min": int(L.min()),
        "max": int(L.max()),
        "std": float(np.std(L.astype(np.float64))),
    }


def _spread(S: np.ndarray) -> Dict[str, Any]:
    S = np.asarray(S, dtype=np.float64)
    if S.shape[1] < 4:
        return {"note": "too few points"}
    X = S.T
    ctr = np.mean(X, axis=0)
    d = np.linalg.norm(X - ctr, axis=1)
    p95 = float(np.percentile(d, 95))
    outlier = d > p95
    _, _, vt = np.linalg.svd(X - ctr, full_matrices=False)
    n = vt[-1]
    plane_d = np.abs((X - ctr) @ n)
    return {
        "mean_radius": float(np.mean(d)),
        "std_radius": float(np.std(d)),
        "pct_past_p95_radius": float(100.0 * np.mean(outlier)),
        "mean_plane_dist": float(np.mean(plane_d)),
        "std_plane_dist": float(np.std(plane_d)),
    }


def _label_stability(
    seg_maps: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
    final_labels: np.ndarray,
) -> Dict[str, Any]:
    T, H, W = seg_maps.shape
    n_points, n_frames = track_x.shape
    tf = min(T, n_frames)
    final_labels = np.asarray(final_labels, dtype=np.int32).reshape(-1)
    if final_labels.shape[0] != n_points:
        return {"ok": False, "why": "label length mismatch"}

    consistencies: List[float] = []
    entropies: List[float] = []
    by_id: Dict[str, List[float]] = {}

    for i in range(n_points):
        fl = int(final_labels[i])
        samples: List[int] = []
        for t in range(tf):
            x, y = track_x[i, t], track_y[i, t]
            if np.isnan(x) or np.isnan(y):
                continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W:
                samples.append(int(seg_maps[t, yi, xi]))
        if not samples:
            continue
        consistencies.append(sum(1 for s in samples if s == fl) / len(samples))

        counts: Dict[int, int] = {}
        for s in samples:
            counts[s] = counts.get(s, 0) + 1
        tot = len(samples)
        h = 0.0
        for c in counts.values():
            p = c / tot
            h -= p * math.log(p + 1e-15)
        entropies.append(h)
        by_id.setdefault(str(fl), []).append(consistencies[-1])

    if not consistencies:
        return {
            "ok": True,
            "mean_agree": None,
            "mean_H": None,
            "by_map_id": {},
        }

    return {
        "ok": True,
        "mean_agree": float(np.mean(consistencies)),
        "std_agree": float(np.std(consistencies)),
        "median_agree": float(np.median(consistencies)),
        "mean_H": float(np.mean(entropies)),
        "std_H": float(np.std(entropies)),
        "by_map_id": {k: float(np.mean(v)) for k, v in by_id.items()},
    }


def _fg_counts(labels: np.ndarray) -> Dict[str, Any]:
    fl = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = fl.size
    if n == 0:
        return {}
    nz = int(np.sum(fl > 0))
    return {"n": int(n), "n_fg": nz, "fg_frac": float(nz / n)}


def _n_cross_class_faces(simplices: np.ndarray, lab: np.ndarray) -> int:
    n = 0
    for sim in simplices:
        i, j, k = int(sim[0]), int(sim[1]), int(sim[2])
        a, b, c = int(lab[i]), int(lab[j]), int(lab[k])
        bad = False
        for u, v in ((a, b), (a, c), (b, c)):
            if u > 0 and v > 0 and u != v:
                bad = True
                break
        if bad:
            n += 1
    return n


def _mesh_counts(
    out_dir: str,
    S_eucl: np.ndarray,
    *,
    mesh_max_edge_frac: Optional[float],
    mesh_max_edge_3d_frac: Optional[float],
    mesh_uv_frame_start: int,
    mesh_uv_num_frames: Optional[int],
    mesh_drop_cross_class_triangles: bool,
    point_labels: Optional[np.ndarray],
) -> Dict[str, Any]:
    tracks_txt = os.path.join(out_dir, "tracks.txt")
    image_path = os.path.join(out_dir, "frame000000.jpg")
    if not os.path.isfile(tracks_txt) or not os.path.isfile(image_path):
        return {"ok": False, "why": "need tracks.txt and frame000000.jpg"}

    W = normalize_marix(
        tracks_txt, normalize=False, plot_first_frame=False, fill_nan=False
    )
    points_2d = points_2d_for_mesh_uv(
        W,
        frame_start=mesh_uv_frame_start,
        mesh_uv_num_frames=mesh_uv_num_frames,
    )
    img = mpimg.imread(image_path)
    h, w = int(img.shape[0]), int(img.shape[1])

    S = np.asarray(S_eucl, dtype=np.float64)
    p0 = int(S.shape[1])
    pl = (
        np.asarray(point_labels, dtype=np.int32).reshape(-1)
        if point_labels is not None
        else None
    )
    if pl is not None and pl.shape[0] != p0:
        pl = None

    edge = effective_mesh_max_edge_frac(mesh_max_edge_frac)
    edge3 = effective_mesh_max_edge_3d_frac(mesh_max_edge_3d_frac)

    mask = mesh_mask_inlier_points_3d(S)
    n_in = int(mask.sum())
    if n_in < 3:
        return {
            "ok": False,
            "why": "too few inliers",
            "n_pts_in": n_in,
            "n_pts": p0,
        }

    S_f = S[:, mask]
    p2d = np.asarray(points_2d, dtype=np.float64)[mask]
    pl_f = pl[mask] if pl is not None else None

    try:
        tri = Delaunay(p2d)
    except Exception as e:
        return {"ok": False, "why": f"delaunay: {e}"}

    n_del = int(tri.simplices.shape[0])
    simp = _simplices_with_2d_and_3d_edge_limits(S_f, p2d, tri, w, h, edge, edge3)
    n_filt = int(simp.shape[0])
    pct_drop_geom = float(100.0 * (1.0 - n_filt / n_del)) if n_del else None

    out: Dict[str, Any] = {
        "ok": True,
        "tris_raw": n_del,
        "tris_after_geom": n_filt,
        "pct_drop_geom": pct_drop_geom,
        "n_pts": p0,
        "n_pts_kept": n_in,
        "pct_pts_dropped": float(100.0 * (p0 - n_in) / p0) if p0 else None,
    }

    if (
        mesh_drop_cross_class_triangles
        and pl_f is not None
        and pl_f.shape[0] == p2d.shape[0]
        and simp.size
    ):
        n_bad = _n_cross_class_faces(simp, pl_f)
        simp2 = _filter_simplices_cross_class(simp, pl_f)
        n2 = int(simp2.shape[0])
        out["tris_before_class_cut"] = n_filt
        out["tris_cross_class"] = int(n_bad)
        out["frac_cross_class"] = float(n_bad / n_filt) if n_filt else None
        out["tris_after_class_cut"] = n2
        out["pct_drop_class"] = (
            float(100.0 * (1.0 - n2 / n_filt)) if n_filt else None
        )

    return out


def save_metrics(
    out_dir: str,
    *,
    mesh_max_edge_frac: Optional[float] = None,
    mesh_max_edge_3d_frac: Optional[float] = None,
    mesh_uv_frame_start: int = 0,
    mesh_uv_num_frames: Optional[int] = None,
    mesh_drop_cross_class_triangles: bool = True,
) -> Optional[str]:
    out_dir = os.path.abspath(out_dir)
    tracks_txt = os.path.join(out_dir, "tracks.txt")
    q2_path = os.path.join(out_dir, "q2_S_scaled.npy")
    if not os.path.isfile(tracks_txt):
        print(f"metrics: missing {tracks_txt}")
        return None
    if not os.path.isfile(q2_path):
        print(f"metrics: missing {q2_path}")
        return None

    A, R, S = a(tracks_txt, plot_first_frame=False)
    L = b(A)
    R_eucl, S_eucl = c(L, R, S)

    track_x, track_y = load_tracks_xy_nan(tracks_txt)
    reproj, _, _ = _reproj_stats(tracks_txt, R_eucl, S_eucl)
    reproj["track_lengths"] = _track_lengths(track_x, track_y)

    S_scaled = np.load(q2_path)
    reproj["spread_q2"] = _spread(S_scaled)
    reproj["spread_eucl"] = _spread(S_eucl)

    labels = load_point_labels_if_present(out_dir)
    if labels is None:
        labels = rebuild_point_labels_from_seg_bundle(out_dir)

    sem: Dict[str, Any] = {"have_labels": labels is not None}
    if labels is not None:
        sem["fg"] = _fg_counts(labels)
        npz_path = os.path.join(out_dir, "seg_class_maps.npz")
        if os.path.isfile(npz_path):
            seg_maps = np.load(npz_path)["maps"]
            sem["labels"] = _label_stability(seg_maps, track_x, track_y, labels)
        else:
            sem["labels"] = {"ok": False, "why": "no seg_class_maps.npz"}

    mesh = _mesh_counts(
        out_dir,
        S_eucl,
        mesh_max_edge_frac=mesh_max_edge_frac,
        mesh_max_edge_3d_frac=mesh_max_edge_3d_frac,
        mesh_uv_frame_start=mesh_uv_frame_start,
        mesh_uv_num_frames=mesh_uv_num_frames,
        mesh_drop_cross_class_triangles=mesh_drop_cross_class_triangles,
        point_labels=labels,
    )

    mean_agree = None
    if isinstance(sem.get("labels"), dict):
        mean_agree = sem["labels"].get("mean_agree")

    mean_px = None
    px = reproj.get("pixels")
    if isinstance(px, dict):
        mean_px = px.get("mean")

    score = None
    if mean_agree is not None and mean_px is not None and mean_px > 0:
        score = float(mean_agree * (1.0 / (mean_px + 1e-6)))

    report: Dict[str, Any] = {
        "run": out_dir,
        "reproj": reproj,
        "mesh": mesh,
        "sem": sem,
        "score": score,
    }

    json_path = os.path.join(out_dir, metrics_json)
    txt_path = os.path.join(out_dir, metrics_txt)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    _dump_txt(txt_path, report)

    print(f"saved {json_path}")
    print(f"saved {txt_path}")
    return json_path


def _dump_txt(path: str, report: Dict[str, Any]) -> None:
    lines = [
        "metrics",
        "=" * 40,
        "",
        "reprojection",
    ]
    r = report.get("reproj") or {}
    for k in ("n_obs", "in_W", "pixels", "track_lengths"):
        if k in r:
            lines.append(f"  {k}: {r[k]}")
    for k in ("spread_q2", "spread_eucl"):
        if k in r:
            lines.append(f"  {k}:")
            sub = r[k]
            if isinstance(sub, dict):
                for kk, vv in sub.items():
                    lines.append(f"    {kk}: {vv}")

    lines.extend(["", "mesh", ""])
    m = report.get("mesh") or {}
    for k, v in m.items():
        lines.append(f"  {k}: {v}")

    lines.extend(["", "sem", ""])
    s = report.get("sem") or {}
    for k, v in s.items():
        if k == "labels" and isinstance(v, dict):
            lines.append("  labels:")
            for kk, vv in v.items():
                if kk == "by_map_id" and isinstance(vv, dict):
                    lines.append(f"    {kk}:")
                    for a, b in sorted(vv.items())[:24]:
                        lines.append(f"      {a}: {b}")
                    if len(vv) > 24:
                        lines.append("      ...")
                else:
                    lines.append(f"    {kk}: {vv}")
        else:
            lines.append(f"  {k}: {v}")

    lines.extend(["", "score", f"  {report.get('score')}"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python evaluation_metrics.py <out_dir>")
        sys.exit(1)
    save_metrics(sys.argv[1])


if __name__ == "__main__":
    main()
