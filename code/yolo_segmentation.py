from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

try:
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset
except ImportError:
    YOLO = None  # type: ignore[assignment, misc]
    attempt_download_asset = None  # type: ignore[assignment, misc]


def resolve_seg_weights(weights: str) -> str:
    w = str(weights).strip().strip("'\"")
    p = Path(os.path.expanduser(w))
    if p.is_file() and p.stat().st_size > 10_000:
        return str(p.resolve())

    if attempt_download_asset is None:
        raise ImportError("Need ultralytics: pip install ultralytics")

    out = attempt_download_asset(p.name)
    out_path = Path(out)
    if out_path.is_file() and out_path.stat().st_size > 10_000:
        return str(out_path.resolve())

    raise FileNotFoundError(
        f"No weights file for {weights!r} (not on disk and ultralytics did not download it).\n"
        f"Fix: use an auto-download name (try yolo11n-seg.pt), "
        f"or pip install -U ultralytics, or pass the full path to a .pt you downloaded."
    )


def _composite_class_map_from_result(r: Any, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint16)
    if r.boxes is None or len(r.boxes) == 0:
        return out
    if r.masks is None:
        return out

    cls = r.boxes.cls.cpu().numpy().astype(np.int32)
    confs = r.boxes.conf.cpu().numpy()
    order = np.argsort(confs)

    md = r.masks.data.cpu().numpy()
    for idx in order:
        m = md[idx]
        if m.ndim != 2:
            continue
        mh, mw = m.shape[:2]
        if mh != h or mw != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = m > 0.5
        cid = int(cls[idx]) + 1
        out[mask] = cid
    return out


def segment_bgr_frames(
    weights: str,
    bgr_frames: Sequence[np.ndarray],
    *,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
) -> Tuple[np.ndarray, Dict[int, str]]:
    if YOLO is None:
        raise ImportError("Need ultralytics: pip install ultralytics")

    if not bgr_frames:
        raise ValueError("no frames")
    model = YOLO(resolve_seg_weights(weights))
    h0, w0 = bgr_frames[0].shape[:2]
    maps = np.zeros((len(bgr_frames), h0, w0), dtype=np.uint16)

    for t, frame in enumerate(bgr_frames):
        if frame.shape[:2] != (h0, w0):
            raise ValueError("all frames must have the same size")
        results = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        maps[t] = _composite_class_map_from_result(results[0], h0, w0)

    names = {int(k): str(v) for k, v in model.names.items()}
    return maps, names


def segment_video_file(
    weights: str,
    video_path: str,
    num_frames: int,
    frame_size: Tuple[int, int],
    *,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
) -> Tuple[np.ndarray, Dict[int, str]]:
    if YOLO is None:
        raise ImportError("Need ultralytics: pip install ultralytics")

    h0, w0 = frame_size
    model = YOLO(resolve_seg_weights(weights))
    maps = np.zeros((num_frames, h0, w0), dtype=np.uint16)
    cap = cv2.VideoCapture(video_path)
    for t in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"video ended early at frame {t}, expected {num_frames}")
        if frame.shape[:2] != (h0, w0):
            raise ValueError(f"frame {t} size {frame.shape[:2]} != {(h0, w0)}")
        results = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        maps[t] = _composite_class_map_from_result(results[0], h0, w0)
    cap.release()
    names = {int(k): str(v) for k, v in model.names.items()}
    return maps, names


def save_seg_bundle(
    out_dir: str,
    maps: np.ndarray,
    names: Dict[int, str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "seg_class_maps.npz"), maps=maps)
    with open(os.path.join(out_dir, "seg_class_names.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in names.items()}, f, indent=2)
    print(f"saved seg maps: {out_dir}/seg_class_maps.npz")


def run_yolo_seg_on_video(
    video_path: str,
    weights: str,
    out_dir: str,
    *,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
) -> Tuple[np.ndarray, Dict[int, str]]:
    cap = cv2.VideoCapture(video_path)
    bgr: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bgr.append(frame)
    cap.release()
    maps, names = segment_bgr_frames(weights, bgr, device=device, imgsz=imgsz, conf=conf)
    save_seg_bundle(out_dir, maps, names)
    return maps, names


def label_tracks_from_seg_maps_xy(
    seg_maps: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
) -> np.ndarray:
    T, H, W = seg_maps.shape
    n_points, n_frames = track_x.shape
    if track_y.shape != (n_points, n_frames):
        raise ValueError("track_x and track_y must match")
    out = np.zeros(n_points, dtype=np.int32)
    tf = min(T, n_frames)
    for i in range(n_points):
        votes: List[int] = []
        for t in range(tf):
            x = track_x[i, t]
            y = track_y[i, t]
            if np.isnan(x) or np.isnan(y):
                continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W:
                c = int(seg_maps[t, yi, xi])
                if c > 0:
                    votes.append(c)
        if votes:
            out[i] = int(max(set(votes), key=votes.count))
    return out


def label_tracks_from_seg_maps(
    seg_maps: np.ndarray,
    refined_tracks: Dict[int, List],
    column_ids: List[int],
) -> np.ndarray:
    T = seg_maps.shape[0]
    n = len(column_ids)
    track_x = np.full((n, T), np.nan, dtype=np.float64)
    track_y = np.full((n, T), np.nan, dtype=np.float64)
    for i, tid in enumerate(column_ids):
        tr = refined_tracks[tid]
        for t in range(min(T, len(tr))):
            p = tr[t]
            if p is None:
                continue
            track_x[i, t] = p[0]
            track_y[i, t] = p[1]
    return label_tracks_from_seg_maps_xy(seg_maps, track_x, track_y)


def _bgr_palette(max_id: int) -> np.ndarray:
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    palette[0] = (40, 40, 40)
    phi = 0.618033988749895
    for i in range(1, max_id + 1):
        h = float(((i * phi) % 1.0) * 179.0)
        s = float(155 + (i * 13) % 90)
        v = float(205 + (i * 11) % 50)
        hsv = np.uint8([[[h, s, v]]])
        palette[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return palette


def colorize_seg_map(seg_hw: np.ndarray) -> np.ndarray:
    u = np.clip(seg_hw.astype(np.int64), 0, None)
    mid = int(u.max()) if u.size else 0
    pal = _bgr_palette(mid)
    return pal[u]


def bgr_colors_for_point_labels(labels: np.ndarray) -> np.ndarray:
    u = np.asarray(labels).astype(np.int64).reshape(-1)
    mid = int(u.max()) if u.size else 0
    pal = _bgr_palette(mid)
    u = np.clip(u, 0, mid)
    return pal[u]


def rgb01_color_table(max_label_id: int) -> np.ndarray:
    bgr = _bgr_palette(max_label_id)
    return (bgr[:, ::-1].astype(np.float64) / 255.0).clip(0.0, 1.0)


# Class tint strength when compositing on the first frame (like cv2.addWeighted seg overlay).
overlay_alpha_default = 0.42

# Lighter default for show-frame: more photo, softer “opacity on top” preview.
frame_overlay_alpha = 0.30


def rgb01_sample_image(image_rgb: np.ndarray, x: float, y: float) -> np.ndarray:
    if image_rgb.ndim == 2:
        h, w = image_rgb.shape
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        v = float(image_rgb[yi, xi])
        if image_rgb.dtype == np.uint8:
            s = v / 255.0
        elif v <= 1.0:
            s = float(np.clip(v, 0.0, 1.0))
        else:
            s = v / 255.0
        return np.array([s, s, s], dtype=np.float64)

    h, w = image_rgb.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    pix = image_rgb[yi, xi, :3].astype(np.float64)
    if image_rgb.dtype == np.uint8 or np.nanmax(pix) > 1.0:
        pix = pix / 255.0
    return np.clip(pix, 0.0, 1.0)


def rgb01_seg_overlay_on_image(
    image_rgb: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    labels: np.ndarray,
    *,
    overlay_alpha: float = overlay_alpha_default,
) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    n = labels.shape[0]
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    if xs.shape[0] != n or ys.shape[0] != n:
        raise ValueError("xs, ys, labels length mismatch")

    mid = int(labels.max()) if labels.size else 0
    table = rgb01_color_table(mid)
    out = np.zeros((n, 3), dtype=np.float64)
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    for i in range(n):
        base = rgb01_sample_image(image_rgb, xs[i], ys[i])
        lid = int(np.clip(labels[i], 0, mid))
        if lid == 0:
            out[i] = base
        else:
            cls = table[lid]
            out[i] = (1.0 - a) * base + a * cls
    return out


def _image_rgb01_hw3(image_rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(image_rgb)
    if x.ndim == 2:
        h, w = x.shape
        g = x.astype(np.float64)
        if x.dtype == np.uint8 or np.nanmax(g) > 1.0:
            g = g / 255.0
        g = np.clip(g, 0.0, 1.0)
        return np.stack([g, g, g], axis=-1)
    if x.shape[2] >= 4:
        x = x[:, :, :3]
    img = x.astype(np.float64)
    if x.dtype == np.uint8 or np.nanmax(img) > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def rgb_uint8_texture_with_dense_seg_overlay(
    image_rgb: np.ndarray,
    seg_hw: np.ndarray,
    *,
    overlay_alpha: float = overlay_alpha_default,
) -> np.ndarray:
    img = _image_rgb01_hw3(image_rgb)
    h, w = img.shape[:2]
    seg = np.asarray(seg_hw)
    if seg.ndim != 2:
        raise ValueError("seg_hw must be HxW")
    if seg.shape[0] != h or seg.shape[1] != w:
        seg = cv2.resize(
            seg.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        seg = np.round(seg).astype(np.uint16)

    color_bgr = colorize_seg_map(seg)
    color_rgb = color_bgr[:, :, ::-1].astype(np.float64) / 255.0
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    blend = (1.0 - a) * img + a * color_rgb
    return (np.clip(blend, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_seg_preview_video(
    npz_path: str,
    out_mp4: str,
    fps: float = 30.0,
    *,
    overlay_bgr_video: Optional[str] = None,
    alpha: float = 0.55,
) -> None:
    data = np.load(npz_path)
    maps = data["maps"]
    if maps.ndim != 3:
        raise ValueError("expected maps with shape (T, H, W)")

    T, h, w = maps.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_mp4}")

    cap = None
    if overlay_bgr_video:
        cap = cv2.VideoCapture(overlay_bgr_video)
        if not cap.isOpened():
            raise FileNotFoundError(overlay_bgr_video)

    for t in range(T):
        color = colorize_seg_map(maps[t])
        if cap is not None:
            ret, base = cap.read()
            if not ret or base.shape[:2] != (h, w):
                raise ValueError("overlay video must match map size and length")
            blend = cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0)
            writer.write(blend)
        else:
            writer.write(color)

    if cap is not None:
        cap.release()
    writer.release()
    print(f"saved {out_mp4} ({T} frames)")


def visualize_seg_map_frame(
    npz_path: str,
    frame_index: int,
    *,
    video_path: Optional[str] = None,
    overlay_alpha: Optional[float] = None,
) -> np.ndarray:
    if video_path and overlay_alpha is None:
        overlay_alpha = frame_overlay_alpha
    data = np.load(npz_path)
    maps = data["maps"]
    if maps.ndim != 3:
        raise ValueError("expected maps with shape (T, H, W)")
    t = int(frame_index)
    if t < 0 or t >= maps.shape[0]:
        raise IndexError(f"frame_index {t} out of range [0, {maps.shape[0]})")
    seg = maps[t]
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, bgr = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"could not read frame {t} from {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb_uint8_texture_with_dense_seg_overlay(
            rgb, seg, overlay_alpha=overlay_alpha
        )
    color_bgr = colorize_seg_map(seg)
    return color_bgr[:, :, ::-1]


def visualize_seg_map_frame_with_legend(
    npz_path: str,
    frame_index: int,
    *,
    video_path: Optional[str] = None,
    overlay_alpha: Optional[float] = None,
    out_png: Optional[str] = None,
) -> None:
    from reconstruction_pipeline import class_legend_patches, load_seg_class_names_if_present

    rgb = visualize_seg_map_frame(
        npz_path, frame_index, video_path=video_path, overlay_alpha=overlay_alpha
    )
    data = np.load(npz_path)
    maps = data["maps"]
    t = int(frame_index)
    seg = maps[t]
    uniq = sorted(set(np.unique(seg).astype(int).tolist()))
    out_dir = os.path.dirname(os.path.abspath(npz_path))
    names = load_seg_class_names_if_present(out_dir)
    handles = class_legend_patches(uniq, names)

    fig = plt.figure(figsize=(11, max(4.0, 0.35 * len(uniq) + 2.0)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1.0], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(rgb)
    ax0.set_axis_off()
    ax0.set_title(f"YOLO seg — frame index {t}", fontsize=11)
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.legend(
        handles=handles,
        loc="center",
        fontsize=9,
        frameon=True,
        title="Classes",
    )
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_png}")
    else:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "view":
        # python yolo_segmentation.py view maps.npz out.mp4 [fps] [--overlay video.mp4]
        rest = sys.argv[2:]
        overlay_path = None
        if "--overlay" in rest:
            i = rest.index("--overlay")
            overlay_path = rest[i + 1]
            rest = rest[:i] + rest[i + 2 :]
        if len(rest) < 2:
            print(
                "usage: python yolo_segmentation.py view <seg_class_maps.npz> <out.mp4> [fps] [--overlay source.mp4]"
            )
            sys.exit(1)
        npz_path, out_mp4 = rest[0], rest[1]
        fps = float(rest[2]) if len(rest) > 2 else 30.0
        save_seg_preview_video(npz_path, out_mp4, fps=fps, overlay_bgr_video=overlay_path)
    elif len(sys.argv) >= 2 and sys.argv[1] == "show-frame":
        # python yolo_segmentation.py show-frame <npz> <idx> [--video clip.mov] [--out x.png] [--opacity 0.3]
        args = sys.argv[2:]
        if len(args) < 2:
            print(
                "usage: python yolo_segmentation.py show-frame <seg_class_maps.npz> <frame_index> "
                "[--video source.mov] [--out out.png] [--opacity 0.3] [--legend]"
            )
            print("  frame_index is 0-based (0 = first frame in the npz).")
            print(
                "  --opacity / --alpha: class layer strength on top of the frame (0=photo only, 1=flat classes); "
                f"default with --video is {frame_overlay_alpha} (light overlay)."
            )
            print(
                "  --legend: side panel with class colors + names (needs seg_class_names.json beside the npz)."
            )
            sys.exit(1)
        npz_path, frame_s = args[0], args[1]
        overlay_video = None
        out_png = None
        alpha: Optional[float] = None
        with_legend = False
        i = 2
        while i < len(args):
            if args[i] == "--video" and i + 1 < len(args):
                overlay_video = args[i + 1]
                i += 2
            elif args[i] == "--out" and i + 1 < len(args):
                out_png = args[i + 1]
                i += 2
            elif args[i] in ("--alpha", "--opacity") and i + 1 < len(args):
                alpha = float(args[i + 1])
                i += 2
            elif args[i] == "--legend":
                with_legend = True
                i += 1
            else:
                print(f"bad arg: {args[i]!r}")
                sys.exit(1)
        if with_legend:
            visualize_seg_map_frame_with_legend(
                npz_path,
                int(frame_s),
                video_path=overlay_video,
                overlay_alpha=alpha,
                out_png=out_png,
            )
        else:
            rgb = visualize_seg_map_frame(
                npz_path, int(frame_s), video_path=overlay_video, overlay_alpha=alpha
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if out_png:
                cv2.imwrite(out_png, bgr)
                print(f"saved {out_png}")
            else:
                cv2.imshow(f"YOLO seg frame {frame_s}", bgr)
                print("press any key")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    elif len(sys.argv) >= 4:
        run_yolo_seg_on_video(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("run:")
        print("  python yolo_segmentation.py <video_path> <weights.pt> <out_dir>")
        print("view:")
        print("  python yolo_segmentation.py view <seg_class_maps.npz> <out.mp4> [fps] [--overlay video.mp4]")
        print("frame:")
        print(
            "  python yolo_segmentation.py show-frame <seg_class_maps.npz> <frame_index> "
            "[--video source.mov] [--out seg_frame.png] [--opacity 0.3] [--legend]"
        )
        sys.exit(1)
