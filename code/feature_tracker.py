import gc
import cv2
import numpy as np
import os
import sys
from typing import Optional, Tuple
from scipy.optimize import least_squares
import scipy.io as sio

from reconstruction_pipeline import (
    save_point_labels,
    track_arrays_from_dict,
    write_tracks_txt,
)
from yolo_segmentation import (
    label_tracks_from_seg_maps,
    save_seg_bundle,
    segment_bgr_frames,
    segment_video_file,
)

# Playback speed for tracked_output.avi (lower FPS = longer gap between frames, easier to follow).
preview_fps = 15.0

# Faster Lucas–Kanade (slightly less robust than OpenCV defaults; big CPU win on long clips).
lk_fast = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
lk_accurate = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
lk_fastest = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.05)


def _gray_for_corner_detect(frame: np.ndarray, use_clahe: bool) -> np.ndarray:
    if not use_clahe:
        return frame
    g = np.asarray(frame)
    if g.ndim != 2:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def farthest_point_spread_subsample(
    pts: np.ndarray,
    k: int,
    min_sep: float,
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.array([]).reshape(0, 2)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    n = pts.shape[0]
    k = max(1, min(int(k), n))
    c = pts.mean(axis=0)
    chosen = [int(np.argmax(np.sum((pts - c) ** 2, axis=1)))]
    mind = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(k - 1):
        mind = np.minimum(mind, np.linalg.norm(pts - pts[chosen[-1]], axis=1))
        mind[np.array(chosen)] = -1.0
        nxt = int(np.argmax(mind))
        if mind[nxt] < 0:
            break
        chosen.append(nxt)
    sel = pts[np.array(chosen, dtype=int)]
    if min_sep > 0 and sel.shape[0] > 1:
        keep = [0]
        for j in range(1, sel.shape[0]):
            d = np.linalg.norm(sel[j] - sel[np.array(keep)], axis=1)
            if np.all(d >= min_sep):
                keep.append(j)
        sel = sel[np.array(keep, dtype=int)]
    return sel.astype(np.float32)


def load_video_frames(video_path, max_frames=None, scale: float = 1.0):
    cap = cv2.VideoCapture(video_path)
    frames = []
    first_gray_full = None
    while max_frames is None or len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if first_gray_full is None:
            first_gray_full = gray.copy()
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()
    return frames, first_gray_full


def scale_track_coordinates(refined_tracks, sx: float, sy: float):
    out = {}
    for k, lst in refined_tracks.items():
        out[k] = [
            None if p is None else (float(p[0]) * sx, float(p[1]) * sy) for p in lst
        ]
    return out


def load_video_bgr_and_gray(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    bgr, gray = [], []
    while max_frames is None or len(gray) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        bgr.append(frame)
        gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return bgr, gray

def center_roi_mask(h: int, w: int, roi_frac: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    rw, rh = int(w * roi_frac), int(h * roi_frac)
    x0 = max(0, (w - rw) // 2)
    y0 = max(0, (h - rh) // 2)
    mask[y0 : y0 + rh, x0 : x0 + rw] = 255
    return mask


def read_first_frame_bgr(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def read_first_k_bgr_frames(video_path: str, k: int):
    k = max(0, int(k))
    if k == 0:
        return []
    cap = cv2.VideoCapture(video_path)
    out = []
    for _ in range(k):
        ret, frame = cap.read()
        if not ret:
            break
        out.append(frame)
    cap.release()
    return out


def detect_features(
    frame,
    max_corners=1000,
    quality_level=0.01,
    min_distance=10,
    detection_mask=None,
    feature_grid: Optional[Tuple[int, int]] = None,
    use_clahe: bool = False,
):
    work = _gray_for_corner_detect(frame, use_clahe)
    h, w = work.shape[:2]
    base_mask = None
    if detection_mask is not None:
        base_mask = np.asarray(detection_mask, dtype=np.uint8)
        if base_mask.shape[:2] != (h, w):
            base_mask = cv2.resize(base_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if feature_grid is None:
        gr, gc = 1, 1
    else:
        gr, gc = max(1, int(feature_grid[0])), max(1, int(feature_grid[1]))

    if gr <= 1 and gc <= 1:
        corners = cv2.goodFeaturesToTrack(
            work,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            mask=base_mask,
        )
        if corners is not None:
            return np.squeeze(corners)
        return np.array([])

    per_cell = max(1, int(max_corners) // (gr * gc))
    cell_h = max(1, h // gr)
    cell_w = max(1, w // gc)
    chunks = []
    for i in range(gr):
        for j in range(gc):
            y0 = i * cell_h
            x0 = j * cell_w
            y1 = h if i == gr - 1 else min(h, (i + 1) * cell_h)
            x1 = w if j == gc - 1 else min(w, (j + 1) * cell_w)
            rect = np.zeros((h, w), dtype=np.uint8)
            rect[y0:y1, x0:x1] = 255
            if base_mask is not None:
                m = cv2.bitwise_and(rect, base_mask)
            else:
                m = rect
            if not np.any(m):
                continue
            pts = cv2.goodFeaturesToTrack(
                work,
                maxCorners=per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance,
                mask=m,
            )
            if pts is not None and len(pts) > 0:
                chunks.append(np.squeeze(pts).reshape(-1, 2))
    if not chunks:
        return np.array([])
    stacked = np.vstack(chunks)
    return stacked


def detect_new_features(
    frame,
    existing_features,
    max_corners=50,
    quality_level=0.01,
    min_distance=10,
    mask_radius=10,
    detection_mask=None,
    feature_grid: Optional[Tuple[int, int]] = None,
    use_clahe: bool = False,
):
    work = _gray_for_corner_detect(frame, use_clahe)
    h, w = work.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    if detection_mask is not None:
        dm = np.asarray(detection_mask, dtype=np.uint8)
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_and(mask, dm)

    valid_feats = np.array([(round(f[0]), round(f[1])) for f in existing_features if f is not None], dtype=np.int32)
    if len(valid_feats) > 0:
        yy, xx = np.ogrid[:h, :w]
        yy = yy[:, :, np.newaxis]
        xx = xx[:, :, np.newaxis]
        fx = valid_feats[:, 0].reshape(1, 1, -1)
        fy = valid_feats[:, 1].reshape(1, 1, -1)
        dist_sq = (xx - fx) ** 2 + (yy - fy) ** 2
        inside_any = np.any(dist_sq <= mask_radius**2, axis=2)
        mask[inside_any] = 0

    if feature_grid is None:
        gr, gc = 1, 1
    else:
        gr, gc = max(1, int(feature_grid[0])), max(1, int(feature_grid[1]))

    if gr <= 1 and gc <= 1:
        corners = cv2.goodFeaturesToTrack(
            work,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            mask=mask,
        )
        if corners is not None:
            return np.squeeze(corners)
        return np.array([])

    per_cell = max(1, int(max_corners) // (gr * gc))
    cell_h = max(1, h // gr)
    cell_w = max(1, w // gc)
    chunks = []
    for i in range(gr):
        for j in range(gc):
            y0 = i * cell_h
            x0 = j * cell_w
            y1 = h if i == gr - 1 else min(h, (i + 1) * cell_h)
            x1 = w if j == gc - 1 else min(w, (j + 1) * cell_w)
            rect = np.zeros((h, w), dtype=np.uint8)
            rect[y0:y1, x0:x1] = 255
            m = cv2.bitwise_and(mask, rect)
            if not np.any(m):
                continue
            pts = cv2.goodFeaturesToTrack(
                work,
                maxCorners=per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance,
                mask=m,
            )
            if pts is not None and len(pts) > 0:
                chunks.append(np.squeeze(pts).reshape(-1, 2))
    if not chunks:
        return np.array([])
    return np.vstack(chunks)

def track_features_with_redetection(
    frames,
    initial_features,
    redetect_interval=10,
    max_redetect=30,
    redetect_mask=None,
    feature_quality=0.02,
    min_feature_distance=12,
    *,
    lk_preset: str = "fast",
    lk_max_jump_frac: Optional[float] = 0.12,
    feature_grid: Optional[Tuple[int, int]] = None,
    feature_use_clahe: bool = False,
):
    num_frames = len(frames)
    tracks = {i: [] for i in range(len(initial_features))}
    
    for i in range(len(initial_features)):
        tracks[i].append(tuple(initial_features[i]))
    
    prev_frame = frames[0]
    prev_pts = initial_features.astype(np.float32).reshape(-1, 1, 2)
    
    active_features = np.ones(len(initial_features), dtype=bool)

    preset = str(lk_preset).lower()
    if preset == "accurate":
        lk_win = (21, 21)
        lk_level = 3
        lk_crit = lk_accurate
    elif preset == "fastest":
        lk_win = (11, 11)
        lk_level = 1
        lk_crit = lk_fastest
    else:
        lk_win = (15, 15)
        lk_level = 2
        lk_crit = lk_fast

    for i in range(1, num_frames):
        curr_frame = frames[i]

        if np.sum(active_features) > 0:
            active_prev_pts = prev_pts[active_features].reshape(-1, 1, 2)

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame,
                curr_frame,
                active_prev_pts,
                None,
                winSize=lk_win,
                maxLevel=lk_level,
                criteria=lk_crit,
            )

            active_indices = np.where(active_features)[0]
            status_flat = status.flatten()
            success_mask = status_flat == 1
            if lk_max_jump_frac is not None and float(lk_max_jump_frac) > 0:
                h, w = int(curr_frame.shape[0]), int(curr_frame.shape[1])
                max_jump = float(lk_max_jump_frac) * float(max(h, w))
                prev_flat = active_prev_pts[:, 0, :].astype(np.float64)
                next_flat = next_pts[:, 0, :].astype(np.float64)
                d = np.linalg.norm(next_flat - prev_flat, axis=1)
                success_mask = success_mask & (d <= max_jump)
            prev_pts[active_indices] = np.where(
                success_mask[:, np.newaxis, np.newaxis], next_pts, prev_pts[active_indices]
            )
            active_features[active_indices[~success_mask]] = False
            next_pts_flat = next_pts[:, 0]
            for j in range(len(active_features)):
                tracks[j].append(None)
            for idx, j in enumerate(active_indices):
                if success_mask[idx]:
                    t = next_pts_flat[idx]
                    tracks[j][-1] = (float(t[0]), float(t[1]))
        else:
            for j in range(len(active_features)):
                tracks[j].append(None)
        
        if i % redetect_interval == 0:
            current_features = prev_pts[active_features, 0]
            new_features = detect_new_features(
                curr_frame,
                current_features,
                max_corners=max_redetect,
                quality_level=float(feature_quality),
                min_distance=int(min_feature_distance),
                detection_mask=redetect_mask,
                feature_grid=feature_grid,
                use_clahe=feature_use_clahe,
            )
            
            if len(new_features) > 0:
                if len(new_features.shape) == 1:
                    new_features = np.array([new_features])
                new_pts = new_features.astype(np.float32).reshape(-1, 1, 2)
                prev_pts = np.vstack((prev_pts, new_pts))
                active_features = np.append(active_features, np.ones(len(new_features), dtype=bool))
                for new_feat in new_features:
                    new_idx = len(tracks)
                    tracks[new_idx] = [None] * i
                    tracks[new_idx].append(tuple(new_feat))
        
        prev_frame = curr_frame
    
    return tracks

def track_features(frames, initial_features):
    num_frames = len(frames)
    tracks = {i: [] for i in range(len(initial_features))}
    
    for i in range(len(initial_features)):
        tracks[i].append(tuple(initial_features[i]))
    prev_frame = frames[0]
    prev_pts = initial_features.astype(np.float32).reshape(-1, 1, 2)
    
    for i in range(1, num_frames):
        curr_frame = frames[i]
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_pts, None)
        success = status.flatten() == 1
        next_pts_flat = next_pts[:, 0]
        for idx in range(len(status)):
            tracks[idx].append(tuple(next_pts_flat[idx]) if success[idx] else None)
        prev_frame = curr_frame
        prev_pts = next_pts
    return tracks

def get_patch(image, center, patch_size):
    center_tuple = tuple(center)
    
    h, w = image.shape
    x, y = center
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.zeros((patch_size, patch_size), dtype=image.dtype)
    
    patch = cv2.getRectSubPix(image, (patch_size, patch_size), center_tuple)
    return patch

def refine_track(track, frames, patch_size=15):
    valid_positions = np.flatnonzero(np.array([p is not None for p in track])).tolist()
    
    if len(valid_positions) <= 1:
        return track
    first_valid_idx = valid_positions[0]
    template_pos = track[first_valid_idx]
    template_frame = frames[first_valid_idx]
    
    template = get_patch(template_frame, template_pos, patch_size).astype(np.float32)
    
    valid_positions_to_refine = valid_positions[1:]
    initial_positions = np.array([track[i] for i in valid_positions_to_refine])
    initial_guess = initial_positions.flatten()
    
    def residuals(params):
        pts = params.reshape(-1, 2)
        res = np.concatenate([
            (get_patch(frames[valid_positions_to_refine[i]], pts[i], patch_size).astype(np.float32) - template).flatten()
            for i in range(len(valid_positions_to_refine))
        ])
        return res
    
    try:
        result = least_squares(residuals, initial_guess, verbose=0, ftol=1e-5, xtol=1e-5)
        refined_positions = result.x.reshape(-1, 2)
        
        refined_track = list(track)
        for idx in range(len(valid_positions_to_refine)):
            refined_track[valid_positions_to_refine[idx]] = tuple(refined_positions[idx])
            
        return refined_track
    except Exception as e:
        print(f"refine failed: {e}")
        return track

def visualize_tracks(
    frames, refined_tracks, output_path="tracked_output.avi", fps=preview_fps
):
    color_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in frames]
    
    keys = list(refined_tracks.keys())
    colors_arr = np.random.randint(0, 255, (len(keys), 3))
    colors = {k: (int(colors_arr[i, 0]), int(colors_arr[i, 1]), int(colors_arr[i, 2])) for i, k in enumerate(keys)}
    
    num_frames = len(frames)
    for i in range(num_frames):
        for feature_index, track in refined_tracks.items():
            if i < len(track) and track[i] is not None:
                x, y = track[i]
                cv2.circle(color_frames[i], (int(round(x)), int(round(y))), 3, colors[feature_index], -1)
                if i > 0 and i-1 < len(track) and track[i-1] is not None:
                    prev_x, prev_y = track[i-1]
                    cv2.circle(color_frames[i], (int(round(prev_x)), int(round(prev_y))), 1, colors[feature_index], -1)
                    cv2.line(color_frames[i], (int(round(prev_x)), int(round(prev_y))),
                             (int(round(x)), int(round(y))), colors[feature_index], 1)
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in color_frames:
        out.write(frame)
    out.release()
    del color_frames
    gc.collect()
    print(f"saved preview: {output_path}")

def save_tracks_to_mat(refined_tracks, num_frames, filename='tracks.mat'):
    track_x, track_y, _ = track_arrays_from_dict(refined_tracks, num_frames)
    n_features = track_x.shape[0]
    sio.savemat(filename, {'track_x': track_x, 'track_y': track_y})
    print(f"saved tracks: {filename}")
    
    total_points = n_features * num_frames
    tracked_points = np.sum(~np.isnan(track_x))
    percent_tracked = 100 * tracked_points / total_points
    
    print("tracking stats")
    print(f"  features: {n_features}")
    print(f"  frames: {num_frames}")
    print(f"  tracked: {tracked_points}/{total_points} ({percent_tracked:.2f}%)")
    
    coverage_per_frame = np.sum(~np.isnan(track_x), axis=0) / n_features * 100
    print(f"  min coverage: {np.min(coverage_per_frame):.2f}%")
    print(f"  max coverage: {np.max(coverage_per_frame):.2f}%")
    print(f"  mean coverage: {np.mean(coverage_per_frame):.2f}%")

def save_tracks_to_txt(refined_tracks, num_frames, filename='tracks.txt'):
    track_x, track_y, _ = track_arrays_from_dict(refined_tracks, num_frames)
    write_tracks_txt(filename, track_x, track_y)

sample_out_dir = "data/good"


def track_video_to_disk(
    video_path,
    out_dir,
    label_fn=None,
    redetect_interval=15,
    max_redetect=20,
    write_preview=True,
    preview_fps=preview_fps,
    center_roi_frac=None,
    yolo_mask_features=False,
    yolo_mask_union_frames: int = 1,
    redetect_full_frame: Optional[bool] = None,
    feature_quality=0.02,
    min_feature_distance=12,
    feature_max_corners=1500,
    feature_grid: Tuple[int, int] = (4, 4),
    yolo_seg_weights=None,
    yolo_device="cpu",
    yolo_imgsz=640,
    yolo_conf=0.25,
    yolo_save_point_labels=True,
    max_frames=120,
    track_scale=1.0,
    fast_lk=True,
    lk_preset: str = "fast",
    lk_max_jump_frac: Optional[float] = 0.12,
    feature_distribute_experimental: bool = False,
    feature_spread_min_dist: Optional[float] = None,
    feature_use_clahe: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.dirname(video_path):
        video_path = os.path.join(out_dir, video_path)

    cv2.setUseOptimized(True)

    ts = float(track_scale)
    if ts <= 0 or ts > 1.0:
        ts = 1.0

    # YOLO runs in a second pass over the file (see yolo_segmentation.segment_video_file) so we
    # do not keep every BGR frame in RAM — that pattern was getting this process OOM-killed (zsh: killed).
    frames, first_gray_full = load_video_frames(
        video_path, max_frames=max_frames, scale=ts
    )
    if not frames or first_gray_full is None:
        print("no frames loaded")
        return None
    print(f"loaded {len(frames)} frames (scale={ts:.3f})")

    cv2.imwrite(os.path.join(out_dir, "frame000000.jpg"), first_gray_full)

    full_h, full_w = first_gray_full.shape[:2]
    h0, w0 = frames[0].shape[:2]
    det_mask = None
    if center_roi_frac is not None and 0 < center_roi_frac < 1:
        det_mask = center_roi_mask(h0, w0, float(center_roi_frac))
        print(
            f"ROI mask: center {center_roi_frac:.0%}"
        )

    if yolo_seg_weights and yolo_mask_features:
        ku = max(1, int(yolo_mask_union_frames))
        bgr_list = read_first_k_bgr_frames(video_path, ku)
        _adj = []
        for f in bgr_list:
            if f.shape[:2] != (h0, w0):
                f = cv2.resize(f, (w0, h0))
            _adj.append(f)
        bgr_list = _adj
        if bgr_list:
            maps, _ = segment_bgr_frames(
                yolo_seg_weights,
                bgr_list,
                device=yolo_device,
                imgsz=yolo_imgsz,
                conf=yolo_conf,
            )
            acc = np.zeros((h0, w0), dtype=bool)
            for t in range(maps.shape[0]):
                acc |= maps[t] > 0
            fg = acc.astype(np.uint8) * 255
            if np.any(fg > 0):
                ksz = max(3, int(min(h0, w0) * 0.004)) | 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                fg = cv2.erode(fg, kernel, iterations=1)
                if det_mask is not None:
                    det_mask = cv2.bitwise_and(det_mask, fg)
                else:
                    det_mask = fg
                print(
                    f"YOLO mask from first {maps.shape[0]} frame(s)"
                )
            else:
                print(
                    "YOLO mask empty; using ROI/full frame"
                )
        else:
            print("skip yolo mask (no bgr frames)")
        gc.collect()

    if redetect_full_frame is None:
        redetect_full_frame = bool(yolo_seg_weights and yolo_mask_features)

    if redetect_full_frame:
        if center_roi_frac is not None and 0 < center_roi_frac < 1:
            redetect_mask = center_roi_mask(h0, w0, float(center_roi_frac))
        else:
            redetect_mask = None
        print(
            "redetect: no YOLO mask for new points"
            + (
                " (central ROI still applies)."
                if redetect_mask is not None
                else " (full frame)."
            )
        )
    else:
        redetect_mask = det_mask

    fgr, fgc = max(1, int(feature_grid[0])), max(1, int(feature_grid[1]))
    if fgr > 1 or fgc > 1:
        print(
            f"Feature grid {fgr}×{fgc}: up to {int(feature_max_corners) // (fgr * fgc)} corners/cell "
            f"(multi-object spread)."
        )

    detect_cap = int(feature_max_corners)
    if feature_distribute_experimental:
        detect_cap = min(12000, max(detect_cap * 4, detect_cap))

    features = detect_features(
        frames[0],
        max_corners=detect_cap,
        quality_level=float(feature_quality),
        min_distance=int(min_feature_distance),
        detection_mask=det_mask,
        feature_grid=(fgr, fgc) if (fgr > 1 or fgc > 1) else (1, 1),
        use_clahe=feature_use_clahe,
    )
    if features.size == 0:
        print("no features detected")
        return None
    print(f"features on frame 0: {len(features)}")

    if feature_distribute_experimental and features.size > 0:
        diag = float(np.hypot(float(w0), float(h0)))
        sep = (
            float(feature_spread_min_dist)
            if feature_spread_min_dist is not None
            else 0.018 * diag
        )
        features = farthest_point_spread_subsample(
            features, int(feature_max_corners), sep
        )
        print(
            f"Spread subsample (experimental): {len(features)} features "
            f"(min_sep≈{sep:.1f}px, cap={feature_max_corners})."
        )

    preset = str(lk_preset).lower().strip()
    if not fast_lk:
        preset = "accurate"
    elif preset not in ("fast", "fastest", "accurate"):
        preset = "fast"

    tracks = track_features_with_redetection(
        frames,
        features,
        redetect_interval=redetect_interval,
        max_redetect=max_redetect,
        redetect_mask=redetect_mask,
        feature_quality=float(feature_quality),
        min_feature_distance=int(min_feature_distance),
        lk_preset=preset,
        lk_max_jump_frac=lk_max_jump_frac,
        feature_grid=(fgr, fgc) if (fgr > 1 or fgc > 1) else (1, 1),
        feature_use_clahe=feature_use_clahe,
    )
    print("tracking done")

    refined_tracks = tracks

    if write_preview:
        visualize_tracks(
            frames,
            refined_tracks,
            output_path=os.path.join(out_dir, "tracked_output.avi"),
            fps=float(preview_fps),
        )

    if ts != 1.0:
        inv = 1.0 / ts
        refined_tracks = scale_track_coordinates(refined_tracks, inv, inv)
        print(f"rescaled tracks x{inv:.3f}")

    num_frames = len(frames)

    track_x, track_y, column_ids = track_arrays_from_dict(refined_tracks, num_frames)
    tracks_path = os.path.join(out_dir, "tracks.txt")
    write_tracks_txt(tracks_path, track_x, track_y)

    if label_fn is not None:
        labels = label_fn(frames, refined_tracks, column_ids)
        if labels is not None:
            save_point_labels(os.path.join(out_dir, "point_labels.npy"), np.asarray(labels))

    # Drop grayscale list before YOLO + full seg_maps — avoids OOM ("zsh: killed") on long clips.
    del frames
    gc.collect()

    if yolo_seg_weights is not None:
        # seg_maps must match native video resolution (tracks are in full-res coords)
        seg_maps, seg_names = segment_video_file(
            yolo_seg_weights,
            video_path,
            num_frames,
            (full_h, full_w),
            device=yolo_device,
            imgsz=yolo_imgsz,
            conf=yolo_conf,
        )
        save_seg_bundle(out_dir, seg_maps, seg_names)
        if label_fn is None and yolo_save_point_labels:
            pl = label_tracks_from_seg_maps(seg_maps, refined_tracks, column_ids)
            save_point_labels(os.path.join(out_dir, "point_labels.npy"), pl)

    return {"tracks": refined_tracks, "out_dir": out_dir, "column_ids": column_ids, "num_frames": num_frames}


def main(video_path, out_dir=None, label_fn=None, max_frames=120):
    out_dir = out_dir or sample_out_dir
    track_video_to_disk(video_path, out_dir, label_fn=label_fn, max_frames=max_frames)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python feature_tracker.py <video_path>")
    else:
        main(sys.argv[1])
