from __future__ import annotations

import argparse
import os
import sys

from reconstruction_pipeline import video_to_metric_structure

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("-o", "--out", default="data/video_run")
    p.add_argument("--show", action="store_true")
    p.add_argument("--mesh", action="store_true", default=True)
    p.add_argument("--no-mesh", dest="mesh", action="store_false")
    p.add_argument("--yolo", default=None)
    p.add_argument("--feature-clahe", action="store_true", default=True)
    p.add_argument("--no-feature-clahe", dest="feature_clahe", action="store_false")
    p.add_argument("--feature-quality", type=float, default=0.008)
    p.add_argument("--min-feature-distance", type=int, default=5)
    p.add_argument("--feature-max-corners", type=int, default=2000)
    p.add_argument("--feature-grid", nargs=2, type=int, default=[4, 4])
    p.add_argument("--redetect-interval", type=int, default=8)
    p.add_argument("--max-redetect", type=int, default=40)
    p.add_argument("--track-scale", type=float, default=0.75)
    p.add_argument("--max-frames", type=int, default=50)
    args = p.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        print(f"video not found: {video}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.abspath(args.out)
    max_frames = None if args.max_frames <= 0 else args.max_frames

    pipeline_args = {
        "show_plot": args.show,
        "show_mesh": args.mesh,
        "save_mesh_obj": args.mesh,
        "yolo_seg_weights": args.yolo,
        "dump_metrics": True,
        "max_frames": max_frames,
        "feature_quality": args.feature_quality,
        "min_feature_distance": args.min_feature_distance,
        "feature_max_corners": args.feature_max_corners,
        "feature_grid": tuple(args.feature_grid),
        "redetect_interval": args.redetect_interval,
        "max_redetect": args.max_redetect,
        "track_scale": float(args.track_scale),
        "feature_use_clahe": args.feature_clahe,
    }
    result = video_to_metric_structure(video, out_dir, **pipeline_args)
    if result is None:
        sys.exit(1)
    print("done")
    print(f"points: {result['S_scaled_path']}")
    print(f"tracks: {result['tracks_txt']}")
    mp = result.get("metrics_path")
    if mp:
        base = os.path.dirname(mp)
        print(f"metrics: {mp}")
        print(f"           {os.path.join(base, 'metrics.txt')}")


if __name__ == "__main__":
    main()
