# 3D-Segmentation

Turns a short phone video into a 3D point cloud and mesh where each point is labeled with an object class, using structure-from-motion plus a pretrained YOLO11 segmentation model.

Write-up: [docs/report.pdf](docs/report.pdf)

![demo](docs/demo.gif)

Left: class-colored mesh. Right: labeled point cloud and the YOLO segmentation of frame 0. Full recordings: [demo 1](docs/demo1.mp4) (this scene), [demo 2](docs/demo2.mp4) (a second scene).

![results](docs/results.png)

Top row: input frame, affine reconstruction, metric reconstruction. Bottom row: YOLO segmentation, labeled 3D points, textured mesh.

## Why I built this

Labeled 3D data is expensive. It usually needs LiDAR plus manual annotation, while 2D segmentation models are cheap and good. I wanted to see how far you can get by reconstructing geometry from a handheld video with classic factorization SfM, then projecting per-pixel 2D labels into 3D. It started as my CMPUT 428 project, and the write-up is in [docs/report.pdf](docs/report.pdf).

## Run it locally

Needs Python 3.12+. The YOLO weights (`code/yolo11n-seg.pt`) and two sample clips are in the repo. On Apple Silicon the Open3D wheel also needs `brew install libusb`.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cd code
python video_to_3d.py ../test_media/multiple.mov -o ../out/multiple \
  --yolo yolo11n-seg.pt --feature-max-corners 3000 --max-frames 90
```

At the end, an Open3D window opens with the mesh. `--no-mesh` skips building and saving the mesh. Outputs go to `out/multiple/`:
`tracks.txt`, `tracked_output.avi`, `q2_S_scaled.npy`, `seg_class_maps.npz`, `point_labels.npy`,
`mesh_reconstruction.obj`, `reconstruction_labeled.ply`, and `metrics.json` / `metrics.txt`.
`out_test/` is a committed sample run.

To look at an existing run again: `python view_pipeline_3d.py --out-dir ../out/multiple`.

## How it works

![architecture](docs/architecture.jpg)

Two passes over the video. SfM: Shi-Tomasi corners, pyramidal LK tracking, a 2F x P measurement matrix, rank-3 SVD for affine structure, then a metric upgrade from the orthogonality constraints (Cholesky). Segmentation: YOLO11-seg per frame, then a majority vote across frames gives each track one class. The projection step joins the two and the labeled points get Delaunay-meshed on frame 0.

- `code/feature_tracker.py`: grid-based corner detection (optional CLAHE), LK tracking with periodic re-detection and a max-jump filter. Writes `tracks.txt`.
- `code/q1_affine_factorization_method.py` and `code/q2_metric_upgrade.py`: Tomasi-Kanade factorization, then solving for the 3x3 metric correction from the per-frame constraints |r1|=|r2|=1, r1·r2=0.
- `code/yolo_segmentation.py` and `code/q3_texturing.py`: per-frame class maps, label voting across frames, and meshing. `code/reconstruction_pipeline.py` wires everything together.

## The hard part

Meshing a sparse, noisy point cloud without a proper surface reconstruction step. I triangulate the tracked points in 2D on the first frame with Delaunay, not in 3D. That's cheap and gives UV coordinates for free, so frame 0 becomes the texture. The catch is that 2D Delaunay happily connects a point on the mug to a point on the wall behind it, and you get long spikes through space.

So each triangle has to pass three filters (`q3_texturing.py`):
- longest 2D edge under a fraction of the image diagonal
- longest 3D edge under a fraction of the point cloud's bounding-box diagonal
- no two vertices carrying different non-background YOLO labels, since a triangle spanning two objects is almost always a depth discontinuity

Tight thresholds kill the spikes but leave holes. Loose ones keep the surface but let spikes back in. The defaults (0.058 and 0.10) were tuned by eye. If the 3D filter removes everything, it relaxes the limit 1.5x and then falls back to 2D-only rather than returning an empty mesh. The segmentation labels end up doing double duty: they're the output, but they also clean up the geometry.

A smaller fix that was a real problem: on longer clips the process got OOM-killed. I fixed it by tracking on downscaled grayscale frames, freeing them, and running YOLO in a second streaming pass over the video file instead of holding every BGR frame in memory.

## Stack

Python, NumPy, SciPy, OpenCV, Ultralytics YOLO11, Open3D, Matplotlib.

## Status

Course project, working end to end on short clips. Known gaps:
- Factorization assumes an affine camera, so reprojection error is high on handheld video. Bundle adjustment is the obvious next step.
- Reconstruction is metric only up to scale. By default the cloud is normalized to unit RMS radius; pass `--scale-ref I J DIST` (two track indices and their real distance) to get absolute units.
- No tests, and dependencies aren't pinned.

## License

MIT, see `LICENSE`.
