# CMPUT 428 Project

## Setup

```bash
pip install -r requirements.txt
```

## Run main pipeline

From the `code` folder:

```bash
python video_to_3d.py ../test_media/multiple.mov -o ../out_test --yolo yolo11n-seg.pt \ 
  --feature-clahe \
  --feature-quality 0.008 --min-feature-distance 5 --feature-max-corners 3000 \
  --feature-grid 4 4 --redetect-interval 8 --max-redetect 40 \
  --track-scale 0.75 --max-frames 90 --mesh
```

Feel free to run it on different videos or to increase the number of frames / features!
