"""Harvest de blobs: baixa videos por classe e extrai crops de eventos.

Uso: python -m src.harvest_blobs [--skip-download] [--videos N]
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from .events import DVSSimulator
from .detect import detect
from .blob_snn import CLASSES, crop_events

QUERIES = {
    "person": "person walking overhead view fixed camera",
    "car":    "cars traffic fixed camera street",
    "bike":   "cyclist bicycle street fixed camera",
    "truck":  "truck bus highway fixed camera",
}
MAX_CROPS_PER_VIDEO = 120
MAX_FRAMES = 600


def download(cls, query, raw_dir, n):
    out = raw_dir / cls
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "python", "-m", "yt_dlp", f"ytsearch{n}:{query}",
        "-f", "mp4[height<=480]/worst[ext=mp4]/worst",
        "--download-sections", "*0:00-0:40",
        "--no-playlist", "-o", str(out / "v%(autonumber)02d.%(ext)s"),
        "--quiet", "--no-warnings",
    ], check=False)


def extract(video_path, out_dir, start_idx):
    cap = cv2.VideoCapture(str(video_path))
    dvs = DVSSimulator(size=(320, 240), threshold=0.15)
    trail = []
    saved, n_fr = 0, 0
    while n_fr < MAX_FRAMES and saved < MAX_CROPS_PER_VIDEO:
        ok, f = cap.read()
        if not ok:
            break
        n_fr += 1
        trail.append(dvs.process(f))
        trail = trail[-3:]
        if len(trail) < 3:
            continue
        acc = np.clip(sum(trail), -1, 1)
        boxes = [b for b in detect(acc, min_area=25, dilate_k=7,
                                   min_density=0.10)
                 if b[2] >= 5 and b[3] >= 5]
        if not boxes:
            continue
        # maior blob do frame = objeto principal do video
        b = max(boxes, key=lambda b: b[2] * b[3])
        np.save(out_dir / f"{video_path.stem}_{saved:04d}.npy",
                crop_events(acc, b).numpy().astype(np.float16))
        saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--videos", type=int, default=6)
    args = ap.parse_args()

    raw_dir, data_dir = Path("raw_blobs"), Path("data_blobs")
    for cls in CLASSES:
        if not args.skip_download:
            print(f"baixando '{cls}': {QUERIES[cls]}")
            download(cls, QUERIES[cls], raw_dir, args.videos)
        out_dir = data_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for v in sorted((raw_dir / cls).glob("*.*")):
            total += extract(v, out_dir, total)
        print(f"{cls}: {total} crops")


if __name__ == "__main__":
    main()
