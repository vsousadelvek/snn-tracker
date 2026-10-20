"""Demo: video -> eventos -> deteccao -> tracking -> contagem.

Uso: python -m src.demo raw/v02.mp4 -o demo/tracking.mp4
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

from .events import DVSSimulator
from .detect import detect
from .tracker import Tracker
from .blob_snn import BlobSNN, CLASSES, crop_events

EV_W, EV_H = 320, 240
COLORS = [(255, 200, 0), (0, 255, 200), (255, 100, 200), (100, 200, 255),
          (255, 255, 0), (0, 200, 255), (200, 255, 100), (255, 0, 200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="demo/tracking.mp4")
    ap.add_argument("--no-classify", action="store_true")
    args = ap.parse_args()

    clf = None
    if not args.no_classify:
        ckpt = torch.load("checkpoints/blob_snn.pt", map_location="cpu",
                          weights_only=False)
        classes = ckpt["classes"]
        clf = BlobSNN(len(classes))
        clf.load_state_dict(ckpt["model"])
        clf.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (1200, 400))

    dvs = DVSSimulator(size=(EV_W, EV_H), threshold=0.15)
    tracker = Tracker()
    ev_trail = deque(maxlen=4)
    count_up, count_down = 0, 0
    counted = set()
    line_y = EV_H // 2

    n_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        events = dvs.process(frame)
        ev_trail.append(events)

        # detecta em eventos acumulados (ultimos 3 frames): blobs mais densos
        acc_ev = np.clip(sum(ev_trail), -1, 1)
        boxes = [b for b in detect(acc_ev, min_area=30, dilate_k=7,
                                   min_density=0.10)
                 if b[2] >= 5 and b[3] >= 5]
        tracks = tracker.step(boxes)

        # painel camera + tracking
        view = cv2.resize(frame, (400, 400))
        sx, sy = 400 / EV_W, 400 / EV_H
        cv2.line(view, (0, int(line_y * sy)), (400, int(line_y * sy)),
                 (255, 255, 255), 1)
        # painel eventos + tracking (mesmas tracks, resolucao DVS)
        ev_img = cv2.resize(cv2.cvtColor(
            (np.abs(np.clip(sum(ev_trail), -1, 1)) * 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR), (400, 400))
        cv2.line(ev_img, (0, int(line_y * sy)), (400, int(line_y * sy)),
                 (60, 60, 60), 1)

        for t in tracks:
            x, y, w, h = t.box
            color = COLORS[t.id % len(COLORS)]
            label = f"#{t.id}"

            # SNN classifica o crop de eventos; voto por maioria no tempo
            if clf is not None:
                crop = crop_events(acc_ev, t.box).unsqueeze(0)
                with torch.no_grad():
                    p = torch.softmax(clf(crop.to(next(clf.parameters()).device)), 1)
                cls_i = int(p.argmax(1))
                t.votes.append(classes[cls_i])
                if len(t.votes) > 15:
                    t.votes.pop(0)
                label = f"{max(set(t.votes), key=t.votes.count)} #{t.id}"

            for img, s in ((view, 1.0), (ev_img, 1.0)):
                p1 = (int(x * sx), int(y * sy)); p2 = (int((x + w) * sx), int((y + h) * sy))
                cv2.rectangle(img, p1, p2, color, 2)
                cv2.putText(img, label, (p1[0], max(12, p1[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            # trilha
            pts = [(int(c[0] * sx), int(c[1] * sy)) for c in t.trail]
            for a, b in zip(pts, pts[1:]):
                cv2.line(view, a, b, color, 1)
            # contagem: exige cruzar a linha com margem real
            margin = EV_H * 0.08
            if t.id not in counted and len(t.trail) > 10:
                y0, y1 = t.trail[0][1], t.trail[-1][1]
                if y0 < line_y - margin and y1 > line_y + margin:
                    counted.add(t.id); count_down += 1
                elif y0 > line_y + margin and y1 < line_y - margin:
                    counted.add(t.id); count_up += 1

        panel = np.zeros((400, 400, 3), np.uint8)
        cv2.putText(panel, "EVENT-TRACKER", (80, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (120, 120, 120), 2)
        cv2.putText(panel, f"tracks ativos: {len(tracks)}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        if clf is not None:
            by_cls = {}
            for t in tracks:
                if t.votes:
                    c = max(set(t.votes), key=t.votes.count)
                    by_cls[c] = by_cls.get(c, 0) + 1
            cv2.putText(panel, "  ".join(f"{k}:{v}" for k, v in by_cls.items()),
                        (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (180, 180, 180), 1)
        cv2.putText(panel, f"descendo: {count_down}", (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(panel, f"subindo:  {count_up}", (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        cv2.putText(view, "camera + tracking", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ev_img, "eventos (DVS)", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(np.hstack([view, ev_img, panel]))

    cap.release()
    out.release()

    # metricas de qualidade do tracker
    all_t = tracker.tracks + tracker.dead
    conf_t = [t for t in all_t if t.hits >= tracker.min_hits]
    lifetimes = [t.hits + t.misses for t in conf_t]
    dur_s = sum(lifetimes) / max(len(lifetimes), 1) / fps
    print(f"salvo: {args.out}")
    print(f"metricas: {len(conf_t)} tracks validos | "
          f"duracao media {dur_s:.1f}s | "
          f"ids/min {len(conf_t) / (n_frames / fps / 60):.0f} | "
          f"subindo={count_up} descendo={count_down}")


if __name__ == "__main__":
    main()
