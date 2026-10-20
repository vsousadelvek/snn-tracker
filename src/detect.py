"""Deteccao: eventos esparsos -> bounding boxes.

Objetos em movimento geram clusters densos de eventos. Dilatamos para
agregar eventos vizinhos e extraimos componentes conectados.
"""
import cv2
import numpy as np


def detect(events, min_area=8, dilate_k=5):
    """events: (H,W) {-1,0,1} -> lista de boxes (x,y,w,h)."""
    active = (events != 0).astype(np.uint8)
    active = cv2.dilate(active, np.ones((dilate_k, dilate_k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(active)
    return [(int(x), int(y), int(w), int(h))
            for x, y, w, h, a in stats[1:] if a >= min_area]
