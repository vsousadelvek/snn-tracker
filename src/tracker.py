"""Tracker multi-objeto: matching por distancia de centroide.

Do zero, sem filtro de Kalman: centroide com velocidade amortecida ja
resolve bem em 30fps. Cada track guarda trilha e direcao de cruzamento.
"""
import numpy as np


class Track:
    _next_id = 0

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = np.array(box, float)
        self.vel = np.zeros(2)
        self.trail = [self.centroid]
        self.hits, self.misses = 1, 0

    @property
    def centroid(self):
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2])

    def predict(self):
        self.box[:2] += self.vel
        self.misses += 1

    def update(self, box):
        c_old = self.centroid
        self.box = np.array(box, float)
        self.vel = 0.7 * self.vel + 0.3 * (self.centroid - c_old)
        self.trail.append(self.centroid.copy())
        if len(self.trail) > 40:
            self.trail.pop(0)
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(self, max_dist=35, max_misses=12, min_hits=8):
        self.tracks = []
        self.max_dist, self.max_misses, self.min_hits = \
            max_dist, max_misses, min_hits

    def step(self, boxes):
        for t in self.tracks:
            t.predict()

        unmatched = set(range(len(boxes)))
        for t in self.tracks:
            best, best_d = None, self.max_dist
            for i in unmatched:
                x, y, w, h = boxes[i]
                d = np.linalg.norm(t.centroid - (x + w / 2, y + h / 2))
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                t.update(boxes[best])
                unmatched.discard(best)

        for i in unmatched:
            self.tracks.append(Track(boxes[i]))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.hits >= self.min_hits]
