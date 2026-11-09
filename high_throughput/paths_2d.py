"""In-plane paths matching the geometric convention of the PhononBench website.

All qz are zero. Metric labels are plotting conventions, not proof of the
atomic structure's space-group symmetry.
"""
import numpy as np


def band_path_2d(cell, npoints=101):
    a, b = np.asarray(cell)[:2]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    gamma = np.degrees(np.arccos(np.clip(np.dot(a, b) / (la * lb), -1, 1)))
    points = {"G": [0, 0, 0], "X": [.5, 0, 0], "Y": [0, .5, 0],
              "M": [.5, .5, 0], "S": [.5, .5, 0], "K": [1/3, 1/3, 0]}
    if abs(la-lb)/la < .05 and abs(gamma-120) < 2:
        points["M"] = [.5, 0, 0]
        sequence = ["G", "K", "M", "G"]
    elif abs(la-lb)/la < .05 and abs(gamma-90) < 2:
        sequence = ["G", "X", "M", "G"]
    elif abs(la-lb)/max(la, lb) > .05 and abs(gamma-90) < 2:
        sequence = ["G", "X", "S", "Y", "G"]
    else:
        sequence = ["G", "X", "M", "Y", "G"]
    paths = [np.linspace(points[x], points[y], npoints) for x, y in zip(sequence, sequence[1:])]
    labels = [r"$\Gamma$" if s == "G" else s for s in sequence]
    return paths, labels, [True] * (len(paths)-1) + [False]
