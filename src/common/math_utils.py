from __future__ import annotations


def clip_vector(values, lower, upper):
    return [max(l, min(v, u)) for v, l, u in zip(values, lower, upper)]
