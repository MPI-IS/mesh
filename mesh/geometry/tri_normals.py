#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2012-07-22.


"""
tri_normals.py

"""

from ..utils import col
from .cross_product import CrossProduct

import numpy as np


def TriNormals(v, f):
    return NormalizedNx3(TriNormalsScaled(v, f))


def TriNormalsScaled(v, f):
    return CrossProduct(TriEdges(v, f, 1, 0), TriEdges(v, f, 2, 0))


def NormalizedNx3(v):
    ss = np.sum(v.reshape(-1, 3) ** 2, axis=1)
    ss[ss == 0] = 1
    s = np.sqrt(ss)

    return (v.reshape(-1, 3) / col(s)).flatten()


def TriEdges(v, f, cplus, cminus):
    assert(cplus >= 0 and cplus <= 2 and cminus >= 0 and cminus <= 2)
    return _edges_for(v, f, cplus, cminus)


def _edges_for(v, f, cplus, cminus):
    return (
        v.reshape(-1, 3)[f[:, cplus], :] -
        v.reshape(-1, 3)[f[:, cminus], :]).ravel()


def TriToScaledNormal(x, tri):

    v = x.reshape(-1, 3)

    def v_xyz(iV):
        return v[tri[:, iV], :]

    return np.cross(v_xyz(1) - v_xyz(0), v_xyz(2) - v_xyz(0))


def _bsxfun(oper, a, b):
    if a.shape[0] == b.shape[0] or a.shape[1] == b.shape[1]:
        return oper(a, b)
    elif min(a.shape) == 1 and min(b.shape) == 1:
        if a.shape[0] == 1:
            return oper(np.tile(a, (b.shape[0], 1)), b)
        else:
            return oper(np.tile(a, (1, b.shape[1], b)))
    else:
        raise '_bsxfun failure'


def NormalizeRows(x):

    s = (np.sqrt(np.sum(x ** 2, axis=1))).flatten()
    s[s == 0] = 1
    return _bsxfun(np.divide, x, col(s))
