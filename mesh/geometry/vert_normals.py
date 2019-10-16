#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2013-03-12.


import scipy.sparse as sp
import numpy as np
from .tri_normals import NormalizedNx3, TriNormalsScaled
from ..utils import col


def MatVecMult(mtx, vec):
    return mtx.dot(col(vec)).flatten()


def VertNormals(v, f):
    return NormalizedNx3(VertNormalsScaled(v, f))


def VertNormalsScaled(v, f):
    IS = f.flatten()
    JS = np.array([range(f.shape[0])] * 3).T.flatten()
    data = np.ones(len(JS))

    IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
    JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))  # is this right?
    data = np.concatenate((data, data, data))

    faces_by_vertex = sp.csc_matrix((data, (IS, JS)), shape=(v.size, f.size))

    # faces_by_vertex should be 3 x wider...?
    return NormalizedNx3(MatVecMult(faces_by_vertex, TriNormalsScaled(v, f)))
