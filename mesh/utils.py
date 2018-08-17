#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2018 Max Planck Society for non-commercial scientific research
# This file is part of psbody.mesh project which is released under MPI License.
# See file LICENSE.txt for full license details.


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def sparse(i, j, data, m=None, n=None):
    import numpy as np
    from scipy.sparse import csc_matrix
    ij = np.vstack((i.flatten().reshape(1, -1), j.flatten().reshape(1, -1)))

    if m is None:
        return csc_matrix((data, ij))
    else:
        return csc_matrix((data, ij), shape=(m, n))
