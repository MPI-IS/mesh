#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2013 Max Planck Society. All rights reserved.


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
