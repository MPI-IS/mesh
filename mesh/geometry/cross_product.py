#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2012-07-20.

import numpy as np


def CrossProduct(a, b):
    """Computes the cross product of 2 vectors"""
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)

    a1 = a[:, 0]
    a2 = a[:, 1]
    a3 = a[:, 2]

    Ax = np.zeros((len(a1), 3, 3))
    Ax[:, 0, 1] = -a3
    Ax[:, 0, 2] = +a2
    Ax[:, 1, 0] = +a3
    Ax[:, 1, 2] = -a1
    Ax[:, 2, 0] = -a2
    Ax[:, 2, 1] = +a1

    return _call_einsum_matvec(Ax, b)


def _call_einsum_matvec(m, righthand):
    r = righthand.reshape(m.shape[0], 3)
    return np.einsum('ijk,ik->ij', m, r).flatten()


def _call_einsum_matmat(m, righthand):
    r = righthand.reshape(m.shape[0], 3, -1)
    return np.einsum('ijk,ikm->ijm', m, r).reshape(-1, r.shape[2])
