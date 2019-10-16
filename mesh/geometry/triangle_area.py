#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.

from .tri_normals import TriToScaledNormal
import numpy as np


def triangle_area(v, f):
    """Computes the area associated to a set of triangles"""
    return (np.sqrt(np.sum(TriToScaledNormal(v, f) ** 2, axis=1)) / 2.).flatten()
