#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2018 Max Planck Society for non-commercial scientific research
# This file is part of psbody.mesh project which is released under MPI License.
# See file LICENSE.txt for full license details.

from .tri_normals import TriToScaledNormal
import numpy as np


def triangle_area(v, f):
    """Computes the area associated to a set of triangles"""
    return (np.sqrt(np.sum(TriToScaledNormal(v, f) ** 2, axis=1)) / 2.).flatten()
