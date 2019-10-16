#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 MPI. All rights reserved.

import numpy as np
import unittest
from psbody.mesh.visibility import visibility_compute


class TestVisibility(unittest.TestCase):

    def test_box(self):
        v = np.array([[0.50, 0.50, 0.50],
                      [-0.5, 0.50, 0.50],
                      [0.50, -0.5, 0.50],
                      [-0.5, -0.5, 0.50],
                      [0.50, 0.50, -0.5],
                      [-0.5, 0.50, -0.5],
                      [0.50, -0.5, -0.5],
                      [-0.5, -0.5, -0.5]])
        f = np.array([[1, 2, 3], [4, 3, 2], [1, 3, 5], [7, 5, 3],
                      [1, 5, 2], [6, 2, 5], [8, 6, 7], [5, 7, 6],
                      [8, 7, 4], [3, 4, 7], [8, 4, 6], [2, 6, 4]], dtype=np.uint32) - 1
        n = v / np.linalg.norm(v[0])

        # test considering omnidirectional cameras
        vis, n_dot_cam = visibility_compute(v=v, f=f, cams=np.array([[1.0, 0.0, 0.0]]))
        self.assertTrue(((v.T[0] > 0) == vis).all())
        # test considering omnidirectional cameras and minimum dot product
        # between camera-vertex ray and normal .5
        vis, n_dot_cam = visibility_compute(v=v, f=f, n=n, cams=np.array([[1e10, 0.0, 0.0]]))
        vis = np.logical_and(vis, n_dot_cam > .5)
        self.assertTrue(((v.T[0] > 0) == vis).all())
        # test considering two omnidirectional cameras
        vis, n_dot_cam = visibility_compute(v=v, f=f, cams=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        self.assertTrue(((v.T[1:3] > 0) == vis).all())

        vextra = np.array([[.9, .9, .9],
                           [-.9, .9, .9],
                           [.9, -.9, .9],
                           [-.9, -.9, .9]], dtype=np.double)
        fextra = np.array([[1, 2, 3], [4, 3, 2]], dtype=np.uint32) - 1
        # test considering extra meshes that can block light
        cams = np.array([[0.0, 0.0, 10.0]])
        vis, n_dot_cam = visibility_compute(v=v, f=f, cams=cams, extra_v=vextra, extra_f=fextra)
        self.assertTrue((np.zeros_like(v.T[0]) == vis).all())

        # test considering extra meshes that can block light, but only if the
        # if the distance is at least 1.0
        vis, n_dot_cam = visibility_compute(v=v, f=f, cams=np.array([[0.0, 0.0, 10.0]]),
                                            extra_v=vextra, extra_f=fextra, min_dist=1.0)
        self.assertTrue(((v.T[2] > 0) == vis).all())
