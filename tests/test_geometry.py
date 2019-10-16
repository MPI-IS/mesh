#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.

import os

import numpy as np
import unittest

from . import test_data_folder
from unittest.case import skipUnless

try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False


class TestGeometry(unittest.TestCase):

    @skipUnless(has_cv2, 'skipping tests requiring OpenCV')
    def test_rodrigues(self):
        from psbody.mesh.geometry.rodrigues import rodrigues

        test_data = (
            np.array([0, 0, 0], dtype=np.double),
            np.array([1, -1, 0.5], dtype=np.double),
            np.array([[1, -1, 0.5]], dtype=np.double),
            np.array([[1, -1, 0.5]], dtype=np.double).T,
            np.eye(3, dtype=np.double),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.double),
            np.array([[0.22629564, 0.95671228, -0.18300792],
                      [-0.18300792, 0.22629564, 0.95671228],
                      [0.95671228, -0.18300792, 0.22629564]], dtype=np.double),
        )
        for r_in in test_data:
            true_r, true_dr = cv2.Rodrigues(r_in)
            our_r, our_dr = rodrigues(r_in)
            np.testing.assert_array_almost_equal(our_r, true_r, verbose=True)
            np.testing.assert_array_almost_equal(our_dr, true_dr, verbose=True)

    def test_cross_product(self):
        from psbody.mesh.geometry.cross_product import CrossProduct

        npts = 6
        a = np.random.randn(3 * npts)
        b = np.random.randn(3 * npts)

        c = CrossProduct(a, b)

        # this should be close to (or exactly) zero
        our_answer = c.flatten()
        numpy_answer = np.cross(a.reshape(-1, 3), b.reshape(-1, 3)).flatten()

        self.assertTrue(max(abs(our_answer - numpy_answer)) < 1e-15)

    def test_vert_normals(self):
        from psbody.mesh.geometry.vert_normals import VertNormals
        from psbody.mesh.mesh import Mesh
        mesh = Mesh(filename=os.path.join(test_data_folder, 'sphere.ply'))
        pred = VertNormals(mesh.v, mesh.f)

        vn_obs = mesh.estimate_vertex_normals().reshape((-1, 3))
        vn_pred = pred.reshape((-1, 3))

        self.assertTrue(np.max(np.abs(vn_pred.flatten() - vn_obs.flatten())) < 1e-15)

    def test_barycentric_coordinates_of_projection(self):
        """Tests backwards compatibility with old matlab
        function of the same name."""
        from psbody.mesh.geometry.barycentric_coordinates_of_projection import barycentric_coordinates_of_projection

        p = np.array([[-120, 48, -30, 88, -80],
                      [71, 102, 29, -114, -291],
                      [161, 72, -78, -106, 142]]).T

        q = np.array([[32, -169, 32, -3, 108],
                      [-75, -10, 31, -16, 110],
                      [136, -24, -86, 62, -86]]).T

        u = np.array([[8, -1, 37, -108, 109],
                      [-120, 152, -22, 3, 153],
                      [-110, -76, 111, 55, 9]]).T

        v = np.array([[-148, 233, -19, -139, -18],
                      [-73, -61, 88, -141, -19],
                      [-105, 74, -76, 48, 141]]).T

        b = np.array([[1.5266, -0.8601, 1.3245, 2.4450, 1.3452],
                      [-1.5346, 0.8556, -0.1963, -2.1865, -2.0794],
                      [1.0080, 1.0046, -0.1282, 0.7415, 1.7342]]).T

        b_est = barycentric_coordinates_of_projection(p, q, u, v)
        self.assertTrue(np.max(np.abs(b_est.flatten('F') - b.flatten('F'))) < 1e-3)

        p = p[0, :]
        q = q[0, :]
        u = u[0, :]
        v = v[0, :]
        b = b[0, :]

        b_est = barycentric_coordinates_of_projection(p, q, u, v)
        self.assertTrue(np.max(np.abs(b_est.flatten('F') - b.flatten('F'))) < 1e-3)

    @unittest.skipIf(
        not os.path.isfile(os.path.join(test_data_folder, 'female_template.ply')),
        'No data file.')
    def test_trinormal(self):

        from psbody.mesh.mesh import Mesh
        from psbody.mesh.geometry.tri_normals import TriNormals, TriToScaledNormal, TriNormalsScaled, NormalizeRows

        m = Mesh(filename=os.path.join(test_data_folder, 'female_template.ply'))

        # Raffi: I do not know what this thing is supposed to test, maybe stability over some noise...
        tn = TriNormals(m.v, m.f)
        tn2 = NormalizeRows(TriToScaledNormal(m.v, m.f))

        eps = 1e-8
        mvc = m.v.copy()
        mvc[0] += eps
        tn_b = TriNormals(mvc, m.f)
        tn2_b = NormalizeRows(TriToScaledNormal(mvc, m.f))
        # our TriNormals empirical: sp.csc_matrix(tn_b.flatten() - tn.flatten()) / eps
        # old TriToScaledNormals empirical': sp.csc_matrix(tn2_b.flatten() - tn2.flatten()) / eps

        # apparently just for printing sparsly
        # import scipy.sparse as sp
        # print sp.csc_matrix(tn_b.flatten() - tn.flatten()) / eps
        np.testing.assert_almost_equal(tn_b.flatten() - tn.flatten(),
                                       tn2_b.flatten() - tn2.flatten())

        tn = TriNormalsScaled(m.v, m.f)
        tn2 = TriToScaledNormal(m.v, m.f)
        eps = 1e-8
        mvc = m.v.copy()
        mvc[0] += eps

        tn_b = TriNormalsScaled(mvc, m.f)
        tn2_b = TriToScaledNormal(mvc, m.f)

        np.testing.assert_almost_equal(tn_b.flatten() - tn.flatten(),
                                       tn2_b.flatten() - tn2.flatten())
