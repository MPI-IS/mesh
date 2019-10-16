#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2014 Max Planck Society. All rights reserved.

import os
import numpy as np
import unittest

from . import test_data_folder
from psbody.mesh.mesh import Mesh
from psbody.mesh.geometry.tri_normals import TriToScaledNormal, NormalizeRows
import psbody.mesh.aabb_normals as aabb_normals


class TestAABBNormal(unittest.TestCase):

    def setUp(self):
        simpleobjpath = os.path.join(test_data_folder, 'test_doublebox.obj')
        self.simple_m = Mesh(filename=simpleobjpath)
        cylinderpath = os.path.join(test_data_folder, 'cylinder.obj')
        self.cylinder_m = Mesh(filename=cylinderpath)
        cylinder_trans_path = os.path.join(test_data_folder, 'cylinder_trans.obj')
        self.cylinder_trans_m = Mesh(filename=cylinder_trans_path)
        self_int_cyl_path = os.path.join(test_data_folder, 'self_intersecting_cyl.obj')
        self.self_int_cyl_m = Mesh(filename=self_int_cyl_path)

    # error_p = ||p - q|| + eps*(1 - p_n*p_q)
    # therefore, eps=0 should give the classic NN
    def test_dist_classic(self):
        tree_handle = aabb_normals.aabbtree_n_compute(self.simple_m.v,
                                                      self.simple_m.f.astype(np.uint32).copy(),
                                                      0.0)
        query_v = np.array([[0.5, 0.1, 0.25],
                            [0.5, 0.1, 0.25]])
        query_n = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]])
        closest_tri, closest_p = aabb_normals.aabbtree_n_nearest(tree_handle, query_v, query_n)
        self.assertTrue((closest_tri == np.array([[0, 0]])).all())
        self.assertTrue((closest_p == query_v).all())

    def test_dist_normals(self):
        tree_handle = aabb_normals.aabbtree_n_compute(self.simple_m.v,
                                                      self.simple_m.f.astype(np.uint32).copy(),
                                                      0.5)
        query_v = np.array([[0.5, 0.1, 0.25],
                            [0.5, 0.1, 0.25]])
        query_n = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]])
        closest_tri, closest_p = aabb_normals.aabbtree_n_nearest(tree_handle, query_v, query_n)
        self.assertTrue((closest_tri == np.array([[2, 0]])).all())
        self.assertTrue((closest_p == np.array([[0.5, 0.5, 0.25],
                                               [0.5, 0.1, 0.25]])).all())

    def test_cylinders(self):
        create_tree = lambda eps: aabb_normals.aabbtree_n_compute(self.cylinder_m.v,
                                                                  self.cylinder_m.f.astype(np.uint32).copy(),
                                                                  eps)
        tree_handle_no_normals = create_tree(0)
        tree_handle_normals = create_tree(10)

        query_v = self.cylinder_trans_m.v

        tri_n = NormalizeRows(TriToScaledNormal(self.cylinder_trans_m.v, self.cylinder_trans_m.f))

        query_n = np.zeros(self.cylinder_trans_m.v.shape)
        for i_f in range(self.cylinder_trans_m.f.shape[0]):
            query_n[self.cylinder_trans_m.f[i_f, :], :] += tri_n[i_f, :]
        query_n = NormalizeRows(query_n)

        closest_tri, _ = aabb_normals.aabbtree_n_nearest(tree_handle_no_normals, query_v, query_n)
        # all closest triangles are the two extremes
        self.assertTrue(np.unique(closest_tri).shape[0] <= 4)

        closest_tri_n, _ = aabb_normals.aabbtree_n_nearest(tree_handle_normals, query_v, query_n)
        # there are four triangles that do not need to be reached, in the center and in the extremes
        self.assertTrue(np.unique(closest_tri_n).shape[0] >= (self.cylinder_m.f.shape[0] - 4))

    def test_selfintersects(self):
        tree_handle_no = aabb_normals.aabbtree_n_compute(self.simple_m.v,
                                                         self.simple_m.f.astype(np.uint32).copy(),
                                                         0.5)

        self.assertTrue(aabb_normals.aabbtree_n_selfintersects(tree_handle_no) == 0)

        tree_handle_yes = aabb_normals.aabbtree_n_compute(self.self_int_cyl_m.v,
                                                          self.self_int_cyl_m.f.astype(np.uint32).copy(),
                                                          0.5)

        self.assertTrue(aabb_normals.aabbtree_n_selfintersects(tree_handle_yes) == (2 * 8))
