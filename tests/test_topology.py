#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.

import numpy as np
import unittest
import os

from . import test_data_folder, temporary_files_folder


class TestVisibility(unittest.TestCase):

    @unittest.skip('Too long - skipping for the moment.')
    def test_qslim_smoke_test(self):
        from psbody.mesh.mesh import Mesh
        from psbody.mesh.topology.decimation import qslim_decimator
        from psbody.mesh.geometry.triangle_area import triangle_area

        m = Mesh(filename=os.path.join(test_data_folder, 'female_template.ply'))

        ta = triangle_area(m.v, m.f)
        m.set_face_colors(ta / np.max(ta))

        qslimmer = qslim_decimator(m, factor=0.1)
        m2 = qslimmer(m)
        ta = triangle_area(m2.v, m2.f)
        m2.set_face_colors(ta / np.max(ta))


class TestLoopSubdivision(unittest.TestCase):

    @unittest.skipIf(
        not os.path.isfile(os.path.join(test_data_folder, 'female_template.ply')),
        'No data file.')
    def test_loop_subdivision_smoke_test(self):
        from psbody.mesh import Mesh
        from psbody.mesh.topology.subdivision import loop_subdivider

        m1 = Mesh(filename=os.path.join(test_data_folder, 'female_template.ply'))
        sdv = loop_subdivider(m1)

        self.assertIsNotNone(sdv)
        self.assertTrue(hasattr(sdv, "faces"))

        f_new = sdv.faces

        v_new = sdv(m1.v)
        self.assertIsNotNone(v_new)
        v_new = v_new.reshape((-1, 3))

        v_new_want_edge = sdv(m1.v, want_edges=True)
        self.assertIsNotNone(v_new_want_edge)
        v_new_want_edge = v_new_want_edge.reshape((-1, 3))

        m2 = Mesh(v=v_new, f=f_new)

        m1.reset_normals()
        m2.reset_normals()

        m1.write_ply(os.path.join(temporary_files_folder, 'lowres.ply'))
        m2.write_ply(os.path.join(temporary_files_folder, 'highres.ply'))

        if 0:
            from psbody.mesh import MeshViewers
            mvs = MeshViewers(shape=(2, 2))
            mvs[0][0].set_static_meshes([m1])
            m1.f = []
            mvs[0][1].set_static_meshes([m1])
            mvs[1][0].set_static_meshes([m2])
            m2.f = []
            mvs[1][1].set_static_meshes([m2])


class TestConnectivity(unittest.TestCase):

    @unittest.skipIf(
        not os.path.isfile(os.path.join(test_data_folder, 'female_template.ply')),
        'No data file.')
    def test_connectivity_smoke_test(self):

        from psbody.mesh import Mesh
        from psbody.mesh.topology.connectivity import get_vert_connectivity, get_faces_per_edge
        m = Mesh(filename=os.path.join(test_data_folder, 'female_template.ply'))
        vconn = get_vert_connectivity(m)
        fpe = get_faces_per_edge(m)

        self.assertIsNotNone(vconn)
        self.assertIsNotNone(fpe)
