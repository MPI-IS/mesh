import unittest
import numpy as np
import tempfile
import os
import shutil
from os.path import join as pjoin

from psbody.mesh.mesh import Mesh
from psbody.mesh.errors import MeshError, SerializationError

from .unittest_extensions import ExtendedTest

from . import test_data_folder


class TestMesh(ExtendedTest):

    def setUp(self):
        self.box_v = np.array([[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]]).T
        self.box_f = np.array([[0, 1, 2], [3, 2, 1], [0, 2, 4], [6, 4, 2], [0, 4, 1], [5, 1, 4], [7, 5, 6], [4, 6, 5], [7, 6, 3], [2, 3, 6], [7, 3, 5], [1, 5, 3]])
        self.box_segm = {'a': np.array(range(6), dtype=np.uint32),
                         'b': np.array([6, 10, 11], dtype=np.uint32),
                         'c': np.array([7, 8, 9], dtype=np.uint32)}
        self.landm = {'pospospos': 0,
                      'negnegneg': 7}
        self.landm_xyz = {'pospospos': np.array([0.5, 0.5, 0.5]),
                          'negnegneg': np.array([-0.5, -0.5, -0.5])}
        self.test_obj_path = pjoin(test_data_folder, "test_box.obj")
        self.test_ply_path = pjoin(test_data_folder, "test_box.ply")
        self.test_bin_ply_path = pjoin(test_data_folder, "test_box_le.ply")
        self.test_bad_ply_path = pjoin(test_data_folder, "test_ascii_bad_endings.ply")
        self.test_pp_path = pjoin(test_data_folder, "test_box.pp")
        self.test_sphere_path = pjoin(test_data_folder, "sphere.ply")

    def test_load_obj(self):
        m = Mesh(filename=self.test_obj_path)
        self.assertTrue((m.v == self.box_v).all())
        self.assertTrue((m.f == self.box_f).all())
        self.assertDictOfArraysEqual(m.segm, self.box_segm)
        self.assertEqual(m.landm, self.landm)
        self.assertDictOfArraysEqual(m.landm_xyz, self.landm_xyz)

    def test_load_ply(self):
        m = Mesh(filename=self.test_ply_path, ppfilename=self.test_pp_path)
        self.assertTrue((m.v == self.box_v).all())
        self.assertTrue((m.f == self.box_f).all())
        self.assertTrue(m.landm == self.landm)

    def test_ascii_bad_ply(self):
        """Ensure that the proper exception is raised when a file fails to be read."""
        with self.assertRaisesRegex(SerializationError, 'Failed to open PLY file\.'):
            Mesh(filename=self.test_bad_ply_path)

        # The next two tests are unnecessary,
        # just demonstrating the exception hierarchy:
        with self.assertRaises(MeshError):
            Mesh(filename=self.test_bad_ply_path)

        with self.assertRaises(Exception):
            Mesh(filename=self.test_bad_ply_path)

    def test_raw_initialization(self):
        m = Mesh(v=self.box_v, f=self.box_f)
        self.assertTrue((m.v == self.box_v).all())
        self.assertTrue((m.f == self.box_f).all())

    def test_writing_ascii_ply(self):
        m = Mesh(filename=self.test_ply_path)
        (_, tempname) = tempfile.mkstemp()
        m.write_ply(tempname, ascii=True)
        with open(tempname, 'r') as f:
            candidate = f.read()
        os.remove(tempname)
        with open(self.test_ply_path, 'r') as f:
            truth = f.read()
        self.assertEqual(candidate, truth)

    def test_writing_bin_ply(self):
        m = Mesh(filename=self.test_ply_path)
        (_, tempname) = tempfile.mkstemp()
        m.write_ply(tempname)
        with open(tempname, 'rb') as f:
            candidate = f.read()
        os.remove(tempname)
        with open(self.test_bin_ply_path, 'rb') as f:
            truth = f.read()
        self.assertEqual(candidate, truth)

    def test_aabb_tree(self):
        v_src = np.array([[-36, 37, 8], [5, -36, 35], [12, -15, 1], [-10, -42, -26], [-38, -32, -26], [-8, -45, 40], [44, -1, -1], [-16, 40, -13],
                          [-39, 28, -11], [-26, -10, -40], [-37, 44, 46], [8, -44, -27], [-15, 32, -48], [-46, -33, 15], [23, 15, -5],
                          [5, -20, 24], [-31, 19, -32], [-13, 13, 28], [-42, 43, 28], [-1, -6, -5]])
        f_src = np.array([[12, 16, 17], [5, 10, 1], [13, 19, 7], [13, 1, 5], [14, 8, 16], [9, 2, 8], [1, 19, 18], [4, 0, 3], [18, 15, 5], [3, 16, 2]])

        m = Mesh(v=v_src, f=f_src)
        t = m.compute_aabb_tree()

        v_query = np.array([[-19, 1, 1], [32, 29, 14], [-12, 31, 3], [-15, 44, 38], [5, 12, 9]])

        v_expected = np.array([[-19.678178, 0.364208, -1.384218], [23.000000, 15.000000, -5.000000], [-13.729523, 19.930467, 0.278131], [-31.869765, 34.228123, 44.656367], [7.794764, 18.188195, -6.471474]])
        f_expected = np.array([2, 4, 0, 1, 4])

        f_est, v_est = t.nearest(v_query)

        diff1 = abs(f_est - f_expected)
        diff2 = abs(v_est - v_expected)

        self.assertTrue(max(diff1.flatten()) < 1e-6)
        self.assertTrue(max(diff2.flatten()) < 1e-6)

    def test_estimate_vertex_normals(self):
        # normals of a sphere should be scaled versions of the vertices
        m = Mesh(filename=self.test_sphere_path)
        m.v -= np.mean(m.v, axis=0)
        rad = np.linalg.norm(m.v[0])
        vn = np.array(m.estimate_vertex_normals())
        mse = np.mean(np.sqrt(np.sum((vn - m.v / rad) ** 2, axis=1)))
        self.assertTrue(mse < 0.05)

    @unittest.skipIf(
        not os.path.isfile(os.path.join(test_data_folder, 'textured_mean_scape_female.obj')),
        'No data file.')
    def test_landmark_loader(self):
        scan_fname = pjoin(test_data_folder, 'csr0001a.ply')
        scan_lmrk = pjoin(test_data_folder, 'csr0001a.lmrk')
        template_fname = pjoin(test_data_folder, 'textured_mean_scape_female.obj')
        template_pp = pjoin(test_data_folder, 'template_caesar_picked_points.pp')
        scan = Mesh(filename=scan_fname, lmrkfilename=scan_lmrk)
        template = Mesh(filename=template_fname, ppfilename=template_pp)

        # Detecting CAESAR lmrk files:
        m = Mesh(filename=scan_fname, landmarks=scan_lmrk)
        self.assertEqual(m.landm, scan.landm)
        self.assertDictOfArraysEqual(m.landm_xyz, scan.landm_xyz)

        # Detecting Meshlab pp file
        m = Mesh(filename=template_fname, landmarks=template_pp)
        self.assertEqual(m.landm, template.landm)
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, template.landm_xyz)

        del template.landm_regressors

        def test(landmarks):
            m = Mesh(filename=template_fname, landmarks=landmarks)
            self.assertEqual(m.landm, template.landm)
            self.assertDictOfArraysAlmostEqual(m.landm_xyz, template.landm_xyz)

        import json
        import yaml
        import pickle
        tmp_dir = tempfile.mkdtemp('bodylabs-test')
        test_files = [
            (yaml, os.path.join(tmp_dir, 'landmarks.yaml'), 'w'),
            (yaml, os.path.join(tmp_dir, 'landmarks.yml'), 'w'),
            (json, os.path.join(tmp_dir, 'landmarks.json'), 'w'),
            (pickle, os.path.join(tmp_dir, 'landmarks.pkl'), 'wb'),
        ]
        test_data_ind = dict((n, int(v)) for n, v in template.landm.items())
        test_data_xyz = dict((n, v.tolist()) for n, v in template.landm_xyz.items())
        for loader, filename, mode in test_files:
            with open(filename, mode) as fd:
                loader.dump(test_data_ind, fd)
            test(filename)
            with open(filename, mode) as fd:
                loader.dump(test_data_xyz, fd)
            test(filename)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        test(template.landm)
        test(template.landm_xyz)

        m = Mesh(filename=template_fname, landmarks=[0, 1, 2])
        self.assertEqual(m.landm, {'0': 0, '1': 1, '2': 2})

        m = Mesh(filename=template_fname, landmarks=[template.v[0], template.v[7]])
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'0': template.v[0], '1': template.v[7]})

        m = Mesh(filename=template_fname, landmarks=[template.v[0].tolist(), template.v[7].tolist()])
        self.assertDictOfArraysAlmostEqual(m.landm_xyz, {'0': template.v[0], '1': template.v[7]})
