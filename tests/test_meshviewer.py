
import unittest
import os
import time
import itertools

from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh.mesh import Mesh
from . import test_data_folder


class TestMeshViewer(unittest.TestCase):
    """Check the MeshViewer class."""

    def setUp(self):

        fnames = [os.path.join(test_data_folder, i) for i in os.listdir(
            test_data_folder) if os.path.splitext(i)[1].lower() == '.ply']

        # We build a cycle to make sure we have enough meshes
        self.meshes = itertools.cycle(Mesh(filename=fname) for fname in fnames)

        self.mvs = MeshViewers(shape=[2, 2])
        self.mvs[0][0].set_static_meshes([next(self.meshes)])
        self.mvs[0][1].set_static_meshes([next(self.meshes)])
        self.mvs[1][0].set_static_meshes([next(self.meshes)])
        self.mvs[1][1].set_static_meshes([next(self.meshes)])

    def test_launch_smoke_test(self):
        """this test just opens a mesh window, waits, and kills the window"""

        print('keeping MeshViewer alive for 10 seconds..')
        time.sleep(10)
        print('killing MeshViewer and exiting...')

    def test_snapshot(self):
        """test snapshots from mesh windows"""

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', prefix='test_snapshot') as f:
            self.mvs[0][0].save_snapshot(f.name)
            self.assertTrue(os.path.isfile(f.name))
