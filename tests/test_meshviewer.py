
import unittest
import numpy as np
import os
import time

from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh.mesh import Mesh
from . import test_data_folder

try:
    import PIL
    has_pil = True
except ImportError:
    has_pil = False


class TestMeshViewer(unittest.TestCase):

    def setUp(self):
        fnames = [os.path.join(test_data_folder, i) for i in os.listdir(test_data_folder) if os.path.splitext(i)[1].lower() == '.ply']

        self.meshes = [Mesh(filename=fname) for fname in fnames]

        self.mvs = MeshViewers(shape=[2, 2])
        self.mvs[0][0].set_static_meshes([self.meshes[0]])
        self.mvs[0][1].set_static_meshes([self.meshes[1]])
        self.mvs[1][0].set_static_meshes([self.meshes[2]])
        self.mvs[1][1].set_static_meshes([self.meshes[0]])  # only 2 .ply files left in the GitHub version

    def test_launch_smoke_test(self):
        """this test just opens a mesh window, waits, and kills the window"""

        sphere = Mesh(filename=os.path.join(test_data_folder, 'sphere.ply'))
        sphere.v = sphere.v / 10.

        print('keeping MeshViewer alive for 10 seconds..')
        time.sleep(10)
        print('killing MeshViewer and exiting...')

        if 0:
            # this cannot be unit tested
            from psbody.mesh.utils import row
            click = self.mvs[0][0].get_mouseclick()
            subwin_row = click['which_subwindow'][0]
            subwin_col = click['which_subwindow'][1]
            sphere.v = sphere.v - row(np.mean(sphere.v, axis=0)) + row(np.array([click['x'], click['y'], click['z']]))
            self.mvs[subwin_row][subwin_col].set_dynamic_meshes([sphere])

            print('items in mouseclick dict are as follows:')
            print(click)

    @unittest.skipUnless(has_pil, "skipping test that requires Pillow")
    def test_snapshot(self):
        """test snapshots from mesh windows"""

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', prefix='test_snapshot') as f:
            self.mvs[0][0].save_snapshot(f.name)
