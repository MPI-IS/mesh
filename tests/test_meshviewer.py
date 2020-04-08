import itertools
import multiprocessing
import os
import random
import socket
import time
import unittest

from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import (
    MeshViewers,
    MeshViewerRemote,
    ZMQ_PORT_MIN,
    ZMQ_PORT_MAX)

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
        time.sleep(1)
        print('killing MeshViewer and exiting...')

    def test_snapshot(self):
        """test snapshots from mesh windows"""

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', prefix='test_snapshot') as f:
            self.mvs[0][0].save_snapshot(f.name)
            self.assertTrue(os.path.isfile(f.name))


class TestRemoteMeshViewer(unittest.TestCase):
    def is_port_open(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect_ex(("0.0.0.0", port))
            sock.settimeout(0.1)
            return True
        except:
            return False
        finally:
            sock.close()

    def pick_random_open_port(self):
        while True:
            port = random.randint(ZMQ_PORT_MIN, ZMQ_PORT_MAX)
            if self.is_port_open(port):
                return port

    def test_starting_a_remote_opens_a_port_for_listening(self):
        """
        Start a MeshViewerRemote instance and verify that it's listening
        for a given port.
        """
        port = self.pick_random_open_port()
        proc = multiprocessing.Process(target=MeshViewerRemote, kwargs={"port": port})
        self.assertTrue(self.is_port_open(port))
        if proc.is_alive():
            proc.terminate()
