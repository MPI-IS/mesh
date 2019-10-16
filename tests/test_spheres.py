
import unittest
import numpy as np

from psbody.mesh.sphere import Sphere


class TestSphere(unittest.TestCase):

    def test_intersection_is_symmetric(self):
        d = 2
        s0 = Sphere(np.array([0, 0, 0]), 1)
        for dd in np.linspace(0, d, 10):
            s1 = Sphere(np.array([d - dd, 0, 0]), 0.5)
            self.assertAlmostEqual(s0.intersection_vol(s1), s1.intersection_vol(s0))
