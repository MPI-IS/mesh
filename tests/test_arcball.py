
import unittest
import copy
import numpy as np

from psbody.mesh.arcball import Matrix3fT, Matrix4fT, ArcBallT, \
    Point2fT, Matrix3fSetRotationFromQuat4f, Matrix3fMulMatrix3f, Matrix4fSetRotationFromMatrix3f
from numpy import double


class TestArcball(unittest.TestCase):

    def test_arcball(self):
        # Unit testing of the ArcBall class and the real math behind it.
        # Simulates a click and drag followed by another click and drag.

        Transform = Matrix4fT()
        ThisRot = Matrix3fT()

        ArcBall = ArcBallT(640, 480)

        # First click
        LastRot = copy.copy(ThisRot)
        mouse_pt = Point2fT(500, 250)
        ArcBall.click(mouse_pt)

        # First drag
        mouse_pt = Point2fT(475, 275)
        ThisQuat = ArcBall.drag(mouse_pt)
        np.testing.assert_almost_equal(ThisQuat, [0.08438914, -0.08534209, -0.06240178, 0.99080837])

        #
        ThisRot = Matrix3fSetRotationFromQuat4f(ThisQuat)
        # Linear Algebra matrix multiplication A = old, B = New : C = A * B
        ThisRot = Matrix3fMulMatrix3f(LastRot, ThisRot)
        Transform = Matrix4fSetRotationFromMatrix3f(Transform, ThisRot)

        first_transform_gt = np.array([[0.97764552, -0.1380603, 0.15858325, 0.],
                                       [0.10925253, 0.97796899, 0.17787792, 0.],
                                       [-0.17964739, -0.15657592, 0.97119039, 0.],
                                       [0., 0., 0., 1., ]],
                                      dtype=np.double)
        np.testing.assert_almost_equal(Transform, first_transform_gt)

        # Done with first drag

        # second click
        LastRot = copy.copy(ThisRot)
        lastrot_gt = np.array([[0.97764552, -0.1380603, 0.15858325],
                               [0.10925253, 0.97796899, 0.17787792],
                               [-0.17964739, -0.15657592, 0.97119039]],
                              dtype=double)
        np.testing.assert_almost_equal(LastRot, lastrot_gt)

        mouse_pt = Point2fT(350, 260)
        ArcBall.click(mouse_pt)
        # second drag
        mouse_pt = Point2fT(450, 260)
        ThisQuat = ArcBall.drag(mouse_pt)

        np.testing.assert_almost_equal(ThisQuat, [0.00710336, 0.31832787, 0.02679029, 0.94757545])

        ThisRot = Matrix3fSetRotationFromQuat4f(ThisQuat)
        ThisRot = Matrix3fMulMatrix3f(LastRot, ThisRot)
        # print ThisRot
        Transform = Matrix4fSetRotationFromMatrix3f(Transform, ThisRot)

        second_transform_gt = np.array([[0.88022292, -0.08322023, -0.46720669, 0.],
                                        [0.14910145, 0.98314685, 0.10578787, 0.],
                                        [0.45052907, -0.16277808, 0.8777966, 0.],
                                        [0., 0., 0., 1.00000001]],
                                       dtype=double)

        np.testing.assert_almost_equal(Transform, second_transform_gt)
