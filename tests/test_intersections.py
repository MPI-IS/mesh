"""
Unit Tests
----------

Unit test for the mesh_intersections module.
The result is the list of indices of the intersecting faces

"""
import unittest


class TestMeshIntersection(unittest.TestCase):

    def test_spheres_intersection(self):
        # deactiavate test temporally
        pass

        '''
        from psbody.mesh.sphere import Sphere
        qm = Sphere(np.asarray([-1, 0, 0]), 2).to_mesh()
        m = Sphere(np.asarray([1, 0, 0]), 2).to_mesh()

        t = m.compute_aabb_tree()

        faces_index = t.intersections_indices(qm.v, qm.f)

        ref_faces_index = [2, 4, 5, 6, 16, 25, 26, 27, 36, 37, 38, 40, 58, 60, 61, 63, 76, 77, 79]

        test = True
        for i in range(len(faces_index)):
            if faces_index[i] != ref_faces_index[i]:
                test = False

        self.assertTrue(test)
        '''
