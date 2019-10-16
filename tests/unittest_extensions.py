import unittest
import numpy as np


### WTF again ???
def sigfigs(n, sf):
    float('%.1g' % n)


class ExtendedTest(unittest.TestCase):
    def assertAlmostEqual(self, a, b, places=7, msg='', delta=None, sigfigs=None):
        if sigfigs is None:
            super(ExtendedTest, self).assertAlmostEqual(a, b, places, msg, delta)
        else:
            a_ = float(('%%.%dg' % sigfigs) % a)
            b_ = float(('%%.%dg' % sigfigs) % b)
            if a_ != b_:
                raise AssertionError(msg or "%f != %f to %d significant figures (%f != %f)" % (a, b, sigfigs, a_, b_))

    def assertDictOfArraysEqual(self, a, b, msg=''):
        self.assertIsInstance(a, dict, msg or 'First argument is not a dictionary')
        self.assertIsInstance(b, dict, msg or 'Second argument is not a dictionary')
        self.assertSetEqual(set(a.keys()), set(b.keys()), msg or 'Keys do not match')
        for k in a.keys():
            if isinstance(a[k], np.ndarray) and isinstance(b[k], np.ndarray):
                np.testing.assert_array_equal(a[k], b[k], err_msg=msg + "\nwith key [%s]" % (k))
            else:
                np.testing.assert_array_equal(np.array(a[k]), np.array(b[k]), err_msg=msg + "\nwith key [%s]" % (k))

    def assertDictOfArraysAlmostEqual(self, a, b, decimal=6, msg=''):
        self.assertIsInstance(a, dict, msg or 'First argument is not a dictionary')
        self.assertIsInstance(b, dict, msg or 'Second argument is not a dictionary')
        self.assertSetEqual(set(a.keys()), set(b.keys()), msg or 'Keys do not match')
        for k in a.keys():
            if isinstance(a[k], np.ndarray) and isinstance(b[k], np.ndarray):
                np.testing.assert_array_almost_equal(a[k], b[k], decimal=decimal, err_msg=msg + "\nwith key [%s]" % (k))
            else:
                np.testing.assert_array_almost_equal(np.array(a[k]), np.array(b[k]), decimal=decimal, err_msg=msg + "\nwith key [%s]" % (k))
