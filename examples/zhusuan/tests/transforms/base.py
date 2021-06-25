import jittor as jt
import unittest
import numpy as np

class TestInvertibleTransform(unittest.TestCase):
    def assert_invertible(self, *inputs, transform=None, decimal=7):
        z, log_det = transform.execute(*inputs, inverse=False)
        if not isinstance(z, tuple):
            xr, _ = transform.execute(z, inverse=True)
            np.testing.assert_almost_equal(inputs[0].numpy(), xr.numpy(), decimal=decimal)
        else:
            xr = transform.execute(*z, inverse=True)
            for i, _x in enumerate(z):
                np.testing.assert_almost_equal(inputs[i].numpy(), xr[i].numpy(), decimal=decimal)