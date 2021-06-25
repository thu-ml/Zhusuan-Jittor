import jittor as jt
import unittest
from zhusuan.tests.transforms import TestInvertibleTransform

from zhusuan.transforms.invertible import Scaling

class TestScaling(TestInvertibleTransform):
    def test_invertible(self):
        batch_size = 10
        in_out_dim = 10

        x = jt.randn([batch_size, in_out_dim])
        t = Scaling(in_out_dim)
        self.assert_invertible(x, transform=t)

if __name__ == '__main__':
    unittest.main()