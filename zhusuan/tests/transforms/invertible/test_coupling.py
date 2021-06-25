import jittor as jt
from jittor import nn
import unittest

from zhusuan.tests.transforms import TestInvertibleTransform

from zhusuan.transforms.invertible.coupling import *

class TestAdditiveCoupling(TestInvertibleTransform):


    def test_invertible(self):
        batch_size = 10
        in_out_dim = 10
        mid_dim = 20
        hidden = 3
        mask = get_coupling_mask(in_out_dim, 1, 1)[0]
        # Default Net
        t1 = AdditiveCoupling(in_out_dim, mid_dim, hidden, mask)
        x = jt.randn([batch_size, in_out_dim])
        self.assert_invertible(x, transform=t1)
        # Customize Net
        net = nn.Sequential(nn.Linear(in_out_dim, mid_dim),
                            nn.Tanh(),
                            nn.Linear(mid_dim, in_out_dim))
        t2 = AdditiveCoupling(mask=mask, inner_nn=net)
        self.assert_invertible(x, transform=t2)

if __name__ == '__main__':
    unittest.main()