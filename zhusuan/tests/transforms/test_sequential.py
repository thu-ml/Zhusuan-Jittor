import jittor as jt
from jittor import nn
import unittest

from zhusuan.tests.transforms import TestInvertibleTransform

from zhusuan.transforms import Sequential
from zhusuan.transforms.invertible import InvertibleTransform

class TestSequential(TestInvertibleTransform):
    def test_invertible(self):
        class SimpleTransform(InvertibleTransform):
            def __init__(self):
                super().__init__()
            
            def _forward(self, x, v, **kwargs):
                return 2 * x + 1, v + 1, None
            
            def _inverse(self, x, v, **kwargs):
                return (x - 1) / 2, v - 1, None

        class SimpleTransform2(InvertibleTransform):
            def __init__(self):
                super().__init__()
            
            def _forward(self, x, v, **kwargs):
                return (2 * x + 1, v + 1), None
            
            def _inverse(self, x, v, **kwargs):
                return ((x - 1) / 2, v - 1), None
        
        modules = []
        # Two different ways to pass vars
        for i in range(5):
            modules.append(SimpleTransform())
        for i in range(5):
            modules.append(SimpleTransform2())
        transform = Sequential(modules)
        x = jt.randn([10, 10])
        v = jt.randn([10, 10])
        self.assert_invertible(x, v, transform=transform, decimal=5)

if __name__ == '__main__':
    unittest.main()