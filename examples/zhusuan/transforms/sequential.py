import jittor as jt
from jittor import nn

from zhusuan.transforms.base import Transform

class Sequential(Transform):
    def __init__(self, modules):
        super().__init__()
        self.modules = nn.Sequential(modules)
    
    def _forward(self, *x, **kwargs):
        log_detJ = []
        for i in range(len(self.modules)):
            x = self.modules[i](*x, inverse=False, **kwargs)
            assert isinstance(x, tuple)
            assert len(x) >= 2
            if x[-1] is not None:
                log_detJ.append(x[-1])
            if isinstance(x[0], tuple):
                x = x[0]
            else:
                x = x[:len(x) - 1]
            assert isinstance(x, tuple)
        return x, sum(log_detJ) if log_detJ else jt.zeros([1])
    
    def _inverse(self, *z, **kwargs):
        # No log_det(jacobian) in inverse process
        for i in reversed(range(len(self.modules))):
            z = self.modules[i](*z, inverse=True, **kwargs)
            assert isinstance(z, tuple)
            assert len(z) >= 2
            assert z[-1] is None
            if isinstance(z[0], tuple):
                z = z[0]
            else:
                z = z[:len(z) - 1]
            assert isinstance(z, tuple)
        return z