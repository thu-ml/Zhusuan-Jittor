import jittor as jt
from jittor import nn

from zhusuan.transforms.invertible import InvertibleTransform

class Scaling(InvertibleTransform):
    def __init__(self, n_dim):
        super().__init__()
        self.log_scale = nn.init.constant(shape=[1, n_dim], dtype='float32')
    
    def _forward(self, x, **kwargs):
        log_detJ = self.log_scale.clone()
        x *= jt.exp(self.log_scale)
        return x, log_detJ

    def _inverse(self, z, **kwargs):
        z *= jt.exp(-self.log_scale)
        return z, None
