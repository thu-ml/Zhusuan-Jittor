import jittor as jt
from jittor import nn
import numpy as np

import math

from zhusuan.transforms.invertible import InvertibleTransform

def get_coupling_mask(n_dim, n_channel, n_mask, split_type='ChessBoard'):
    with jt.no_grad():
        masks = []
        if split_type == 'ChessBoard':
            if n_channel == 1:
                mask = jt.arange(0, n_dim, dtype='float32') % 2
                for i in range(n_mask):
                    masks.append(mask)
                    mask = 1. - mask
        else:
            raise NotImplementedError()
        return masks


class AdditiveCoupling(InvertibleTransform):
    def __init__(self, in_out_dim=-1, mid_dim=-1, hidden=-1, mask=None, inner_nn=None):
        super().__init__()
        if inner_nn is None:
            self.nn = []
            self.nn += [nn.Linear(in_out_dim, mid_dim),
                        nn.ReLU()]
            for _ in range(hidden - 1):
                self.nn += [nn.Linear(mid_dim, mid_dim),
                            nn.ReLU()]
            self.nn += [nn.Linear(mid_dim, in_out_dim)]
            self.nn = nn.Sequential(*self.nn)
        else:
            self.nn = inner_nn
        self.mask = mask
    
    def _forward(self, x, **kwargs):
        x1, x2 = self.mask * x, (1 - self.mask) * x
        shift = self.nn(x1)
        z1, z2 = x1, x2 + shift * (1. - self.mask)
        return z1 + z2, None
    
    def _inverse(self, z, **kwargs):
        z1, z2 = self.mask * z, (1 - self.mask) * z
        shift = self.nn(z1)
        x1, x2 = z1, z2 - shift * (1. - self.mask)
        return x1 + x2, None


