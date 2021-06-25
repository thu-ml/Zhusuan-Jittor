import jittor as jt
from jittor import Module

from zhusuan.transforms.base import Transform

class InvertibleTransform(Transform):
    def __init__(self):
        super().__init__()
        self.is_invertible = True
    