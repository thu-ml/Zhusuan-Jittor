import jittor as jt
from jittor import Module

class Transform(Module):
    def __init__(self):
        super().__init__()
        self.is_invertible = True
    
    def _forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def _inverse(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, *args, inverse=False, **kwargs):
        if not inverse:
            return self._forward(*args, **kwargs)
        else:
            return self._inverse(*args, **kwargs)