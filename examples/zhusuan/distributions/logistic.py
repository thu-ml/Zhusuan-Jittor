import jittor as jt
from jittor import nn
import numpy as np

from zhusuan.distributions import Distribution

class Logistic(Distribution):
    def __init__(self,
                dtype='float32',
                param_dtype='float32',
                is_continues=True,
                is_reparameterized=True,
                group_ndims=0,
                **kwargs):
        super(Logistic, self).__init__(dtype,
                                       param_dtype,
                                       is_continues,
                                       is_reparameterized,
                                       group_ndims=group_ndims,
                                       **kwargs)
        self._loc = kwargs['loc']
        self._scale = kwargs['scale']
    
    def _batch_shape(self):
        return self._loc.shape
    
    def _sample(self, n_samples=1, **kwargs):
        if n_samples > 1:
            _shape = self._loc.shape
            _shape = [n_samples] + _shape
            _len = len(self._loc.shape)
            _loc = jt.cast(jt.repeat(self._loc, [n_samples, * _len * [1]]), self._dtype)
            _scale = jt.cast(jt.repeat(self._scale, [n_samples * _len *[1]]), self._dtype)
        else:
            _shape = self._loc.shape
            _loc = jt.cast(self._loc, self._dtype)
            _scale = jt.cast(self._scale, self._dtype)
        
        if not self.is_reparameterized:
            _loc.stop_grad()
            _std.stop_grad()
        
        uniform = jt.init.uniform(_shape, self._dtype, 0., 1.)
        epsilon = jt.log(uniform) - jt.log(1 - uniform)
        _sample = _loc + _scale * epsilon
        self.sample_cache = _sample
        return _sample
    
    def _log_prob(self, sample=None, **kwargs):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._loc.shape):
            n_samples = sample.shape[0]
            _len = len(self._loc.shape)
            _loc = jt.repeat(self._loc, [n_samples * _len * [1]])
            _scale = jt.repeat(self._std, [n_samples * _len * [1]])
        else:
            _loc = self._loc
            _scale = self._scale
        if self.is_reparameterized:
            _loc.stop_grad()
            _scale.stop_grad()
        z = (sample - _loc) / _scale
        return -z - 2. * nn.Softplus()(-z) - jt.log(_scale)


        