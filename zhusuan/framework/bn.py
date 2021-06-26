import jittor as jt
from jittor import Module

from zhusuan.framework.stochastic_tensor import StochasticTensor
from zhusuan.distributions import *
from zhusuan.flows import *


class BayesianNet(Module):
    def __init__(self, observed=None):
        super(BayesianNet, self).__init__()
        self._nodes = {}
        self._cache = {}
        self._observed = observed if observed else {}

    @property
    def nodes(self):
        return self._nodes

    @property
    def cache(self):
        return self._cache

    @property
    def observed(self):
        return self._observed

    def observe(self, observed):
        self._observed = {}
        for k, v in observed.items():
            self._observed[k] = v
        return self

    def sn(self, *args, **kwargs):
        return self.stochastic_node(*args, **kwargs)

    def stochastic_node(self, distribution, name, **kwargs):
        _dist = globals()[distribution](**kwargs)
        self._nodes[name] = StochasticTensor(self, name, _dist, **kwargs)
        return self._nodes[name].tensor

    def _log_joint(self):
        _ret = 0
        for k, v in self._nodes.items():
            if isinstance(v, StochasticTensor):
                try:
                    _ret = _ret + v.log_prob()
                except:
                    _ret = v.log_prob()
        return _ret

    def log_joint(self, use_cache=False):
        if use_cache:
            if not hasattr(self, '_log_joint_cache'):
                self._log_joint_cache = self._log_joint()
        else:
            self._log_joint_cache = self._log_joint()
        return self._log_joint_cache
