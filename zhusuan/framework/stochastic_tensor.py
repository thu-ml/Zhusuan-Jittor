import jittor as jt


class StochasticTensor(object):
    def __init__(self, bn, name, dist, observation=None, **kwargs):
        if bn is None:
            pass
        self._bn = bn
        self._name = name
        self._dist = dist
        self._n_samples = kwargs.get("n_samples", None)
        self._observation = observation
        super(StochasticTensor, self).__init__()

        self._reduce_mean_dims = kwargs.get("reduce_mean_dims", None)
        self._reduce_sum_dims = kwargs.get("reduce_sum_dims", None)
        self._multiplier = kwargs.get("multiplier", None)

    def _check_observation(self, observation):
        return observation

    @property
    def bn(self):
        return self._bn

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dist(self):
        return self._dist

    def is_observed(self):
        return self._observation is not None

    @property
    def tensor(self):
        if self._name in self._bn.observed.keys():
            self._dist.sample_cache = self._bn.observed[self._name]
            return self._bn.observed[self._name]
        else:
            _samples = self._dist.sample(n_samples=self._n_samples)
            return _samples

    @property
    def shape(self):
        return self.tensor.shape

    def log_prob(self, sample=None):
        _log_probs = self._dist.log_prob(sample)
        if self._reduce_mean_dims:
            _log_probs = jt.mean(_log_probs, self._reduce_mean_dims, keepdims=True)

        if self._reduce_sum_dims:
            _log_probs = jt.sum(_log_probs, self._reduce_sum_dims, keepdims=True)

        if self._reduce_mean_dims or self._reduce_sum_dims:
            _m = self._reduce_mean_dims if self._reduce_mean_dims else []
            _s = self._reduce_sum_dims if self._reduce_sum_dims else []
            _dims = [*_m, *_s]
            _dims.sort(reverse=True)
            for d in _dims:
                if _log_probs.shape == [1]:
                    break
                _log_probs = jt.squeeze(_log_probs, d)

        if self._multiplier:
            _log_probs = _log_probs * self._multiplier

        return _log_probs
