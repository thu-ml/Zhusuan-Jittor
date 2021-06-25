import jittor as jt


class Distribution(object):
    def __init__(self,
                 dtype,
                 param_dtype,
                 is_continuous,
                 is_reparameterized,
                 use_path_derivative=False,
                 group_ndims=0,
                 **kwargs):

        self._dtype = dtype
        self._param_dtype = param_dtype
        self._is_continuous = is_continuous
        self._is_reparameterized = is_reparameterized
        self._use_path_derivative = use_path_derivative

        if isinstance(group_ndims, int):
            if group_ndims < 0:
                raise ValueError("group_ndims must be non-negative.")
            self._group_ndims = group_ndims
        else:
            pass

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    def batch_shape(self):
        return self._batch_shape()

    def _batch_shape(self):
        raise NotImplementedError()

    def sample(self, n_samples=None):
        if n_samples is None:
            samples = self._sample(n_samples=1)
            return samples
        elif isinstance(n_samples, int):
            return self._sample(n_samples)
        else:
            pass

    def _sample(self, n_samples):
        raise NotImplementedError()

    def log_prob(self, given):
        log_p = self._log_prob(given)

        if self._group_ndims > 0:
            return jt.sum(log_p, [i for i in range(-self._group_ndims, 0)])
        else:
            return log_p

    def _log_prob(self, given):
        raise NotImplementedError()
