import jittor as jt

from zhusuan.distributions import Distribution

class Flow(Distribution):
    def __init__(self, latents=None, transform=None, flow_kwargs=None, dtype='float32', group_ndims=0, **kwargs):
        super().__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

        self._latents = latents
        self._transform = transform
    
    def _sample(self, n_samples=1, **kwargs):
        if n_samples == -1:
            return 0
        else:
            z = self._latents.sample(n_samples)
            x_hat = self._transform.execute(z, inverse=True, **kwargs)
            return x_hat[0]

    def _log_prob(self, *given, **kwargs):
        z, log_detJ = self._transform.execute(*given, inverse=False, **kwargs)
        log_likelihood = jt.sum(self._latents.log_prob(z[0]) + log_detJ, 1)
        return log_likelihood
