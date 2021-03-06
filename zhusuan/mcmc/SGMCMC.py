import jittor as jt
from jittor import Module


class SGMCMC(Module):
    """
    Base class for stochastic gradient MCMC (SGMCMC) algorithms.

    SGMCMC is a class of MCMC algorithms which utilize stochastic gradients
    instead of the true gradients. 
    
    The typical code for SGMCMC inference is like::

        sgmcmc = zs.mcmc.SGLD(learning_rate=lr)
        net = BayesianNet()
        for epoch in range(epoch_size):
            for step in range(num_batches):
                w_samples = model.sample(net, {'x': x, 'y': y})

                for i, (k, w) in enumerate(w_samples.items()):
                    # Utilize stochastic gradients by samples and update parameters.
                    ...

    """
    def __init__(self):
        super().__init__()
        self.t = 0

    def _update(self, bn, observed):
        raise NotImplementedError()

    def execute(self, bn, observed, resample=False, step=1):
        if resample:
            self.t = 0
            bn.execute(observed)
            self.t += 1

            self._latent = {k: v.tensor for k, v in bn.nodes.items() if k not in observed.keys()}
            self._latent_k = self._latent.keys()
            self._var_list = [self._latent[k] for k in self._latent_k]
            sample_ = dict(zip(self._latent_k, self._var_list))

            for i in range(len(self._var_list)):
                self._var_list[i] = self._var_list[i].detach()
            return sample_

        for s in range(step):
            self._update(bn, observed)
            self.t += 1

        sample_ = dict(zip(self._latent_k, self._var_list))
        return sample_

    def initialize(self):
        self.t = 0

    def sample(self, bn, observed, resample=False, step=1):
        """
        Running one sgmcmc iteration.

        :param bn: A instance of :class:`~zhusuan.framework.bn.BayesianNet`.
        :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from names of observed `StochasticTensor` s to their values.
        :param resample: Flag indicates if the sampler need get the var list of the :class:`~zhusuan.framework.bn.BayesianNet` instance, usually set to True on first sgmcmc iteration.
        :return: A list of Var, samples generated by sgmcmc iteration.
        """
        return self.execute(bn, observed, resample, step)
