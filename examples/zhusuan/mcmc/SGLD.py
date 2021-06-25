import jittor as jt

import math

from zhusuan.mcmc.SGMCMC import SGMCMC


class SGLD(SGMCMC):
    def __init__(self, learning_rate):
        super().__init__()
        self.lr = jt.array(learning_rate)
        self.lr_min = jt.array(1e-4)

    def _update(self, bn, observed):
        observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
        bn.execute(observed_)

        log_joint_ = bn.log_joint()
        grad = jt.grad(log_joint_, self._var_list)

        for i, _ in enumerate(grad):
            epsilon = jt.normal(0., math.sqrt(self.lr), size=self._var_list[i].shape)
            self._var_list[i] = self._var_list[i] + 0.5 * self.lr * grad[i] + epsilon
            self._var_list[i] = self._var_list[i].detach()
