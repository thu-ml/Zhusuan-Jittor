import jittor as jt
from jittor import Module


class SGMCMC(Module):
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
        return self.execute(bn, observed, resample, step)
