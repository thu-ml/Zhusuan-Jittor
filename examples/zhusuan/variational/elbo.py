import jittor as jt
from jittor import Module


class ELBO(Module):
    def __init__(self, generator, variational):
        super(ELBO, self).__init__()

        self.generator = generator
        self.variational = variational

    def log_joint(self, nodes):
        log_joint_ = None
        for n_name in nodes.keys():
            try:
                log_joint_ += nodes[n_name].log_prob()
            except:
                log_joint_ = nodes[n_name].log_prob()
        return log_joint_

    def execute(self, observed, reduce_mean=True):
        nodes_q = self.variational(observed).nodes

        _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        nodes_p = self.generator(_observed).nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)
        # sgvb
        if len(logqz.shape) > 0 and reduce_mean:
            elbo = jt.mean(logpxz - logqz)
        else:
            elbo = logpxz - logqz
        return -elbo
