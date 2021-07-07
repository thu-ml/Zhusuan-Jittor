Variational Autoencoders
========================

**Variational Auto-Encoders** (VAE) :cite:`vae-kingma2013auto` is one of the
most widely used deep generative models.
In this tutorial, we show how to implement VAE in ZhuSuan step by step.
The full script is at
`examples/variational_autoencoders/vae.py <https://github.com/McGrady00H/Zhusuan-Jittor/blob/main/examples/variational_autoencoders/vae_mnist.py>`_.

The generative process of a VAE for modeling binarized
`MNIST <http://yann.lecun.com/exdb/mnist/>`_ data is as
follows:

.. math::

    z &\sim \mathrm{N}(z|0, I) \\
    x_{logits} &= f_{NN}(z) \\
    x &\sim \mathrm{Bernoulli}(x|\mathrm{sigmoid}(x_{logits}))

This generative process is a stereotype for deep generative models, which
starts with a latent representation (:math:`z`) sampled from a simple
distribution (such as standard Normal).
Then the samples are forwarded through a deep neural network (:math:`f_{NN}`)
to capture the complex generative process of high dimensional observations
such as images.
Finally, some noise is added to the output to get a tractable likelihood for
the model.
For binarized MNIST, the observation noise is chosen to be Bernoulli, with
its parameters output by the neural network.

Build the model
---------------

In ZhuSuan, a model is constructed using
:class:`~zhusuan.framework.bn.BayesianNet`, which describes a directed
graphical model, i.e., Bayesian networks. ::

    import zhusuan as zs

    class Generator(BayesianNet):
        def __init__(self, x_dim, z_dim, batch_size):
            # Initialize...
        def execute(self, observed):
            # Forward propagation...
            

Following the generative process, first we need a standard Normal
distribution to generate the latent representations (:math:`z`).
As presented in our graphical model, the data is generated in batches with
batch size ``n``, and for each data, the latent representation is of
dimension ``z_dim``.
So we add a stochastic node by ``stochastic_node`` method to generate samples of shape
``[n, z_dim]``::

    # z ~ N(z|0, I)
    mean = jt.zeros([batch_size, z_dim])
    std = jt.ones([batch_size, z_dim])

    z = self.stochastic_node('Normal',
                             name='z',
                             mean=mean,
                             std=std,
                             reparameterize=False,
                             reduce_mean_dims=[0],
                             reduce_sum_dims=[1])

The method ``bn.normal`` is a helper function that creates a
:class:`~zhusuan.distributions.normal.Normal` distribution and adds a
stochastic node that follows this distribution to the
:class:`~zhusuan.framework.bn.BayesianNet` instance.
The returned ``z`` is a sample of :class:`~zhusuan.framework.bn.StochasticTensor`, which 
can be mixed with Vars and fed into any Jittor operations.

.. note::

    To learn more about :class:`~zhusuan.distributions.base.Distribution` and
    :class:`~zhusuan.framework.bn.BayesianNet`. Please refer to
    :doc:`/tutorials/concepts`.

The shape of ``z_mean`` is ``[n, z_dim]``, which means that
we have ``[n, z_dim]`` independent inputs fed into the univariate
:class:`~zhusuan.distributions.normal.Normal` distribution. 
The shape of samples and probabilities evaluated at this node should
be of shape ``[n, z_dim]``. However, what we want in modeling MNIST data, is a
batch of ``[n]`` independent events, with each one producing samples of ``z``
that is of shape ``[z_dim]``, which is the dimension of latent representations.
And the probabilities in every single event in the batch should be evaluated
together, so the shape of local probabilities should be ``[n]`` instead of
``[n, z_dim]``. In ZhuSuan-Jittor, the way to achieve this is by setting ``reduce_mean_dims`` and ``reduce_sum_dims``.

Then we build a neural network of two fully-connected layers with :math:`z` 
as the input, which is supposed to learn the complex transformation that
generates images from their latent representations::

    # x_logits = f_NN(z)
    # In __init__
    self.fc1 = nn.Linear(z_dim, 500)
    self.act1 = nn.Relu()
    self.fc2 = nn.Linear(500, 500)
    self.act2 = nn.Relu()
    self.fc2_ = nn.Linear(500, x_dim)
    
    # In execute
    x_logits = self.fc2_(self.act2(self.fc2(self.act1(self.fc1(z)))))

Next, we add an observation distribution (noise) that follows the Bernoulli
distribution to get a tractable likelihood when evaluating the probability
of an image::

    # x ~ Bernoulli(x|sigmoid(x_logits))
    x_probs = nn.Sigmoid()(x_logits)
    self.sn('Bernoulli',
            name='x',
            probs=x_probs,
            reduce_mean_dims=[0],
            reduce_sum_dims=[1])

.. note::

    The :class:`~zhusuan.distributions.bernoulli.Bernoulli` distribution
    accepts log-odds of probabilities instead of probabilities.
    This is designed for numeric stability reasons. 

Putting together, the code for constructing a VAE is::

    class Generator(BayesianNet):
        def __init__(self, x_dim, z_dim, batch_size):
            super().__init__()
            self.x_dim = x_dim
            self.z_dim = z_dim
            self.batch_size = batch_size

            self.fc1 = nn.Linear(z_dim, 500)
            self.act1 = nn.Relu()
            self.fc2 = nn.Linear(500, 500)
            self.act2 = nn.Relu()

            self.fc2_ = nn.Linear(500, x_dim)
            self.act2_ = nn.Sigmoid()

        def execute(self, observed):
            self.observe(observed)
            mean = jt.zeros([self.batch_size, self.z_dim])
            std = jt.ones([self.batch_size, self.z_dim])

            z = self.sn('Normal',
                        name='z',
                        mean=mean,
                        std=std,
                        reparameterize=False,
                        reduce_mean_dims=[0],
                        reduce_sum_dims=[1])
            x_mean = self.act2_(self.fc2_(self.act2(self.fc2(self.act1(self.fc1(z))))))
            sample_x = self.sn('Bernoulli',
                            name='x',
                            probs=x_mean,
                            reduce_mean_dims=[0],
                            reduce_sum_dims=[1])
            return self

    generator = Generator(x_dim, z_dim, batch_size)

.. Reuse the model
.. ---------------

.. Unlike common deep learning models (MLP, CNN, etc.), which is for supervised
.. tasks, a key difficulty in designing programing primitives for generative
.. models is their inner reusability.
.. This is because in a probabilistic graphical model, a stochastic node can
.. have two kinds of states, **observed or latent**.
.. Consider the above case, if ``z`` is a var sampled from the prior, how
.. about when you meet the condition that ``z`` is observed?

.. We provide a solution for this. To observe any stochastic nodes,
.. pass a dictionary mapping of ``(name, Tensor)`` pairs when constructing
.. :class:`~zhusuan.framework.bn.BayesianNet`.
.. This will assign observed values to corresponding ``StochasticTensor`` s.
.. For example, to observe a batch of images ``x_batch``, write::

..     bn = zs.BayesianNet(observed={"x": x_batch})

.. .. note::

..     The observation passed must have the same type and shape as the
..     `StochasticTensor`.

.. However, we usually need to pass different configurations of observations to
.. the same :class:`~zhusuan.framework.bn.BayesianNet` more than once.
.. To achieve this, ZhuSuan provides a new class called
.. :class:`~zhusuan.framework.meta_bn.MetaBayesianNet`
.. to represent the meta version of :class:`~zhusuan.framework.bn.BayesianNet`
.. which can repeatedly produce :class:`~zhusuan.framework.bn.BayesianNet`
.. objects by accepting different observations.
.. The recommended way to construct a
.. :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` is by wrapping the
.. function with a :func:`~zhusuan.framework.meta_bn.meta_bayesian_net`
.. decorator::

..     @zs.meta_bayesian_net(scope="gen")
..     def build_gen(x_dim, z_dim, n, n_particles=1):
..         ...
..         return bn

..     model = build_gen(x_dim, z_dim, n, n_particles)

.. which transforms the function into returning a
.. :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance::

..     >>> print(model)
..     <zhusuan.framework.meta_bn.MetaBayesianNet object at ...

.. so that we can observe stochastic nodes in this way::

..     # no observation
..     bn1 = model.observe()

..     # observe x
..     bn2 = model.observe(x=x_batch)

.. Each time the function is called, a different observation assignment is used
.. to construct a :class:`~zhusuan.framework.bn.BayesianNet` instance.
.. One question you may have in mind is that if there are Tensorflow
.. `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
.. created in the above function, will them be reused across these ``bn`` s?
.. The answer is no by default, but you can enable this by switching on the
.. `reuse_variables` option in the decorator::

..     @zs.meta_bayesian_net(scope="gen", reuse_variables=True)
..     def build_gen(x_dim, z_dim, n, n_particles=1):
..         ...
..         return bn

..     model = build_gen(x_dim, z_dim, n, n_particles)

.. Then ``bn1`` and ``bn2`` will share the same set of Tensorflow Variables.

.. .. note::

..     This only shares Tensorflow Variables across different
..     :class:`~zhusuan.framework.bn.BayesianNet` instances generated by the same
..     :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` through the
..     :meth:`~zhusuan.framework.meta_bn.MetaBayesianNet.observe` method.
..     Creating multiple :class:`~zhusuan.framework.meta_bn.MetaBayesianNet`
..     objects will recreate the tensorflow Variables, for example, in

..     ::

..         m = build_gen(x_dim, z_dim, n, n_particles)
..         bn = m.observe()

..         m_new = build_gen(x_dim, z_dim, n, n_particles)
..         bn_new = m_new.observe()

..     ``bn`` and ``bn_new`` will use a different set of Tensorflow
..     Variables.

.. Since reusing Tensorflow Variables in repeated function calls is a typical
.. need, we provide another decorator
.. :func:`~zhusuan.framework.utils.reuse_variables` for the more general cases.
.. Any function decorated by :func:`~zhusuan.framework.utils.reuse_variables`
.. will automatically create Tensorflow Variables the first time they are called
.. and reuse them thereafter.

Inference and learning
----------------------

Having built the model, the next step is to learn it from binarized MNIST
images.
We conduct
`Maximum Likelihood <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`_
learning, that is, we are going to maximize the log likelihood of data in our
model:

.. math::

    \max_{\theta} \log p_{\theta}(x)

where :math:`\theta` is the model parameter.

.. note::

    In this variational autoencoder, the model parameter is the network
    weights, in other words, it's the Jittor Vars created in the
    ``fully_connected`` layers.

However, the model we defined has not only the observation (:math:`x`) but
also latent representation (:math:`z`).
This makes it hard for us to compute :math:`p_{\theta}(x)`, which we call
the marginal likelihood of :math:`x`, because we only know the joint
likelihood of the model:

.. math::

    p_{\theta}(x, z) = p_{\theta}(x|z)p(z)

while computing the marginal likelihood requires an integral over latent
representation, which is generally intractable:

.. math::

    p_{\theta}(x) = \int p_{\theta}(x, z)\;dz

The intractable integral problem is a fundamental challenge in learning latent
variable models like VAEs.
Fortunately, the machine learning society has developed many approximate
methods to address it. One of them is
`Variational Inference <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_.
As the intuition is very simple, we briefly introduce it below.

Because directly optimizing :math:`\log p_{\theta}(x)` is infeasible, we choose
to optimize a lower bound of it.
The lower bound is constructed as

.. math::

    \log p_{\theta}(x) &\geq \log p_{\theta}(x) - \mathrm{KL}(q_{\phi}(z|x)\|p_{\theta}(z|x)) \\
    &= \mathbb{E}_{q_{\phi}(z|x)} \left[\log p_{\theta}(x, z) - \log q_{\phi}(z|x)\right] \\
    &= \mathcal{L}(\theta, \phi)

where :math:`q_{\phi}(z|x)` is a user-specified distribution of :math:`z`
(called **variational posterior**) that is chosen to match the true posterior
:math:`p_{\theta}(z|x)`.
The lower bound is equal to the marginal log likelihood if and only if
:math:`q_{\phi}(z|x) = p_{\theta}(z|x)`, when the
`Kullback–Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
between them (:math:`\mathrm{KL}(q_{\phi}(z|x)\|p_{\theta}(z|x))`) is zero.

.. note::

    In Bayesian Statistics, the process represented by the Bayes' rule

    .. math::

        p(z|x) = \frac{p(z)(x|z)}{p(x)}

    is called
    `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`_,
    where :math:`p(z)` is called the **prior**, :math:`p(x|z)` is the
    conditional likelihood, :math:`p(x)` is the marginal likelihood or
    **evidence**, and :math:`p(z|x)` is known as the **posterior**.

This lower bound is usually called Evidence Lower Bound (ELBO). Note that the
only probabilities we need to evaluate in it is the joint likelihood and
the probability of the variational posterior.

In variational autoencoder, the variational posterior (:math:`q_{\phi}(z|x)`)
is also parameterized by a neural network (:math:`g`), which accepts input
:math:`x`, and outputs the mean and variance of a Normal distribution:

.. math::

    \mu_z(x;\phi), \log\sigma_z(x;\phi) &= g_{NN}(x) \\
    q_{\phi}(z|x) &= \mathrm{N}(z|\mu_z(x;\phi), \sigma^2_z(x;\phi))

In ZhuSuan, the variational posterior can also be defined as a
:class:`~zhusuan.framework.bn.BayesianNet` . The code for above definition is::

    class Variational(BayesianNet):
        def __init__(self, x_dim, z_dim, batch_size):
            super().__init__()
            self.x_dim = x_dim
            self.z_dim = z_dim
            self.batch_size = batch_size

            self.fc1 = nn.Linear(x_dim, 500)
            self.act1 = nn.Relu()
            self.fc2 = nn.Linear(500, 500)
            self.act2 = nn.Relu()

            self.fc3 = nn.Linear(500, z_dim)
            self.fc4 = nn.Linear(500, z_dim)

            self.dist = None

        def execute(self, observed):
            self.observe(observed)
            x = self.observed['x']
            z_logits = self.act2(self.fc2(self.act1(self.fc1(x))))

            z_mean = self.fc3(z_logits)
            z_std = jt.exp(self.fc4(z_logits))

            z = self.sn('Normal',
                        name='z',
                        mean=z_mean,
                        std=z_std,
                        reparameterize=True,
                        reduce_mean_dims=[0],
                        reduce_sum_dims=[1])
            return self

    variational = Variational(x_dim, z_dim, batch_size)

Having both ``model`` and ``variational``, we can build a model which calculate the lower bound as::

    model = zs.variational.ELBO(generator, variational)

The returned ``lower_bound`` is an
:class:`~zhusuan.variational.elbo.EvidenceLowerBoundObjective`
instance, which is a derivativation of Jittor's `Module`. However,
optimizing the lower bound objective needs special care.
The easiest way is to do
`stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_
(SGD), which is very common in deep learning literature.
However, the gradient computation here involves taking derivatives of an
expectation, which needs Monte Carlo estimation.
This often induces large variance if not properly handled.

.. note::

    Directly using auto-differentiation to compute the gradients of
    :class:`~zhusuan.variational.elbo.EvidenceLowerBoundObjective`
    often gives you the wrong results.
    This is because auto-differentiation is not designed to handle
    expectations.

Many solutions have been proposed to estimate the gradient of some
type of variational lower bound (ELBO or others) with relatively low variance.
To make this more automatic and easier to handle, ZhuSuan has wrapped these
gradient estimators all into methods of the corresponding
variational objective (e.g., the
:class:`~zhusuan.variational.exclusive_kl.EvidenceLowerBoundObjective`).
These functions don't return gradient estimates but a more convenient
surrogate cost.
Applying SGD on this surrogate cost with
respect to parameters is equivalent to optimizing the
corresponding variational lower bounds using the well-developed low-variance
estimator.

Here we are using the **Stochastic Gradient Variational Bayes** (SGVB)
estimator from the original paper of variational autoencoders
:cite:`vae-kingma2013auto`.
This estimator takes benefits of a clever reparameterization trick to
greatly reduce the variance when estimating the gradients of ELBO.
In ZhuSuan, one can use this estimator by calling the method
:meth:`~zhusuan.variational.exclusive_kl.EvidenceLowerBoundObjective.sgvb`
of the class:`~zhusuan.variational.exclusive_kl.EvidenceLowerBoundObjective`
instance.
The code for this part is::

    # the surrogate cost for optimization
    lower_bound = model({'x': batch_x})


.. note::

    For readers who are interested, we provide a detailed explanation of the
    :meth:`~zhusuan.variational.exclusive_kl.EvidenceLowerBoundObjective.sgvb`
    estimator used here, though this is not required for you to use
    ZhuSuan's variational functionality.

    The key of SGVB estimator is a reparameterization trick, i.e., they
    reparameterize the random variable
    :math:`z\sim q_{\phi}(z|x) = \mathrm{N}(z|\mu_z(x;\phi), \sigma^2_z(x;\phi))`,
    as

    .. math::

        z = z(\epsilon; x, \phi) = \epsilon \sigma_z(x;\phi) + \mu_z(x;\phi),\; \epsilon\sim \mathrm{N}(0, I)

    In this way, the expectation can be rewritten with respect to
    :math:`\epsilon`:

    .. math::

        \mathcal{L}(\phi, \theta) &=
        \mathbb{E}_{z\sim q_{\phi}(z|x)} \left[\log p_{\theta}(x, z) - \log q_{\phi}(z|x)\right] \\
        &= \mathbb{E}_{\epsilon\sim \mathrm{N}(0, I)} \left[\log p_{\theta}(x, z(\epsilon; x, \phi)) -
        \log q_{\phi}(z(\epsilon; x, \phi)|x)\right]

    Thus the gradients with variational parameters :math:`\phi` can be
    directly moved into the expectation, enabling an unbiased low-variance
    Monte Carlo estimator:

    .. math::

        \nabla_{\phi} L(\phi, \theta) &=
        \mathbb{E}_{\epsilon\sim \mathrm{N}(0, I)} \nabla_{\phi} \left[\log p_{\theta}(x, z(\epsilon; x, \phi)) -
        \log q_{\phi}(z(\epsilon; x, \phi)|x)\right] \\
        &\approx \frac{1}{k}\sum_{i=1}^k \nabla_{\phi} \left[\log p_{\theta}(x, z(\epsilon_i; x, \phi)) -
        \log q_{\phi}(z(\epsilon_i; x, \phi)|x)\right]

    where :math:`\epsilon_i \sim \mathrm{N}(0, I)`

Now that we have had the cost, the next step is to do the stochastic gradient
descent.
Jittor provides many advanced
`optimizers <https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/jittor.optim.html>`_
that improves the plain SGD, among which Adam :cite:`vae-kingma2014adam`
is probably the most popular one in deep learning society.
Here we are going to use Jittor's Adam optimizer to do the learning::

    optimizer = jt.optim.Adam(model.parameters(), 1e-3)
    
    # During each iter
    optimizer.step(lower_bound)

Generate images
---------------

What we've done above is to define and learn the model. To see how it
performs, we would like to let it generate some images in the learning process.
We put the Var ``x_mean``  in the `cache` of ``Generator`` to keep track of it. ::

    class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        ...

    def execute(self, observed):
        ...
        x_probs = self.act2_(self.fc2_(self.act2(self.fc2(self.act1(self.fc1(z))))))
        self.cache['x_mean'] = x_probs
        self.sn('Bernoulli',
                name='x',
                probs=x_probs,
                reduce_mean_dims=[0],
                reduce_sum_dims=[1])
        ...

so that we can easily access it from a
:class:`~zhusuan.framework.bn.BayesianNet` instance.
For random generations, no observation about the model is made, so we
pass an empty observation to the model and get the generated sample by the ``cache['x_mean']`` of
``Generator``::

    cache = generator({}).cache
    sample_gen = cache['x_mean']

Run gradient descent
--------------------

Now, everything is good before a run.
So we could just run the training loop,
print statistics, and write generated images to disk using Jittor::

    for epoch in range(epoch_size):
        for step in range(num_batches):
            x = jt.array(x_train[step * batch_size:min((step + 1) * batch_size, len_)])
            x = jt.reshape(x, [-1, x_dim])
            if x.shape[0] != batch_size:
                break
            loss = model({'x': x})
            optimizer.step(loss)
            if (step + 1) % 100 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, epoch_size, step + 1, num_batches,
                                                                        float(loss.numpy())))
    
    batch_x = x_test[0:64]
    batch_x = jt.array(batch_x)

    cache = generator({}).cache
    sample_gen = cache['x_mean'].numpy()


Below is a sample image of random generations from the model.
Keep watching them and have fun :)

.. image:: ../_static/images/vae_mnist.png
    :align: center
    :width: 25%

.. rubric:: References

.. bibliography:: ../refs.bib
    :style: unsrtalpha
    :labelprefix: VAE
    :keyprefix: vae-
