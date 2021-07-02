.. ZhuSuan documentation master file, created by
   sphinx-quickstart on Wed Feb  8 15:01:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ZhuSuan-Jittor
==================

.. image:: _static/images/index.png
    :align: center

ZhuSuan-PaddlePaddle is a python probabilistic programming library for
**Bayesian deep learning**, which conjoins the complimentary advantages of
Bayesian methods and deep learning. ZhuSuan is built upon
`PaddlePaddle <https://www.paddlepaddle.org.cn/>`_. Unlike existing deep learning
libraries, which are mainly designed for deterministic neural networks and
supervised tasks, ZhuSuan-PaddlePaddle provides deep learning style primitives and
algorithms for building probabilistic models and applying Bayesian inference.
The supported inference algorithms include:

* Variational inference with programmable variational posteriors, various
  objectives and advanced gradient estimators (SGVB, SWI, etc.).
* Importance sampling for learning and evaluating models, with programmable
  proposals.
* MCMC samplers: Hamiltonian Monte Carlo (HMC) with parallel chains, and 
  Stochastic Gradient MCMC (sgmcmc).

.. toctree::
   :maxdepth: 2


Installation
------------

ZhuSuan-PaddlePaddle is still under development. Before the first stable release (1.0),
please clone the `GitHub repository <https://github.com/McGrady00H/Zhusuan-PaddlePaddle/>`_ and
run
::

   pip install .

in the main directory. This will install ZhuSuan-PaddlePaddle and its dependencies
automatically. Notice that currently ZhuSuan-PaddlePaddle is built on PaddlePaddle version 2.0rc.

If you are developing ZhuSuan-PaddlePaddle, you may want to install in an "editable" or
"develop" mode. Please refer to the Contributing section.

After installation, open your python console and type::

   >>> import zhusuan as zs

If no error occurs, you've successfully installed ZhuSuan.

.. Tutorial slides <https://docs.google.com/presentation/d/1Xqi-qFHciAdV9z1FHpGkUcHT-yugNVzwGX3MM74rMuM/edit?usp=sharing>
   tutorials/lntm

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/concepts
   tutorials/vae
   tutorials/bnn


.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/zhusuan.distributions
   api/zhusuan.framework
   api/zhusuan.variational
   api/zhusuan.mcmc
   api/zhusuan.evaluation

.. toctree::
   :maxdepth: 1
   :caption: Community

   contributing



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
