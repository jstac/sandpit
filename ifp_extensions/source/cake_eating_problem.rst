

.. highlight:: python3

**********************************************
Cake Eating I: Introduction to Optimal Saving
**********************************************

.. contents:: :depth: 2



Overview
========


In this lecture we introduce a simple "cake eating" problem.

The intertemporal problem is: how much to enjoy today and how much to leave
for the future?

All though the topic sounds trivial, this kind of trade-off between current
and future utility is at the heart of many savings and consumption problems.

Once we master the ideas in this simple environment, we will apply them to
progressively more challenging---and useful---problems.

The main tool we will use to solve the cake eating problem is dynamic programming.

This topic is an excellent way to build dynamic programming skills.

Although not essential, readers will find it helpful to review the following
lectures before reading this one:

* The :doc:`shortest paths lecture <short_path>`
* The :doc:`basic McCall model <mccall_model>`
* The :doc:`McCall model with separation <mccall_model_with_separation>`
* The :doc:`McCall model with separation and a continuous wage distribution <mccall_fitted_vfi>` 

In what follows, we require the following imports:


.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline



The Model
==================

We operating on a infinite time horizon :math:`t=0, 1, 2, 3..`

At :math:`t=0` the agent is given a complete cake with size :math:`\bar y`.

Let :math:`y_t` denote the size of the cake at the beginning of each period,
so that, in particular, :math:`y_0=\bar y`.

We choose how much of the cake to eat in any given period :math:`t`.

We assume that consuming quantity :math:`c` of the cake gives the agent utility :math:`u(c)`.

We adopt the CRRA utility function

.. math::
    u(c) = \left\{
            \begin{array}{ll}
                \frac{c^{1-\gamma}}{1-\gamma}& \quad \text{for all}\ \gamma \neq 1 \text{and } \gamma\geq 0\\
                \ln(c) & \quad \gamma = 1
            \end{array}
        \right.


After choosing to consume :math:`c_t` of the cake in period :math:`t` there is

.. math::
    y_{t+1} = y_t - c_t 

left in period :math:`t+1`.

Future cake consumption utility is discounted according to :math:`\beta\in(0, 1)`.

In particular, consumption of :math:`c` units :math:`t` periods hence has present value :math:`\beta^t u(c)`

The agent's problem can be written as

.. math::
    \max_{\{c_t\}} \sum_{t=0}^\infty \beta^t u(c_t)

subject to

.. math::
    y_{t+1} = y_t - c_t 
    \quad \text{and} \quad
    0\leq c_t\leq y_t
    :label: cake_feasible

for all :math:`t`.


A consumption path :math:`\{c_t\}` satisfying :eq:`cake_feasible` where
:math:`y_0 = \bar y` is called **feasible**.


Trade-Off
---------

The key trade-off is this:

* Delaying consumption is costly because of the discount factor.

* But delaying some consumption is also attractive because :math:`u` is concave.


The concavity of :math:`u` implies that the consumer gains value from
*consumption smoothing*, which means spreading consumption out over time.

This is because concavity implies diminishing marginal utility---a progressively smaller gain in utility for each additional spoonful of cake consumed within one period.


Intuition
---------

The reasoning given above suggests that the discount factor :math:`\beta` and the curvature parameter :math:`\gamma` will play a key role in determining the rate of consumption.

Here's an educated guess as to what impact these parameters will have.

First, higher :math:`\beta` implies less discounting, which should reduce
the rate of consumption.

Second, higher :math:`\gamma` implies that marginal utility :math:`u'(c) =
c^{-\gamma}` falls faster with :math:`c`.

This suggests more smoothing, and hence a lower rate of consumption.

In summary, we expect the rate of consumption to be *decreasing in both
parameters*.

Let's see if this is true.




The Value Function
==================

The first step of our dynamic programming treatment is to obtain the Bellman
equation.

The next step is to use it to calculate the solution.


The Bellman Equation
--------------------

To this end, we let :math:`v(y)` be maximum lifetime utility attainable from
the current time when :math:`y` units of cake are left.

That is,

.. math::
    v(y) = \max \sum_{t=0}^{\infty} \beta^t u(c_t) 
    :label: value_fun

where the maximization is over all paths :math:`\{ c_t \}` that are feasible
from :math:`y_0 = y`.

At this point, we do not have an expression for :math:`v`, but we can still
make inferences about it.

For example, as was the case with the :doc:`McCall model <mccall_model>`, the
value function will satisfy a version of the *Bellman equation*.

In the present case, this equation states that :math:`v` satisfies 

.. math::
    :label: bellman

    v(y) = \max_{0\leq c \leq y} \{u(c) + \beta v(y-c)\}
    \quad \text{for any given } y \geq 0.

The intuition here is essentially the same it was for the McCall model.

Suppose that the current size of the cake is :math:`y`.

choosing :math:`c` optimally means trading off current vs future rewards.

Current rewards from choice :math:`c` are just :math:`u(c)`.

Future rewards, assuming optimal behavior, are :math:`v(y-c)`.

These are the two terms on the right hand side of :eq:`bellman`, after discounting.

If :math:`c` is chosen optimally using this strategy, then we obtain maximal
lifetime rewards from our current state :math:`y`.

Hence, :math:`v(y)` equals the right hand side of :eq:`bellman`, as claimed.



Foo
===


The function defined below computes the analytical solution of a given ``CakeEating`` instance.

.. code-block:: python3

    def v_star(ce):

        β, γ = ce.β, ce.γ
        y_grid = ce.y_grid
        u = ce.u

        a = β ** (1 / γ)
        x = 1 - a
        z = u(y_grid)

        return z / x ** γ

.. code-block:: python3

    v_analytical = v_star(ce)

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(y_grid, v_analytical, label='value function')
    ax.set_ylabel('$v(x)$', fontsize=12)
    ax.set_xlabel('$x$', fontsize=12)
    ax.legend()
    plt.show()


.. math::
    c^*_t = \sigma(x_t) = \arg \max_{c_t} \{u(c_t) + \beta v(x_t - c_t)\}


The analytical optimal policy function in this cake eating problem is

.. math::
    c^* = \left(1-\beta^\frac{1}{\gamma}\right)y


.. code-block:: python3

    def c_star(ce):

        β, γ = ce.β, ce.γ
        y_grid = ce.y_grid

        return (1 - β ** (1/γ)) * y_grid


.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(ce.y_grid, c_analytical, label='Analytical')
    ax.plot(ce.y_grid, c, label='Numerical')
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.legend()
    ax.set_title('Comparison between analytical and numerical optimal policies')
    plt.show()



The Euler Equation
==================

Roadmap.


Statement and Implications
--------------------------



Derivation I: An Intuitive Approach
-----------------------------------


In this section, we will
show you that a little more math helps us understand the intertemporal trade-offs of consumptions analytically.

We will show you two ways of deriving the optimality conditions.

First, we focus on the original optimization problem and maximize the discounted sum of utilities using Lagrange multiplier.




Derivation II: The Lagrangian Approach
--------------------------------------

Define the Lagrangian function as

.. math::

    \mathcal{L}=\sum_{t=0}^{\infty}\beta^{t}\left(u\left(c_{t}\right)+\lambda_{t}\left(x_{t}-c_{t}-x_{t+1}\right)\right)

Taking first derivatives with respect to two sequences of control variables :math:`\{c_t\}_{t=0}^{\infty}` and
:math:`\{x_{t+1}\}_{t=0}^{\infty}`, we have

.. math::

    u^{\prime}\left(c_{t}\right)-\lambda_{t}=0 \quad \text{for all} \ t \\
    \lambda_{t}-\beta\lambda_{t+1}=0 \quad \text{for all} \ t

when the consumptions are optimal. Combining these two first order conditions together gives us the
following equation for optimal consumptions today and tomorrow

.. math::
    :label: euler

    u^{\prime}\left(c^*_{t}\right)=\beta u^{\prime}\left(c^*_{t+1}\right)

which is what we call *Euler function*. Intuitively, this suggests that if :math:`\{c^*_t\}_{t=0}^{\infty}` is the optimal
consumption sequence, then the marginal utility of consuming *one more unit* of cake today equals to the discounted
marginal utility of consuming *one more unit* of cake tomorrow.


Derivation III: Using the Bellman Equation
------------------------------------------

The other way of deriving the Euler equation is to use the Bellman equation :eq:`bellman`. Since the Bellman equation is recursive,
we can focus on finding the optimal :math:`c_t^*` given :math:`x_t` instead of finding :math:`\{c^*_t\}_{t=0}^{\infty}` as a whole.

Taking first derivative with respect to :math:`c_t`, we get

.. math::
    :label: bellman_FOC

    u^{\prime}\left(c_{t}\right)=\beta V^{\prime}\left(x_{t+1}\right).

To know what :math:`V^{\prime}\left(x_{t+1}\right)` is, we first define the right hand side of the Bellman equation
as :math:`f\left(c_t,x_t\right)` and therefore

.. math::
    :label: bellman_equality

    V\left(x_{t}\right) = f\left(c_{t}^{*},x_{t}\right)

Taking differential on both sides of :eq:`bellman_equality` at :math:`c_t=c_t^*`, we have

.. math::
    dV\left(x_{t}\right) = df\left(c_{t},x_{t}\right)\bigg|_{c_{t}=c_{t}^{*}}
    =\left(\frac{\partial f\left(c_{t},x_{t}\right)}{\partial c_{t}}dc_{t}+\frac{\partial f\left(c_{t},x_{t}\right)}{\partial x_{t}}dx_{t}\right)\bigg|_{c_{t}=c_{t}^{*}}

Note that :math:`f\left(c_{t},x_{t}\right)` is maximized at :math:`c^*_t`, which implies :math:`\frac{\partial f\left(c_{t},x_{t}\right)}{\partial c_{t}}\big|_{c_{t}=c_{t}^{*}}=0` and

.. math::
    
    dV\left(x_{t}\right)=\frac{\partial f\left(c_{t},x_{t}\right)}{\partial x_{t}}dx_{t}=\beta V^{\prime}\left(x_{t+1}\right)dx_{t}

which is a result of *Envelope Theorem*. Dividing both sides by :math:`dx_{t}` gives us

.. math::
    :label: bellman_envelope

    V^{\prime}\left(x_{t}\right)=\beta V^{\prime}\left(x_{t+1}\right)

We can substitute :math:`\beta V^{\prime}\left(x_{t+1}\right)` in :eq:`bellman_FOC` using :eq:`bellman_envelope`,

.. math::
    :label: bellman_v_prime

    u^{\prime}\left(c_{t}\right)=V^{\prime}\left(x_{t}\right)

and we can derive the Euler equation again using :eq:`bellman_v_prime` and :eq:`bellman_FOC`.

It is interesting to observe the connection between methods of Lagrange multiplier and Bellman equation, which is

.. math::
    
    V^{\prime}\left(x_{t}\right)=\lambda_{t}

This will be much more clear if we think about the intuition behind these two terms: they both represent
the change in the optimal value of the objective function due to the relaxation of a given constraint (in this
case, it is one additional unit of cake for free). :math:`\lambda_{t}` is usually referred to as *shadow price*
in economics or *costate variable* in control theory.


Exercises
=========

Exercise 1
------------

Prove that the optimal policy function is linear and there exists an postive :math:`\theta` such that :math:`c_t^*=\theta y_t`


(might change this to verify the value function above is the value function?)

Exercise 2
-----------

In our example above we assumed that the production function of captial was :math:`f(k)=k` because we were talking specficially about a cake.

Derive the Euler equation.



Solutions
==========


Exercise 1
-----------

Suppose that the optimal policy is :math:`c_t^*=\theta y_t`

then

.. math::
    y_{t+1}=y_t(1-\theta)

which means

.. math::
    y_t = y_{0}(1-\theta)^t


Thus the optimal value function is.

.. math::
    v^*(y_0) = \sum_{t=0}^{\infty} \beta^{t} u(c_t)\\
    v^*(y_0) = \sum_{t=0}^{\infty} \beta^{t} u(\theta y_{t})\\
    v^*(y_0) = \sum_{t=0}^{\infty} \beta^{t} u\left(\theta y_{0}(1-\theta)^t\right)\\
    v^*(y_0) = \sum_{t=0}^{\infty} \theta^{1-\gamma}\beta^{t} (1-\theta)^{t(1-\gamma)}u(y_{0})\\
    v^*(y_0) = \frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}u(y_{0})


Now with the optimal form of the value funciton we can impliment it in to the bellman equation.

.. math::
    v(y) = \max_{0\leq c\leq y}
        \left\{
            u(c) + 
            \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot u(y-c)
        \right\}\\
    v(y) = \max_{0\leq c\leq y}
    \left\{
        \frac{c^{1-\gamma}}{1-\gamma} + 
        \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot\frac{(y-c)^{1-\gamma}}{1-\gamma}
    \right\}


taking the F.O.C we have

.. math::
    c^{-\gamma} + \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot(y-c)^{-\gamma}(-1) = 0\\
    c^{-\gamma} = \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot(y-c)^{-\gamma}


with :math:`c = \theta y` we get

.. math::
    \left(\theta y\right)^{-\gamma} =  \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot(y(1-\theta))^{-
    \gamma}

With some re-arrangment we get

.. math::
    \theta = 1-\beta^{\frac{1}{\gamma}}


this gives the optimal policy of

.. math::
    c_t^* = \left(1-\beta^{\frac{1}{\gamma}}\right)y_t


substituting :math:`\theta` into the value function above gives.

.. math::
    v^*(y_t) = \frac{\left(1-\beta^{\frac{1}{\gamma}}\right)^{1-\gamma}}{1-\beta\left(\beta^{\frac{{1-\gamma}}{\gamma}}\right)}u(y_{t})\\


.. math::
    v^*(y_t) = \left(1-\beta^\frac{1}{\gamma}\right)^{-\gamma}u(y_t)


Now we must verify that this value function is a fixed point, using the bellman equation.

.. math::
    v(y) = \max_{0\leq c\leq y}
        \left\{
            u(c) +
            \beta\left(1-\beta^\frac{1}{\gamma}\right)^{-\gamma}u(y-c)
        \right\}\\

taking the F.O.C we have

.. math::
    c^{-\gamma} - \beta\left(1-\beta^\frac{1}{\gamma}\right)^{-\gamma}(y-c)^{-\gamma} = 0

Rearranging gives

.. math::
    c_t^* = \left(1-\beta^{\frac{1}{\gamma}}\right)y_t




Exercise 2
----------

To be added.
