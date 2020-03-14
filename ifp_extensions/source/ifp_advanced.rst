.. _ifp:

.. include:: /_static/includes/header.raw

.. highlight:: python3

************************************************************************************
:index:`The Income Fluctuation Problem with Stochastic Returns and Discounting`
************************************************************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install interpolation


Overview
========

In this lecture, we continue our study of the income fluctuation problem initiated in :doc:`an earlier lecture <ifp>`.  

(See that lecture for background and intuition.)

In this lecture, we introduce two significant extensions to the version we first considered:

1. Returns on assets are allowed to be stochastic.

2. The discount factor of the agent is allowed to vary with the state.

The first modification makes our model more realistic in terms of content and predictions.

The second modification allows us to encompass recent quantitative work using heterogeneous agent models, which often adopts this feature.

* See, for example, :cite:`krusell1998income`,
  :cite:`cao2016existence`, :cite:`hubmer2018comprehensive` and
  :cite:`hills2018fiscal`.

In addition to these generalizations of the income fluctuation problem, we also introduce an alternative method for solving such problems computationally.

This method is sometimes called **time iteration** and involves iterating on the Euler equation.

(In one of our lectures, we use similar ideas to solve :doc:`the optimal growth model <coleman_policy_iter>`.)

This method can be shown to be globally convergent under mild assumptions, even when utility is unbounded (both above and below).

We will make use of the following imports:

.. code-block:: ipython

    import numpy as np
    from quantecon.optimize import brent_max, brentq
    from interpolation import interp
    from numba import njit
    import matplotlib.pyplot as plt
    %matplotlib inline
    from quantecon import MarkovChain



The Household Problem
=====================

.. index::
    single: Optimal Savings; Problem


Set Up
------

A household chooses a consumption-asset path :math:`\{(c_t, a_t)\}` to
maximize

.. math::
    \mathbb E \left\{ 
                \sum_{t=0}^\infty 
                \left(\prod_{i=0}^t \beta_i \right) u(c_t)
             \right\}   
    :label: trans_at

subject to 

.. math::
    a_{t+1} = R_{t+1} (a_t - c_t) + Y_{t+1}
    \; \text{ and } \;
    0 \leq c_t \leq a_t, 
    :label: trans_at2

with initial condition :math:`(a_0, Z_0)=(a,z)` treated as given.

Note that 

* :math:`\{R_t\}_{t \geq 1}`, the gross rate of return on wealth,
is allowed to be stochastic.

* the constant discount factor :math:`\beta` has been replaced by a discount factor process :math:`\{\beta_t\}_{t \geq 0}`  with :math:`\beta_0=1`. 

The sequence :math:`\{Y_t \}_{t \geq 1}` is non-financial income. 

The stochastic components of the problem obey

.. math:: 
    \beta_t = \beta \left( Z_t, \epsilon_t \right), 
      \quad 
    R_t = R \left( Z_t, \zeta_t \right),
      \quad 
    Y_t = Y \left( Z_t, \eta_t \right),
    :label: eq:RY_func

where

* the maps :math:`\beta`, :math:`R` and :math:`Y` are time-invariant, nonnegative Borel-measurable functions and 

* :math:`\{Z_t\}_{t \geq 0}` is an irreducible time-homogeneous Markov chain on a finite set :math:`\mathsf Z`  

Let :math:`P(z, \hat z)` represent the probability of this exogenous state process transitioning from :math:`z` to :math:`\hat z` in one step.

The innovation processes :math:`\{\epsilon_t\}`, :math:`\{\zeta_t\}` and :math:`\{\eta_t\}` are IID and independent of each other.

The utility function :math:`u` maps :math:`\mathbb R+` to :math:`\{ - \infty \} \cup \mathbb R`, is twice differentiable on :math:`(0, \infty)`, satisfies :math:`u' > 0` and :math:`u'' < 0` everywhere
on :math:`(0, \infty)`, and that :math:`u'(c) \to \infty` as :math:`c \to 0` and :math:`u'(c) < 1`
as :math:`c \to \infty`. 

Regarding notation, in what follows :math:`\mathbb E_z \hat X` means expectation of next period value :math:`\hat X` given current value :math:`Z = z`.

For example, 

.. math::
    \mathbb E_z \hat R 
    := \mathbb E 
    \left[ R_{t+1} \, | \, Z_t = z \right]


Assumptions
-----------

We need restrictions to ensure that the objective :eq:`trans_at` is finite and
the solution methods described below converge.

We assume in all of what follows that the discount factor process satisfies 

.. math::
    G_\beta < 1
    \quad \text{where} \quad 
    G_\beta := \lim_{n \to \infty} 
    \left(\mathbb E \prod_{t=1}^n \beta_t \right)^{1/n}
    :label: fpbc

This assumption turns out to be the most natural extension of the standard condition :math:`\beta < 1` from the constant discount case.  

(You can easily confirm that :math:`G_\beta = \beta` in that non-stochastic setting.)

We also need to ensure that the present discounted value of wealth
does not grow too quickly. 

When :math:`\{R_t\}` and :math:`\{ \beta_t\}` are constant at
values :math:`R` and :math:`\beta`, the standard restriction from the existing literature is :math:`\beta R < 1`.  

A natural generalization is

.. math::
    G_{\beta R} < 1
    \quad \text{where} \quad 
    G_{\beta R} := \lim_{n \to \infty} 
    \left(\mathbb E \prod_{t=1}^n \beta_t R_t \right)^{1/n}
    :label: fpbc2

Finally, we impose some routine technical restrictions on non-financial income.

.. math::
    \mathbb E \, Y < \infty \text{ and } \mathbb E \, u'(Y) < \infty
    \label{a:y0}

One relatively simple setting where all these restrictions are satisfied is the CRRA environment of :cite:`benhabib2015`.

See :cite:`ma2020income` for more details on the assumptions given above.



Optimality
----------

The state space for :math:`\{(a_t, Z_t) \}_{t \geq 0}` is taken to be :math:`\mathbb S:= (0, \infty) \times \mathsf Z`.

A **feasible policy** is a Borel measurable function 
:math:`c \colon \mathbb S \to \mathbb R` with :math:`0 \leq c(a,z) \leq a` for all 
:math:`(a,z) \in \mathbb S`. 


A feasible policy :math:`c` and initial condition 
:math:`(a,z) \in \mathbb S` generate an asset path 
:math:`\{ a_t\}_{t \geq 0}` via :eq:`trans_at2` when 
:math:`c_t = c (a_t, Z_t)` and :math:`(a_0, Z_0) = (a,z)`. 

The lifetime value of policy :math:`c` is

.. math::
    V_c (a,z) = \mathbb E_{a,z} \sum_{t = 0}^\infty 
        \beta_0 \cdots \beta_t u \left[ c (a_t, Z_t) \right]
    :label: Vc

where :math:`\{ a_t\}` is the asset path generated by :math:`(c,(a,z))`. 

A feasible policy :math:`c^*` is called **optimal** if :math:`V_c \leq V_{c^*}` on :math:`\mathbb S` for any feasible policy :math:`c`. 

A feasible policy is said to satisfy the **first order optimality  condition** if, for all :math:`(a,z) \in \mathbb S`, 

.. math::
    \left( u' \circ c \right)(a,z) = 
    \max \left\{ 
            \mathbb E_z \, \hat{\beta} \hat{R} 
               \left( u' \circ c \right) 
               \left( \hat{R} \left[ a - c(a,z)\right] + \hat{Y}, 
                      \, \hat{Z} \right),
            u'(a)
         \right\}


This is a version of the Euler equation discussed in [add suitable reference]


(The maximization over two values is due to the possibility of corner solutions, which can occur when non-financial income is additive.)

Let :math:`\mathscr C` be the space of continuous functions :math:`c \colon \mathbb S \to \mathbb R` such that :math:`c` is increasing in the first argument, :math:`0 < c(a,z) \leq a` for all
:math:`(a,z) \in \mathbb S`, and


.. math::
   \sup_{(a,z) \in \mathbb S} 
   \left| (u' \circ c)(a,z) - u'(a) \right| < \infty
   :label: ifpC4

The following is proved in :cite:`ma2020income`:

**Theorem.** If :math:`c \in \mathscr C` and the first order optimality condition holds, then :math:`c` is an optimal policy.

Now our task is to find a feasible policy satisfying the first order
optimality condition.

To do this we use time iteration, as discussed below.


.. _ifp_computation:

Solution Algorithm
==================

.. index::
    single: Optimal Savings; Computation

A Time Iteration Operator
-------------------------

First we introduce the time iteration operator :math:`K` defined 
as follows: 

For fixed :math:`c \in \mathscr C` and :math:`(a,z) \in \mathbb S`, the value :math:`Kc(a,z)` of the 
function :math:`Kc` at :math:`(a,z)` is defined as the :math:`\xi \in (0,a]` that solves

.. math::
    u'(\xi) = 
    \max \left\{
              \mathbb E_z \, \hat{\beta} \hat{R}
                 (u' \circ c)[\hat{R}(a - \xi) + \hat{Y}, \, \hat{Z}], 
              \, u'(a)
           \right\}
    :label: k_opr



Convergence Properties
----------------------

When iterating in policy space, we need a way to compare two consumption policies. 

To this end, we pair :math:`\mathscr C` with the distance

.. math::
   \rho(c,d) 
   := \sup_{(a,z) \in \mathbb S} 
             \left| 
                 \left(u' \circ c \right)(a,z) - 
                 \left(u' \circ d \right)(a,z) 
             \right|,

which evaluates the maximal difference in terms of marginal utility. 

The benefit of this measure of distance is that, while elements of :math:`\mathscr C` are not generally bounded, :math:`\rho` is always finite under our assumptions.

In fact, it can be shown that 

1. :math:`(\mathscr C, \rho)` is a complete metric space,

2. there exists an integer :math:`n` such that :math:`K^n` is a contraction
   mapping on :math:`(\mathscr C, \rho)`, and

3. The unique fixed point of :math:`K` in :math:`\mathscr C` is 
   the unique optimal policy in :math:`\mathscr C`.

See :cite:`ma2020income`  for details.


We now have a clear path to successfully approximating the optimal policy:
choose some :math:`c \in \mathscr C` and then iterate with :math:`K` until
convergence (as measured by the distance :math:`\rho`)



Testing the Assumptions
-----------------------

[to be added.]


Implementation
==============

.. index::
    single: Optimal Savings; Programming Implementation





Exercises
=========

Repeat earlier IFP exercises with some new twists.
