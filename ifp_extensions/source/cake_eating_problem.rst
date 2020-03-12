

.. highlight:: python3

****************************************************
Introduction To Optimal Growth: Cake Eating Problem
****************************************************

.. contents:: :depth: 2


In addition to what's in Anaconda, this lecture will require the following library:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade interpolation



Overview
========


In this lecture we will consider a simple optimal growth model with one agent and captial with no growth and perfect transformation into output. It is often referred to as the cake eating problem.

On the surface it may seem a simple problem.
How much of a cake to eat today and leave for tomorrow?

However it quickly becomes an interesting dynamic programming problem.

This problem is a good introduction to the stochastic optimal growth problem which we look at in later :doc:`lectures <optgrowth>`.


We require the following imports.


.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    from interpolation import interp
    from scipy.optimize import minimize_scalar

    %matplotlib inline

The Model
==================

We are operating on a infinite time horizen :math:`t=0,1,2,3...`

At :math:`t=0` the agent is given the perfect cake with size :math:`\bar{y}`, this is exogenously given.

At the beginning of each period the size of the cake is :math:`y_t`, with :math:`y_0=\bar{y}`.


We are asked to choose how much of the cake to eat in any given period :math:`t`.

We will assume that consuming :math:`c` of the cake gives the agent utility in the form of the CRRA utility function.

.. math::

    u(c) = \left\{
            \begin{array}{ll}
                \frac{c^{1-\gamma}}{1-\gamma}& \quad \text{for all}\ \gamma \neq 1 \text{and } \gamma\geq 0\\
                \ln(c) & \quad \gamma = 1
            \end{array}
        \right.





After choosing to consumed :math:`c_t` of the cake in period :math:`t` there is

.. math::

    y_{t+1} = y_t - c_t \ \text{for all}\ t



left in period :math:`t+1`.

Future cake consumption utility is discounted by :math:`\beta\in(0,1)`.

The agent's problem can be written as

.. math::

    \max_{\{c_t\}_{t = 0}^{t = \infty}} \sum_{t=0}^{\infty} \beta^t u(c_t)

subject to.

.. math::

    y_{t+1} = y_t - c_t \ \text{for all}\ t\text{,}\\
    0\leq c_t\leq y_t\ \text{for all}\ t\text{,}\\
    y_0=\bar{y}\text{.}


Trade-Off
----------------------------
Our agent is faced with the trade offs:
* Eating more today to avoid the discount factor.
* Eating more tomorrow with the agents preference aiming to smooth consumption, due to the risk aversion.

Value Function
================
We can break down the the agents problem in terms of these two trade-offs. 

As such we can set up the agent's problem to look like. 

.. math::
    :label: value_fun

    v(y_0) = \max_{\{c_t\}_{t = 0}^{t = \infty}\\0\leq c_{t}\leq y_{t}} \sum_{t=0}^{\infty} \beta^t u(c_t)

:math:`v(y_0)`, our value function, can be interpreted as the agents total lifetime utility when optimially choosing :math:`\{c_t\}_{t = 0}^{t = \infty}` for a given :math:`y_0`. 

Breaking :eq:`value_fun` down into the two trade-offs gives.

.. math::
    :label: tra_value_fun

    v(y_0) = \max_{0\leq c_0\leq y_0}
        \left\{
            u(c_0) + 
            \beta\max_{\{c_t\}_{t = 1}^{t = \infty}\\0\leq c_{t}\leq y_{t}} \sum_{t=1}^{\infty} \beta^{t-1} u(c_t)
        \right\}

The Bellman Equation
-----------------------

:eq:`tra_value_fun` yields the bellman equation.

.. math::

    v(y_0) = \max_{0\leq c_0\leq y_0}\{u(c_0) + \beta v(y_1)\}\\
    v(y_t) = \max_{0\leq c_t\leq y_t}\{u(c_t) + \beta v(y_{t+1})\}

Here :math:`y_t` is the state variable, with the transtion of the state variable depending on

.. math::
    :label: state_tran

    y_{t+1} = y_t - c_t \ \text{for all}\ t


As in previous dynamic programming lectures(LINK) we will use the Bellman operator $T$ to solve the Bellman equation.
Any fixed point of $T$ solves the Bellman equation and vice versa.

.. math::

    Tv(y) = \max_{0 \leq c \leq y}\{u(c) + \beta v(y')\}

From :eq:`state_tran` we can re-write this as

.. math::
    :label: bellman_val

    Tv(y) = \max_{0 \leq c \leq y}\{u(c) + \beta v(y - c)\}



Value Function Iteration
============================

In order to determine the value function we need to:

#. Take an arbitary intial guess of :math:`v'`.
#. Plug :math:`v'` into the right hand side of :eq:`bellman_val`, find and store $c$ and $v$.
#. Unless a condition is met, set :math:`v'=v` and go back to step 2.

As consumption choice :math:`c` is a continous variable, the state variable :math:`y` is continous. This makes things tricky.

In order to determine :math:`v` we have to store every :math:`v(y)` for every :math:`y\in [0,\bar{y}]`, which is difficult given there are infinitly many points.

To get around this we'll create a finite grid of different size cakes :math:`\bar{y}=y_0>y_1>y_2>...y_I>0` and determine the :math:`v` for each point on the grid and store them.

The process looks like:

#. Begin with an array of values :math:`\{ v_0, \ldots, v_I \}`  representing
   the values of some initial function $ v $ on the grid points :math:`\{ y_0, \ldots, y_I \}`.
#. Build a function :math:`\hat v` on the state space :math:`\mathbb R_+` by
   linear interpolation, based on these data points.
#. Obtain and record the value :math:`T \hat v(y_i)` on each grid point
   :math:`y_i` by repeatedly solving.
#. Unless some stopping condition is satisfied, set
   :math:`\{ v_0, \ldots, v_I \} = \{ T \hat v(y_0), \ldots, T \hat v(y_I) \}` and go to step 2.

In step 2 we'll use the same continuous piecewise linear interpolation strategy as is the previous :doc:`lecture <mccall_fitted_vfi>`

Implementation
=================
Firstly we need to be able to find both the maximum and the maximizer of the value function. However scipy only has a ``minimize_scalar`` function which finds the minimum and the minimizer of a function on a certain bound. 

In order find the maximum of the value function we have to take the negative of the value function and find its minimum and minimizer with ``minimize_scalar``.

The ``maximize`` function below, takes a function ``g`` and does just that.

.. code-block:: python3

    def maximize(g, a, b, args):
        """
        Maximize the function g over the interval [a, b].

        We use the fact that the maximizer of g on any interval is
        also the minimizer of -g.  The tuple args collects any extra
        arguments to g.

        Returns the maximal value and the maximizer.
        """

        objective = lambda x: -g(x, *args)
        result = minimize_scalar(objective, bounds=(a, b), method='bounded')
        maximizer, maximum = result.x, -result.fun
        return maximizer, maximum

We'll store the primitives such as :math:`\beta` and :math:`\gamma` in the class ``EatCake``. 

This class will also have a function which returns the right hand right of the bellman equation which needs to be maximized, which is the function that will run through the ``maximize`` function. 

.. code-block:: python3

    class EatCake:

        def __init__(self,
                    β=0.96,       # discount factor
                    γ=0.5,        # degree of relative risk aversion
                    y_grid_max=10,  # inital stock of capital Y
                    y_grid_size=120):

            self.β, self.γ = β, γ

            # Set up grid
            self.y_grid = np.linspace(1e-04, y_grid_max, y_grid_size)
            
        # Utility function
        def u(self, c):
            
            if self.γ == 1:
                return np.log(c)
            else:
                return (c**(1 - self.γ)) / (1 - self.γ)


        def state_action_value(self, c, y, v_array):
            """
            Right hand side of the Bellman equation.
            """

            u, β = self.u, self.β

            v = lambda x: interp(self.y_grid, v_array, x)

            return u(c) + β * v(y - c)


We now use the Bellman operator to determine the value function for each point on the grid.

.. code-block:: python3

    def T(ec, v):
        """
        The Bellman operator.  Updates the guess of the value function.

        * ec is an instance of EatCake
        * v is an array representing a guess of the value function

        """
        v_new = np.empty_like(v)

        for i in range(len(ec.y_grid)):
            y = ec.y_grid[i]
            # Maximize RHS of Bellman equation at state y
            v_max = maximize(ec.state_action_value, 1e-10, y, (y, v))[1]
            v_new[i] = v_max


        return v_new


Next let’s create an instance of the model and assign it to the variable `ec`.

.. code-block:: python3

    ec = EatCake()

Now lets see the iteration of the value function in action. The intial guess will be $0$ for every $y$ point. 

We should see that the value functions converge to a single function as the intial guess is updated.

.. code-block:: python3

    y_grid = ec.y_grid
    v = np.zeros(len(y_grid))  # Initial guess
    n = 35                   # Number of iterations

    fig, ax = plt.subplots()

    ax.plot(y_grid, v, color=plt.cm.jet(0),
            lw=2, alpha=0.6, label='Initial condition')

    for i in range(n):
        v = T(ec, v)  # Apply the Bellman operator
        ax.plot(y_grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

    ax.legend()
    ax.set_ylabel('$v(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)

    plt.show()


We can return the converged function by making a function ``value_fun`` that returns the value function once there has been sufficent convergence.


.. code-block:: python3

    def value_fun(ec,
            tol=1e-4,
            max_iter=1000,
            verbose=True,
            print_skip=25):

        # Set up loop
        v = np.zeros(len(ec.y_grid)) # Initial condition
        i = 0
        error = tol + 1

        while i < max_iter and error > tol:
            v_new = T(ec, v)
            error = np.max(np.abs(v - v_new))
            i += 1
            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            v = v_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return v_new

.. code-block:: python3

    v_solution = value_fun(ec)

Now we can plot and see what the value and policy functions look like. 

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(y_grid, v_solution, lw=2, alpha=0.6,
            label='Approximate value function')

    ax.set_ylabel('$v^*(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)

    ax.legend()
    plt.show()

Analytical Solution
====================

This problem has an analytical solution.
We can check the results from our code matches up with the analytical solution.

For a CRRA utility function the solution for the value function is:

.. math::
    v^*(y) = \left(1-\beta^{\frac{1}{\gamma}}\right)^{-\gamma}u(y)

We leave the proof as an exercise for the reader.

The function below returns the value function for a given y

.. code-block:: python3

    def v_star(y, u=ec.u, β=ec.β, γ=ec.γ):
    
        a = β**(1/γ)
        x = 1-a
        z = u(y)
        
        return z*x**(-γ)

    v_star = v_star(u=ec.u, y=y_grid)

.. code-block:: python3

    fig, ax = plt.subplots()


    ax.plot(y_grid, v_star, lw=2,
            alpha=0.6, label='True value function')
            
    ax.plot(y_grid, v_solution, lw=2,
            alpha=0.6, label='Approximate value function')

    ax.set_ylabel('$v^*(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    ax.legend()
    plt.show()

Hooray! It looks like our code is pretty close to the analytical solution.

Policy Function
===============

Now we have the converged value function it is easy for us to determine the optimal consumption of the cake. 

This optimal choice of :math:`\{c_t\}_{t = 0}^{t = \infty}` is often referred to as the agents policy function. 

It is the mapping from the state space to the action space which maximises the agents lifetime utility.

.. math::
    c^*_t = \sigma(y_t) = \arg \max_{y_t} v(y_t)


We can you the converged value function in the right hand side of the bellman equation and find the maximizer at every point on the grid. This will return us the optimal consumption :math:`c^*` for a given state :math:`y`.

Below is a similar function to the bellman operator function above. It finds the maximizer for every point on the grid. 

.. code-block:: python3

    def σ(ec, v):
        """
        The Bellman operator.  Updates the guess of the value function
        and also computes a c_new policy.

        * ec is an instance of EatCake
        * v is an array representing a guess of the value function

        """
        c_new = np.empty_like(v)


        for i in range(len(ec.y_grid)):
            y = ec.y_grid[i]
            # Maximize RHS of Bellman equation at state y
            c_max = maximize(ec.state_action_value, 1e-10, y, (y, v))[0]
            c_new[i] = c_max


        return c_new

We can use the converged value function in ``σ`` and get the immediate solution without iterating.

.. code-block:: python3

    v_greedy = σ(ec, v_solution)  


    fig, ax = plt.subplots()


    ax.plot(y_grid, v_greedy, lw=2, alpha=0.6)

    ax.set_ylabel('$\sigma(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)

    plt.show()

.. _pol_an:
Analytical Solution
--------------------
We can compare the policy function our code gives us with the analytical solution. 

For this problem the analytical solution is

.. math::
    c^* = \left(1-\beta^\frac{1}{\gamma}\right)y

.. code-block:: python3

    def c_star(y, β=ec.β, γ=ec.γ):
        
        return y*(1-β**(1/γ))
    c_star = c_star(y_grid)

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(y_grid, c_star, lw=2, alpha=0.6, label='True policy')

    ax.plot(y_grid, v_greedy, lw=2, alpha=0.6, label='Approximate policy')

    ax.legend()
    ax.set_ylabel('$\sigma(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    plt.show()

Exercises
==============

Exercise 1
------------
Prove that the optimal policy function is linear and there exists an postive :math:`\theta` such that :math:`c_t^*=\theta y_t`


(might change this to verify the value function above is the value function?)

Exercise 2
-----------
In our example above we assumed that the production function of captial was :math:`f(k)=k` because we were talking specficially about a cake.

Now assume that the production function is in the form of :math:`f(k)=k^{\alpha}` where :math:`\alpha\in(0,1)`

Make the required changes to the code above and plot the value and policy functions. Comment on the change in the policy function. 

Note :math:`y_t=f(k_t)`

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

re-arraning this gives

.. math::
    c_t^* = \left(1-\beta^{\frac{1}{\gamma}}\right)y_t



Exercise 2
----------
First note the resource contraint binds

.. math::
    k_{t+1}=y_t-c_t\ \text{for all}\ t


from the production function output tomorrow is.

.. math::
    y_{t+1}=f(y_t-c_t)\ \text{for all}\ t


We need to create a class to hold our primitives and return the right hand side of the bellman equation.


.. code-block:: python3

    class OptimalGrowth:

        def __init__(self,
                    β=0.96,       # discount factor
                    γ=0.5,        # degree of relative risk aversion
                    α=0.4,
                    y_grid_max=10,  # inital stock of capital Y
                    y_grid_size=120):

            self.β, self.γ, self.α = β, γ, α

            # Set up grid
            self.y_grid = np.linspace(1e-04, y_grid_max, y_grid_size)
            
        def u(self, c):
            
            if self.γ == 1:
                return np.log(c)
            else:
                return (c**(1 - self.γ)) / (1 - self.γ)
        def f(self, k):
            return k**self.α

        def state_action_value(self, c, y, v_array):

            u, f, β = self.u, self.f, self.β

            v = lambda x: interp(self.y_grid, v_array, x)

            return u(c) + β * v(f(y - c))

.. code-block:: python3

    og = OptimalGrowth()

Now I'll graph the iterations in of the value function.

.. code-block:: python3

    v = value_fun(og, verbose=False)

    fig, ax = plt.subplots()


    ax.plot(y_grid, v, lw=2, alpha=0.6)
    ax.set_ylabel('v*(y)', fontsize=12)
    ax.set_xlabel('y', fontsize=12)

    plt.show()


.. code-block:: python3

    c_new = σ(og, v)

    fig, ax = plt.subplots()

    ax.plot(y_grid, c_new,lw=2, alpha=0.6)

    ax.set_ylabel('$\sigma(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    plt.show()

The slope of the policy function has increased from what we saw :ref:`above <pol_an>`.

Because there is diminishing returns to capital and there is no growth in captial. The agent wants to eat more today to avoid the shrinking of the cake tomorrow. 