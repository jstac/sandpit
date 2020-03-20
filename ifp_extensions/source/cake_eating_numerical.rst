.. highlight:: python3

*********************************
Cake Eating II: Numerical Methods
*********************************

.. contents:: :depth: 2


In addition to what's in Anaconda, this lecture will require the following library:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade interpolation



Overview
========


In this lecture we continue the study of :doc:`the cake eating problem
<cake_eating_problem>`.

The aim of this lecture is to solve the cake eating problem using numerical
methods.


At first this might appear unnecessary, since we already obtained the optimal
policy analytically.

[add link]

However, the cake eating problem is too simple to be useful without
modifications.

Once we start modifying the problem, numerical methods become essential.

Hence it makes sense to introduce numerical methods now, and test them on this
simple problem.

Because we know the analytical solution, we can confirm that the numerical
methods are sound.

This will give us confidence in the methods before we shift to
generalizations.

We will use the following imports:


.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from interpolation import interp
    from scipy.optimize import minimize_scalar



Reviewing The Model
===================


Recall that the problem is to solve 

.. math::
    \max_{\{c_t\}} \sum_{t=0}^\infty \beta^t u(c_t)

subject to

.. math::
    x_{t+1} = x_t - c_t 
    \quad \text{and} \quad
    0\leq c_t \leq x_t
    :label: cake_feasible

for all :math:`t`.

We obtained the Bellman equation

.. math::
    :label: bellman

    v(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
    \quad \text{for any given } x \geq 0.

as a restriction on the value function :math:`v`.

We found an analytical solution of the form 

.. math::
    v^*(x) = \left(1-\beta^{\frac{1}{\gamma}}\right)^{-\gamma} u(x)

The optimal consumption policy was then shown to satisfy

.. math::
    \sigma^*(x) = \left(1-\beta^\frac{1}{\gamma}\right) x

We also pointed out that this policy satisfies the Euler equation 

.. math::
    :label: euler

    u' (\sigma(x))
        = \beta u' (x - \sigma(x)).



Value Function Iteration
========================



Computationally, we will define a Bellman operator :math:`T` as in the previous dynamic programming lectures to solve for `v`.

.. math::

    Tv(y) = \max_{0 \leq c \leq y}\{u(c) + \beta v(y')\}

By contraction mapping theorem, given any intial guess of :math:`v`, this operation will converge to a unique fixed point, which is
the correct solution.

Incorporating the transition law of the state variable :math:`y_{t+1} = y_t - c_t` into the Bellman equation, we have

.. math::
    :label: bellman_val

    Tv(y) = \max_{0 \leq c \leq y}\{u(c) + \beta v(y - c)\}



In order to determine the value function we need to:

#. Take an arbitary intial guess of :math:`v'`.
#. Plug :math:`v'` into the right hand side of :eq:`bellman_val`, find and store :math:`c` and :math:`v`.
#. Unless a condition is met, set :math:`v'=v` and go back to step 2.

As consumption choice :math:`c` is a continous variable, the state variable :math:`y` is continous. This makes things tricky.

In order to determine :math:`v` we have to store :math:`v(y)` for every :math:`y\in [0,\bar{y}]`, which is difficult given there are infinitly many points.

To get around this we'll create a finite grid of different size cakes :math:`\bar{y}=y_0>y_1>y_2>...y_I>0` and determine the :math:`v` for each point on the grid and store them.

The process looks like:

#. Begin with an array of values :math:`\{ v_0, \ldots, v_I \}`  representing
   the values of some initial function :math:`v` on the grid points :math:`\{ y_0, \ldots, y_I \}`.
#. Build a function :math:`\hat v` on the state space :math:`\mathbb R_+` by
   linear interpolation, based on these data points.
#. Obtain and record the value :math:`T \hat v(y_i)` on each grid point
   :math:`y_i` by repeatedly solving.
#. Unless some stopping condition is satisfied, set
   :math:`\{ v_0, \ldots, v_I \} = \{ T \hat v(y_0), \ldots, T \hat v(y_I) \}` and go to step 2.

In step 2 we'll use the same continuous piecewise linear interpolation strategy as is the previous :doc:`lecture <mccall_fitted_vfi>`




Implementation
--------------

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

We'll store the primitives such as :math:`\beta` and :math:`\gamma` in the class ``CakeEating``. 

This class will also have a function which returns the right hand right of the bellman equation which needs to be maximized, which is the function that will run through the ``maximize`` function. 

.. code-block:: python3

    class CakeEating:

        def __init__(self,
                     β=0.96,         # discount factor
                     γ=0.5,          # degree of relative risk aversion
                     y_grid_max=10,  # inital stock of capital Y
                     y_grid_size=120):

            self.β, self.γ = β, γ

            # Set up grid
            self.y_grid = np.linspace(1e-04, y_grid_max, y_grid_size)

        # Utility function
        def u(self, c):

            γ = self.γ

            if γ == 1:
                return np.log(c)
            else:
                return (c ** (1 - γ)) / (1 - γ)

        # first derivative of utility function
        def du(self, c):

            return c ** (-self.γ)

        # the inverse of the first derivative
        def du_inv(self, u_prime):

            return  u_prime ** (- 1 / self.γ)

        def state_action_value(self, c, y, v_array):
            """
            Right hand side of the Bellman equation given y and c.
            """

            u, β = self.u, self.β

            v_func = lambda y: interp(self.y_grid, v_array, y)

            return u(c) + β * v_func(y - c)


We now define ``T`` which implement the Bellman operation and update the value at each grid point.

.. code-block:: python3

    def T(ce, v):
        """
        The Bellman operator.  Updates the guess of the value function.

        * ce is an instance of CakeEating
        * v is an array representing a guess of the value function

        """
        v_new = np.empty_like(v)

        for i in range(len(ce.y_grid)):
            y = ce.y_grid[i]
            # Maximize RHS of Bellman equation at state y
            v_new[i] = maximize(ce.state_action_value, 1e-10, y, (y, v))[1]

        return v_new

After defining the Bellman operator, we are ready to solve the model.
Let's start with creating a ``CakeEating`` instance using default parameterization.

.. code-block:: python3

    ce = CakeEating()

Now let's see the iteration of the value function in action. We choose an intial guess whose value
is :math:`0` for every :math:`y` grid point. 

We should see that the value functions converge to a fixed point as we apply Bellman operations.

.. code-block:: python3

    y_grid = ce.y_grid
    v = np.zeros(len(y_grid))  # Initial guess
    n = 35                     # Number of iterations

    fig, ax = plt.subplots()

    ax.plot(y_grid, v, color=plt.cm.jet(0),
            lw=2, alpha=0.6, label='Initial guess')

    for i in range(n):
        v = T(ce, v)  # Apply the Bellman operator
        ax.plot(y_grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

    ax.legend()
    ax.set_ylabel('$v(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    ax.set_title('Value function iterations')

    plt.show()

We can define a wrapper function ``compute_value_function`` which does the value function iterations
until some convergence conditions are satisfied and then return a converged value function.

.. code-block:: python3

    def compute_value_function(ce,
                               tol=1e-4,
                               max_iter=1000,
                               verbose=True,
                               print_skip=25):

        # Set up loop
        v = np.zeros(len(ce.y_grid)) # Initial guess
        v_new = np.empty_like(v)
        i = 0
        error = tol + 1

        while i < max_iter and error > tol:
            v_new[:] = T(ce, v)

            error = np.max(np.abs(v - v_new))
            i += 1

            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")

            v[:] = v_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return v_new

.. code-block:: python3

    v = compute_value_function(ce)

Now we can plot and see what the converged value function looks like. 

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(y_grid, v, label='Approximate value function')
    ax.set_ylabel('$V(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    ax.set_title('Value function')
    ax.legend()
    plt.show()



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

    ax.plot(y_grid, v_analytical, label='Analytical value function')
    ax.plot(y_grid, v, label='Numerical value function')
    ax.set_ylabel('$V(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    ax.legend()
    ax.set_title('Comparison between analytical and numerical value functions')
    plt.show()

Hooray! It looks like value function iteration gives result that is pretty close to the analytical solution.





Policy Function
---------------

Now that we have the solution of the value function it is straightforward for us to bakc out the optimal consumption
sequence :math:`\{c_t\}_{t = 0}^{\infty}` given the initial size of the cake :math:`y_0`.

As we have seen before, the Bellman equation is recursive and the optimal consumption at each time :math:`t` only
depends on the current state :math:`y_t`. The one-to-one mapping which determines the optimal consumption
:math:`c^*_t` given :math:`y_t` is often referred to as the agents' optimal policy function :math:`\sigma`.

.. math::
    c^*_t = \sigma(y_t) = \arg \max_{c_t} \{u(c_t) + \beta v(y_t - c_t)\}

Below we implement the optimal policy function. It is very similar with the Bellman operator ``T``, while this time
we focus on the optimal consumptions instead of updating values.

.. code-block:: python3

    def σ(ce, v):
        """
        The optimal policy function. Given the value function,
        it finds optimal consumption in each state.

        * ce is an instance of CakeEating
        * v is a value function array

        """
        c = np.empty_like(v)

        for i in range(len(ce.y_grid)):
            y = ce.y_grid[i]
            # Maximize RHS of Bellman equation at state y
            c[i] = maximize(ce.state_action_value, 1e-10, y, (y, v))[0]

        return c

Let's pass the converged value function array we got before to ``σ`` and compute the optimal consumptions.

.. code-block:: python3

    c = σ(ce, v)  

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(y_grid, c)
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.set_title('Optimal policy')
    plt.show()

.. _pol_an:


We can compare the optimal policy function computed numerically with the analytical one. 

The analytical optimal policy function in this cake eating problem is

.. math::
    c^* = \left(1-\beta^\frac{1}{\gamma}\right)y

We define a function ``c_star`` that computes analytical optimal consumptions in each state :math:`y`,
taking a ``CakeEating`` instance as input.

.. code-block:: python3

    def c_star(ce):

        β, γ = ce.β, ce.γ
        y_grid = ce.y_grid

        return (1 - β ** (1/γ)) * y_grid

.. code-block:: python3

    c_analytical = c_star(ce)

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(ce.y_grid, c_analytical, label='Analytical')
    ax.plot(ce.y_grid, c, label='Numerical')
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.legend()
    ax.set_title('Comparison between analytical and numerical optimal policies')
    plt.show()





Time Iteration
==============




It must satisfy the following functional equation:

.. math::
    u^{\prime}\circ\sigma\left(x\right)=\beta u^{\prime}\left(x-\sigma\left(x\right)\right)

or equivalently

.. math::
    \sigma\left(x\right)=u^{\prime-1}\left(\beta u^{\prime}\left(x-\sigma\left(x\right)\right)\right)

Computationally, we can start with any initial guess of :math:`\sigma\left(x\right)` and apply the following policy function operator
:math:`K` repeatedly until it converges,

.. math::
    \sigma_{k+1}\left(x\right)=K\sigma_{k}\left(x\right)=\min\left\{ u^{\prime-1}\left(\beta u^{\prime}\left(x-\sigma_{k}\left(x\right)\right)\right),x\right\}

Note that in each iteration we make sure the consumption is no more than the state :math:`x`.




Exercises
=========






Exercise 1
------------

In our example above we assumed that the production function of captial was :math:`f(k)=k` because we were talking specficially about a cake.

Now assume that the production function is in the form of :math:`f(k)=k^{\alpha}` where :math:`\alpha\in(0,1)`

Make the required changes to the code above and plot the value and policy functions. Comment on the change in the policy function. 

Note :math:`y_t=f(k_t)`

Output tomorrow is

.. math::
    y_{t+1}=f(y_t-c_t)\ \text{for all}\ t




Exercise 2
----------

Try to accelerate the code using Numba.

Specially, please speed up the ``CakeEating`` class using ``jitclass``, and speed up the operator functions ``T`` and ``K`` and the optimal policy function ``σ`` with ``jit`` using ``nopython`` mode.

One basic function that is called by other functions is ``maximize``. You can choose to "jit" this function, or use an alternative
``quantecon.optimize.brent_max`` which has already been "jitted" and is easy to use.



Exercise 3
----------

Implement time iteration.






Solutions
==========


Exercise 1
-----------

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

    v = compute_value_function(og, verbose=False)

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



Because there is diminishing returns to capital and there is no growth in capital. The agent wants to eat more today to avoid the shrinking of the cake tomorrow.




Exercise 2
----------

Let's start with importing from numba.

.. code-block:: python3

    from numba import jit, jitclass, float64
    from quantecon.optimize import brent_max

First, we define a ``jitclass`` version of ``CakeEating`` class. We need to declare the types of fields of ``CakeEating`` and pass
them to the ``jitclass`` decorator.

.. code-block:: python3

    cake_eating_data = [
        ('β', float64),              # discount factor
        ('γ', float64),              # degree of relative risk aversion
        ('y_grid', float64[:])       # grid of y values
    ]

.. code-block:: python3

    @jitclass(cake_eating_data)
    class CakeEating:

        def __init__(self, β=0.96, γ=0.5, y_grid_max=10, y_grid_size=120):

            self.β, self.γ = β, γ
            self.y_grid = np.linspace(1e-5, y_grid_max, y_grid_size)

        # Utility function
        def u(self, c):

            γ = self.γ

            if γ == 1:
                return np.log(c)
            else:
                return (c ** (1 - γ)) / (1 - γ)

        # FOC of utility function
        def du(self, c):
            
            return c ** (-self.γ)
        
        def du_inv(self, u_prime):

            return  u_prime ** (- 1 / self.γ)

        # objective function for optimization
        def state_action_value(self, c, y, v_array):

            u, β = self.u, self.β
            y_grid = self.y_grid

            v = lambda y: interp(y_grid, v_array, y)

            return u(c) + β * v(y - c)

.. code-block:: python3

    ce = CakeEating()

Now, let's redefine all the operator functions with decorator ``@jit(nopython=True)`` and solve the model again.
We are going to replace ``maximize`` with ``brent_max``.

Value function iteration

.. code-block:: python3

    @jit(nopython=True)
    def T(ce, v):

        v_new = np.empty_like(v)

        for i in range(len(ce.y_grid)):
            y = ce.y_grid[i]

            # Maximize RHS of Bellman equation at state y
            v_new[i] = brent_max(ce.state_action_value, 1e-10, y, args=(y, v))[1]

        return v_new

.. code-block:: python3

    @jit(nopython=True)
    def compute_value_function(ce, max_iter=500, tol=1e-6):

        v = np.zeros(ce.y_grid.size)
        v_new = np.empty_like(v)

        i = 0
        error = tol + 1
        while i < max_iter and error > tol:

            v_new[:] = T(ce, v)

            error = np.max(np.abs(v_new - v))
            i += 1

            v[:] = v_new

        return v

.. code-block:: python3

    v = compute_value_function(ce)
    fig, ax = plt.subplots()

    ax.plot(ce.y_grid, v)
    ax.set_ylabel('$V(y)$')
    ax.set_xlabel('$y$')
    ax.set_title('Value function')
    plt.show()

Optimal policy function

.. code-block:: python3

    @jit(nopython=True)
    def compute_policy(ce, v):

        y_grid = ce.y_grid
        c = np.empty_like(v)

        for i in range(len(y_grid)):
            y = y_grid[i]
            c[i] = brent_max(ce.state_action_value, 1e-10, y, args=(y, v))[0]

        return c

.. code-block:: python3

    c = compute_policy(ce, v)

    fig, ax = plt.subplots()
    ax.plot(ce.y_grid, c)
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.set_title('Optimal policy')
    plt.show()

Euler equation iteration

.. code-block:: python3

    @jit(nopython=True)
    def K(ce, c):

        y_grid = ce.y_grid
        β = ce.β
        
        y_next = y_grid - c # state transition
        du_next = ce.du(interp(y_grid, c, y_next))
        c_new = np.minimum(ce.du_inv(β * du_next), y_grid)

        return c_new

.. code-block:: python3

    @jit(nopython=True)
    def iterate_euler_equation(ce, max_iter=500, tol=1e-10):

        y_grid = ce.y_grid

        c = np.copy(y_grid) # initial guess
        c_new = np.empty_like(c)

        i = 0
        error = tol + 1
        while i < max_iter and error > tol:

            c_new[:] = K(ce, c)

            error = np.max(np.abs(c_new - c))
            i += 1

            c[:] = c_new

        return c

.. code-block:: python3

    c_euler = iterate_euler_equation(ce)

    fig, ax = plt.subplots()

    ax.plot(ce.y_grid, c_euler)
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.set_title('Optimal policy')
    plt.show()








Exercise 3
----------

Here's one way to implement time iteration.


.. code-block:: python3

    def K(ce, c):
        """
        The policy function operator. Given the policy function,
        it updates the optimal consumption using Euler equation.

        * ce is an instance of CakeEating
        * c is a policy function array

        """

        y_grid = ce.y_grid
        β = ce.β
        
        y_next = y_grid - c # state transition
        du_next = ce.du(interp(y_grid, c, y_next))
        c_new = np.minimum(ce.du_inv(β * du_next), y_grid)

        return c_new

.. code-block:: python3

    def iterate_euler_equation(ce,
                               max_iter=500,
                               tol=1e-10,
                               verbose=True,
                               print_skip=25):

        y_grid = ce.y_grid

        c = np.copy(y_grid) # initial guess
        c_new = np.empty_like(c)

        i = 0
        error = tol + 1
        while i < max_iter and error > tol:

            c_new[:] = K(ce, c)

            error = np.max(np.abs(c_new - c))
            i += 1

            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")

            c[:] = c_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return c

.. code-block:: python3

    c_euler = iterate_euler_equation(ce)

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(ce.y_grid, c_analytical, label='Analytical')
    ax.plot(ce.y_grid, c_euler, label='Euler')
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.legend()
    ax.set_title('Optimal consumption computed using Euler equation iteration')
    plt.show()

