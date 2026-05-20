# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# <a id='optgrowth'></a>
# <a href="#"><img src="/_static/img/jupyter-notebook-download-blue.svg" id="notebook_download_badge"></a>
#
# <script>
# var path = window.location.pathname;
# var pageName = path.split("/").pop().split(".")[0];
# var downloadLink = ["/", "_downloads/ipynb/py/", pageName, ".ipynb"].join("");
# document.getElementById('notebook_download_badge').parentElement.setAttribute('href', downloadLink);
# </script>
#
# <a href="/status.html"><img src="https://img.shields.io/badge/Execution%20test-not%20available-lightgrey.svg" id="executability_status_badge"></a>
#
# <div class="how-to">
#         <a href="#" class="toggle"><span class="icon icon-angle-double-down"></span>How to read this lecture...</a>
#         <div class="how-to-content">
#                 <p>Code should execute sequentially if run in a Jupyter notebook</p>
#                 <ul>
#                         <li>See the <a href="/py/getting_started.html">set up page</a> to install Jupyter, Python and all necessary libraries</li>
#                         <li>Please direct feedback to <a href="mailto:contact@quantecon.org">contact@quantecon.org</a> or the <a href="http://discourse.quantecon.org/">discourse forum</a></li>
#                 </ul>
#         </div>
# </div>

# %% [markdown]
# # Optimal Growth I: The Stochastic Optimal Growth Model

# %% [markdown]
# ## Contents
#
# - [Optimal Growth I: The Stochastic Optimal Growth Model](#Optimal-Growth-I:-The-Stochastic-Optimal-Growth-Model)  
#   - [Overview](#Overview)  
#   - [The Model](#The-Model)  
#   - [Computation](#Computation)  
#   - [Exercises](#Exercises)  
#   - [Solutions](#Solutions)  

# %% [markdown]
# ## Overview
#
# In this lecture we’re going to study a simple optimal growth model with one agent
#
# The model is a version of the standard one sector infinite horizon growth model studied in
#
# - [[SLP89]](zreferences.ipynb#stokeylucas1989), chapter 2  
# - [[LS18]](zreferences.ipynb#ljungqvist2012), section 3.1  
# - [EDTC](http://johnstachurski.net/edtc.html), chapter 1  
# - [[Sun96]](zreferences.ipynb#sundaram1996), chapter 12  
#
#
# The technique we use to solve the model is dynamic programming
#
# Our treatment of dynamic programming follows on from earlier
# treatments in our lectures on [shortest paths](short_path.ipynb#) and
# [job search](mccall_model.ipynb#)
#
# We’ll discuss some of the technical details of dynamic programming as we
# go along

# %% [markdown]
# ## The Model
#
#
# <a id='index-1'></a>
# Consider an agent who owns an amount $ y_t \in \mathbb R_+ := [0, \infty) $ of a consumption good at time $ t $
#
# This output can either be consumed or invested
#
# When the good is invested it is transformed one-for-one into capital
#
# The resulting capital stock, denoted here by $ k_{t+1} $, will then be used for production
#
# Production is stochastic, in that it also depends on a shock $ \xi_{t+1} $ realized at the end of the current period
#
# Next period output is
#
# $$
# y_{t+1} := f(k_{t+1}) \xi_{t+1}
# $$
#
# where $ f \colon \RR_+ \to \RR_+ $ is called the production function
#
# The resource constraint is
#
#
# <a id='equation-outcsdp0'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# k_{t+1} + c_t \leq y_t
# $$
# </td><td width=10% style='text-align:center !important;'>
# (1)
# </td></tr></table>
#
# and all variables are required to be nonnegative

# %% [markdown]
# ### Assumptions and Comments
#
# In what follows,
#
# - The sequence $ \{\xi_t\} $ is assumed to be IID  
# - The common distribution of each $ \xi_t $ will be denoted $ \phi $  
# - The production function $ f $ is assumed to be increasing and continuous  
# - Depreciation of capital is not made explicit but can be incorporated into the production function  
#
#
# While many other treatments of the stochastic growth model use $ k_t $ as the state variable, we will use $ y_t $
#
# This will allow us to treat a stochastic model while maintaining only one state variable
#
# We consider alternative states and timing specifications in some of our other lectures

# %% [markdown]
# ### Optimization
#
# Taking $ y_0 $ as given, the agent wishes to maximize
#
#
# <a id='equation-texs0_og2'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# \mathbb E \left[ \sum_{t = 0}^{\infty} \beta^t u(c_t) \right]
# $$
# </td><td width=10% style='text-align:center !important;'>
# (2)
# </td></tr></table>
#
# subject to
#
#
# <a id='equation-og_conse'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# y_{t+1} = f(y_t - c_t) \xi_{t+1}
# \quad \text{and} \quad
# 0 \leq c_t \leq y_t
# \quad \text{for all } t
# $$
# </td><td width=10% style='text-align:center !important;'>
# (3)
# </td></tr></table>
#
# where
#
# - $ u $ is a bounded, continuous and strictly increasing utility function and  
# - $ \beta \in (0, 1) $ is a discount factor  
#
#
# In [(3)](#equation-og_conse) we are assuming that the resource constraint [(1)](#equation-outcsdp0) holds with equality — which is reasonable because $ u $ is strictly increasing and no output will be wasted at the optimum
#
# In summary, the agent’s aim is to select a path $ c_0, c_1, c_2, \ldots $ for consumption that is
#
# 1. nonnegative,  
# 1. feasible in the sense of [(1)](#equation-outcsdp0),  
# 1. optimal, in the sense that it maximizes [(2)](#equation-texs0_og2) relative to all other feasible consumption sequences, and  
# 1. *adapted*, in the sense that the action $ c_t $ depends only on
#    observable outcomes, not future outcomes such as $ \xi_{t+1} $  
#
#
# In the present context
#
# - $ y_t $ is called the *state* variable — it summarizes the “state of the world” at the start of each period  
# - $ c_t $ is called the *control* variable — a value chosen by the agent each period after observing the state  

# %% [markdown]
# ### The Policy Function Approach
#
#
# <a id='index-2'></a>
# One way to think about solving this problem is to look for the best **policy function**
#
# A policy function is a map from past and present observables into current action
#
# We’ll be particularly interested in **Markov policies**, which are maps from the current state $ y_t $ into a current action $ c_t $
#
# For dynamic programming problems such as this one (in fact for any [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process)), the optimal policy is always a Markov policy
#
# In other words, the current state $ y_t $ provides a sufficient statistic
# for the history in terms of making an optimal decision today
#
# This is quite intuitive but if you wish you can find proofs in texts such as [[SLP89]](zreferences.ipynb#stokeylucas1989) (section 4.1)
#
# Hereafter we focus on finding the best Markov policy
#
# In our context, a Markov policy is a function $ \sigma \colon
# \mathbb R_+ \to \mathbb R_+ $, with the understanding that states are mapped to actions via
#
# $$
# c_t = \sigma(y_t) \quad \text{for all } t
# $$
#
# In what follows, we will call $ \sigma $ a *feasible consumption policy* if it satisfies
#
#
# <a id='equation-idp_fp_og2'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# 0 \leq \sigma(y) \leq y
# \quad \text{for all} \quad
# y \in \mathbb R_+
# $$
# </td><td width=10% style='text-align:center !important;'>
# (4)
# </td></tr></table>
#
# In other words, a feasible consumption policy is a Markov policy that respects the resource constraint
#
# The set of all feasible consumption policies will be denoted by $ \Sigma $
#
# Each $ \sigma \in \Sigma $ determines a [continuous state Markov process](stationary_densities.ipynb#) $ \{y_t\} $ for output via
#
#
# <a id='equation-firstp0_og2'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# y_{t+1} = f(y_t - \sigma(y_t)) \xi_{t+1},
# \quad y_0 \text{ given}
# $$
# </td><td width=10% style='text-align:center !important;'>
# (5)
# </td></tr></table>
#
# This is the time path for output when we choose and stick with the policy $ \sigma $
#
# We insert this process into the objective function to get
#
#
# <a id='equation-texss'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# \mathbb E
# \left[ \,
# \sum_{t = 0}^{\infty} \beta^t u(c_t) \,
# \right]
# =
# \mathbb E
# \left[ \,
# \sum_{t = 0}^{\infty} \beta^t u(\sigma(y_t)) \,
# \right]
# $$
# </td><td width=10% style='text-align:center !important;'>
# (6)
# </td></tr></table>
#
# This is the total expected present value of following policy $ \sigma $ forever,
# given initial income $ y_0 $
#
# The aim is to select a policy that makes this number as large as possible
#
# The next section covers these ideas more formally

# %% [markdown]
# ### Optimality
#
# The **policy value function** $ v_{\sigma} $ associated with a given policy $ \sigma $ is the mapping defined by
#
#
# <a id='equation-vfcsdp00'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# v_{\sigma}(y)
# =
# \mathbb E \left[ \sum_{t = 0}^{\infty} \beta^t u(\sigma(y_t)) \right]
# $$
# </td><td width=10% style='text-align:center !important;'>
# (7)
# </td></tr></table>
#
# when $ \{y_t\} $ is given by [(5)](#equation-firstp0_og2) with $ y_0 = y $
#
# In other words, it is the lifetime value of following policy $ \sigma $
# starting at initial condition $ y $
#
# The **value function** is then defined as
#
#
# <a id='equation-vfcsdp0'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# v^*(y) := \sup_{\sigma \in \Sigma} \; v_{\sigma}(y)
# $$
# </td><td width=10% style='text-align:center !important;'>
# (8)
# </td></tr></table>
#
# The value function gives the maximal value that can be obtained from state $ y $, after considering all feasible policies
#
# A policy $ \sigma \in \Sigma $ is called **optimal** if it attains the supremum in [(8)](#equation-vfcsdp0) for all $ y \in \mathbb R_+ $

# %% [markdown]
# ### The Bellman Equation
#
# With our assumptions on utility and production function, the value function as defined in [(8)](#equation-vfcsdp0) also satisfies a **Bellman equation**
#
# For this problem, the Bellman equation takes the form
#
#
# <a id='equation-fpb30'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# w(y) = \max_{0 \leq c \leq y}
#     \left\{
#         u(c) + \beta \int w(f(y - c) z) \phi(dz)
#     \right\}
# \qquad (y \in \mathbb R_+)
# $$
# </td><td width=10% style='text-align:center !important;'>
# (9)
# </td></tr></table>
#
# This is a *functional equation in* $ w $
#
# The term $ \int w(f(y - c) z) \phi(dz) $ can be understood as the expected next period value when
#
# - $ w $ is used to measure value  
# - the state is $ y $  
# - consumption is set to $ c $  
#
#
# As shown in [EDTC](http://johnstachurski.net/edtc.html), theorem 10.1.11 and a range of other texts
#
#
#     The value function  satisfies the Bellman equation**$ v^* $**
#
# In other words, [(9)](#equation-fpb30) holds when $ w=v^* $
#
# The intuition is that maximal value from a given state can be obtained by optimally trading off
#
# - current reward from a given action, vs  
# - expected discounted future value of the state resulting from that action  
#
#
# The Bellman equation is important because it gives us more information about the value function
#
# It also suggests a way of computing the value function, which we discuss below

# %% [markdown]
# ### Greedy policies
#
# The primary importance of the value function is that we can use it to compute optimal policies
#
# The details are as follows
#
# Given a continuous function $ w $ on $ \mathbb R_+ $, we say that $ \sigma \in \Sigma $ is $ w $-**greedy** if $ \sigma(y) $ is a solution to
#
#
# <a id='equation-defgp20'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# \max_{0 \leq c \leq y}
#     \left\{
#     u(c) + \beta \int w(f(y - c) z) \phi(dz)
#     \right\}
# $$
# </td><td width=10% style='text-align:center !important;'>
# (10)
# </td></tr></table>
#
# for every $ y \in \mathbb R_+ $
#
# In other words, $ \sigma \in \Sigma $ is $ w $-greedy if it optimally
# trades off current and future rewards when $ w $ is taken to be the value
# function
#
# In our setting, we have the following key result
#
#
#     A feasible consumption policy is optimal if and only if it is -greedy- $ v^* $  
#
#
# The intuition is similar to the intuition for the Bellman equation, which was
# provided after [(9)](#equation-fpb30)
#
# See, for example, theorem 10.1.11 of [EDTC](http://johnstachurski.net/edtc.html)
#
# Hence, once we have a good approximation to $ v^* $, we can compute the (approximately) optimal policy by computing the corresponding greedy policy
#
# The advantage is that we are now solving a much lower dimensional optimization
# problem

# %% [markdown]
# ### The Bellman Operator
#
# How, then, should we compute the value function?
#
# One way is to use the so-called **Bellman operator**
#
# (An operator is a map that sends functions into functions)
#
# The Bellman operator is denoted by $ T $ and defined by
#
#
# <a id='equation-fcbell20_optgrowth'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# Tw(y) := \max_{0 \leq c \leq y}
# \left\{
#     u(c) + \beta \int w(f(y - c) z) \phi(dz)
# \right\}
# \qquad (y \in \mathbb R_+)
# $$
# </td><td width=10% style='text-align:center !important;'>
# (11)
# </td></tr></table>
#
# In other words, $ T $ sends the function $ w $ into the new function
# $ Tw $ defined [(11)](#equation-fcbell20_optgrowth)
#
# By construction, the set of solutions to the Bellman equation [(9)](#equation-fpb30) *exactly coincides with* the set of fixed points of $ T $
#
# For example, if $ Tw = w $, then, for any $ y \geq 0 $,
#
# $$
# w(y)
# = Tw(y)
# = \max_{0 \leq c \leq y}
# \left\{
#     u(c) + \beta \int v^*(f(y - c) z) \phi(dz)
# \right\}
# $$
#
# which says precisely that $ w $ is a solution to the Bellman equation
#
# It follows that $ v^* $ is a fixed point of $ T $

# %% [markdown]
# ### Review of Theoretical Results
#
#
# <a id='index-3'></a>
# One can also show that $ T $ is a contraction mapping on the set of continuous bounded functions on $ \mathbb R_+ $ under the supremum distance
#
# $$
# \rho(g, h) = \sup_{y \geq 0} |g(y) - h(y)|
# $$
#
# See  [EDTC](http://johnstachurski.net/edtc.html), lemma 10.1.18
#
# Hence it has exactly one fixed point in this set, which we know is equal to the value function
#
# It follows that
#
# - The value function $ v^* $ is bounded and continuous  
# - Starting from any bounded and continuous $ w $, the sequence $ w, Tw, T^2 w, \ldots $ generated by iteratively applying $ T $ converges uniformly to $ v^* $  
#
#
# This iterative method is called **value function iteration**
#
# We also know that a feasible policy is optimal if and only if it is $ v^* $-greedy
#
# It’s not too hard to show that a $ v^* $-greedy policy exists (see  [EDTC](http://johnstachurski.net/edtc.html), theorem 10.1.11 if you get stuck)
#
# Hence at least one optimal policy exists
#
# Our problem now is how to compute it

# %% [markdown]
# ### Unbounded Utility
#
#
# <a id='index-5'></a>
# The results stated above assume that the utility function is bounded
#
# In practice economists often work with unbounded utility functions — and so will we
#
# In the unbounded setting, various optimality theories exist
#
# Unfortunately, they tend to be case specific, as opposed to valid for a large range of applications
#
# Nevertheless, their main conclusions are usually in line with those stated for
# the bounded case just above (as long as we drop the word “bounded”)
#
# Consult,  for example, section 12.2 of [EDTC](http://johnstachurski.net/edtc.html), [[Kam12]](zreferences.ipynb#kamihigashi2012) or [[MdRV10]](zreferences.ipynb#mv2010)

# %% [markdown]
# ## Computation
#
#
# <a id='index-6'></a>
# Let’s now look at computing the value function and the optimal policy

# %% [markdown]
# ### Fitted Value Iteration
#
#
# <a id='index-7'></a>
# The first step is to compute the value function by value function iteration
#
# In theory, the algorithm is as follows
#
# 1. Begin with a function $ w $ — an initial condition  
# 1. Solving [(11)](#equation-fcbell20_optgrowth), obtain the function $ T w $  
# 1. Unless some stopping condition is satisfied, set $ w = Tw $ and go to step 2  
#
#
# This generates the sequence $ w, Tw, T^2 w, \ldots $
#
# However, there is a problem we must confront before we implement this procedure: The iterates can neither be calculated exactly nor stored on a computer
#
# To see the issue, consider [(11)](#equation-fcbell20_optgrowth)
#
# Even if $ w $ is a known function, unless $ Tw $ can be shown to have
# some special structure, the only way to store it is to record the
# value $ Tw(y) $ for every $ y \in \mathbb R_+ $
#
# Clearly this is impossible
#
# What we will do instead is use **fitted value function iteration**
#
# The procedure is to record the value of the function $ Tw $ at only finitely many “grid” points $ y_1 < y_2 < \cdots < y_I $ and reconstruct it from this information when required
#
# More precisely, the algorithm will be
#
#
# <a id='fvi-alg'></a>
# 1. Begin with an array of values $ \{ w_1, \ldots, w_I \} $ representing the values of some initial function $ w $ on the grid points $ \{ y_1, \ldots, y_I \} $  
# 1. Build a function $ \hat w $ on the state space $ \mathbb R_+ $ by interpolation or approximation, based on these data points  
# 1. Obtain and record the value $ T \hat w(y_i) $ on each grid point $ y_i $ by repeatedly solving [(11)](#equation-fcbell20_optgrowth)  
# 1. Unless some stopping condition is satisfied, set $ \{ w_1, \ldots, w_I \} = \{ T \hat w(y_1), \ldots, T \hat w(y_I) \} $ and go to step 2  
#
#
# How should we go about step 2?
#
# This is a problem of function approximation, and there are many ways to approach it
#
# What’s important here is that the function approximation scheme must not only produce a good approximation to $ Tw $, but also combine well with the broader iteration algorithm described above
#
# The next figure illustrates piecewise linear interpolation of an arbitrary function on grid points $ 0, 0.2, 0.4, 0.6, 0.8, 1 $
#
# We use an interpolation function from the
# [interpolation.py package](https://github.com/EconForge/interpolation.py)
# because it comes in handy later when we want to just-in-time compile our code

# %%
import numpy as np
import matplotlib.pyplot as plt
from interpolation import interp
from numba import njit, prange
from quantecon.optimize.scalar_maximization import brent_max


def f(x):
    y1 = 2 * np.cos(6 * x) + np.sin(14 * x)
    return y1 + 2.5

c_grid = np.linspace(0, 1, 6)

def Af(x):
    return interp(c_grid, f(c_grid), x)

f_grid = np.linspace(0, 1, 150)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f_grid, f(f_grid), 
        'b-', 
        lw=2, 
        alpha=0.8, 
        label='true function')

ax.plot(f_grid, Af(f_grid), 
        'g-',
        alpha=0.8, label='linear approximation')

ax.vlines(c_grid, c_grid * 0, f(c_grid), linestyle='dashed', alpha=0.5)
ax.legend(loc='upper center')
ax.set(xlim=(0, 1), ylim=(0, 6))

plt.show()


# %% [markdown]
# Another advantage of piecewise linear interpolation is that it preserves useful shape properties such as monotonicity and concavity / convexity

# %% [markdown]
# ### Optimal Growth Model
#
# We will hold the primitives of the optimal growth model in a class
#
# The distribution $ \phi $ of the shock is assumed to be lognormal,
# and so a draw from $ \exp(\mu + \sigma \zeta) $ when $ \zeta $ is standard normal

# %%
class OptimalGrowthModel:

    def __init__(self,
                 f,          # production function
                 u,          # utility
                 β=0.96,     # discount factor
                 μ=0,        # shock parameter
                 s=0.1,      # shock parameter
                 grid_max=4,
                 grid_size=200,
                 shock_size=250):

        self.β, self.μ, self.s = β, μ, s
        self.f, self.u = f, u

        self.y_grid = np.linspace(1e-5, grid_max, grid_size)       # Set up grid
        self.shocks = np.exp(μ + s * np.random.randn(shock_size))  # Store shocks


# %% [markdown]
# ### The Bellman Operator
#
# Here’s a function that generates a Bellman operator using linear interpolation

# %%
def bellman_function_factory(og, parallel_flag=True):
    """
    A function factory for building the Bellman operator, as well as
    a function that computes greedy policies.
    
    Here og is an instance of OptimalGrowthModel.
    """

    f, u = og.f, og.u
    y_grid, shocks = og.y_grid, og.shocks

    @njit
    def objective(c, w, y):
        """
        The right hand side of the Bellman equation
        """
        # First turn w into a function via interpolation
        w_func = lambda x: interp(y_grid, w, x)
        return u(c) + β * np.mean(w_func(f(y - c) * shocks))

    @njit(parallel=parallel_flag)
    def T(w):
        """
        The Bellman operator
        """
        w_new = np.empty_like(w)
        for i in prange(len(y_grid)):
            y = y_grid[i]
            # Solve for optimal w at y
            w_max = brent_max(objective, 1e-10, y, args=(w, y))[1]  
            w_new[i] = w_max
        return w_new

    @njit
    def get_greedy(v):
        """
        Computes the v-greedy policy of a given function v
        """
        σ = np.empty_like(v)
        for i in range(len(y_grid)):
            y = y_grid[i]
            # Solve for optimal c at y
            c_max = brent_max(objective, 1e-10, y, args=(v, y))[0]  
            σ[i] = c_max
        return σ

    return T, get_greedy


# %% [markdown]
# The function `bellman_function_factory` takes a class that represents the growth model, and returns the operator T and a function get_greedy that we will use to solve the model
#
# Notice that the expectation in [(11)](#equation-fcbell20_optgrowth) is computed via Monte Carlo, using the approximation
#
# $$
# \int w(f(y - c) z) \phi(dz) \approx \frac{1}{n} \sum_{i=1}^n w(f(y - c) \xi_i)
# $$
#
# where $ \{\xi_i\}_{i=1}^n $ are IID draws from $ \phi $
#
# Monte Carlo is not always the most efficient way to compute integrals numerically but it does have some theoretical advantages in the present setting
#
# (For example, it preserves the contraction mapping property of the Bellman operator — see, e.g., [[PalS13]](zreferences.ipynb#pal2013))
#
#
# <a id='benchmark-growth-mod'></a>

# %% [markdown]
# ### An Example
#
# Let’s test out our operator when
#
# - $ f(k) = k^{\alpha} $  
# - $ u(c) = \ln c $  
# - $ \phi $ is the distribution of $ \exp(\mu + \sigma \zeta) $ when $ \zeta $ is standard normal  
#
#
# As is well-known (see [[LS18]](zreferences.ipynb#ljungqvist2012), section 3.1.2), for this particular problem an exact analytical solution is available, with
#
#
# <a id='equation-dpi_tv'></a>
# <table width=100%><tr style='background-color: #FFFFFF !important;'>
# <td width=10%></td>
# <td width=80%>
# $$
# v^*(y) =
# \frac{\ln (1 - \alpha \beta) }{ 1 - \beta}
# +
# \frac{(\mu + \alpha \ln (\alpha \beta))}{1 - \alpha}
#  \left[
#      \frac{1}{1- \beta} - \frac{1}{1 - \alpha \beta}
#  \right]
#  +
#  \frac{1}{1 - \alpha \beta} \ln y
# $$
# </td><td width=10% style='text-align:center !important;'>
# (12)
# </td></tr></table>
#
# The optimal consumption policy is
#
# $$
# \sigma^*(y) = (1 - \alpha \beta ) y
# $$
#
# We will define functions to compute the closed form solutions to check our answers

# %%
def σ_star(y, α, β):
    """
    True optimal policy
    """
    return (1 - α * β) * y

def v_star(y, α, β, μ):
    """
    True value function
    """
    c1 = np.log(1 - α * β) / (1 - β)
    c2 = (μ + α * np.log(α * β)) / (1 - α)
    c3 = 1 / (1 - β)
    c4 = 1 / (1 - α * β)
    return c1 + c2 * (c3 - c4) + c4 * np.log(y)


# %% [markdown]
# ### A First Test
#
# To test our code, we want to see if we can replicate the analytical solution numerically, using fitted value function iteration
#
# First, having run the code for the general model shown above, let’s
# generate an instance of the model and generate its Bellman operator
#
# We first need to define a jitted version of the production function

# %%
α = 0.4  # Production function parameter

@njit
def f(k):
    """
    Cobb-Douglas production function
    """
    return k**α

og = OptimalGrowthModel(f=f, u=np.log)
T, get_greedy = bellman_function_factory(og)

# %% [markdown]
# Now let’s do some tests
#
# As one preliminary test, let’s see what happens when we apply our Bellman operator to the exact solution $ v^* $
#
# In theory, the resulting function should again be $ v^* $
#
# In practice we expect some small numerical error

# %%
y_grid = og.y_grid
β, μ = og.β, og.μ

w_init = v_star(y_grid, α, β, μ)  # Start at the solution
w = T(w_init)                     # Apply the Bellman operator once

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_ylim(-35, -24)
ax.plot(y_grid, w, lw=2, alpha=0.6, label='$Tv^*$')
ax.plot(y_grid, w_init, lw=2, alpha=0.6, label='$v^*$')
ax.legend(loc='lower right')
plt.show()

# %% [markdown]
# The two functions are essentially indistinguishable, so we are off to a good start
#
# Now let’s have a look at iterating with the Bellman operator, starting off
# from an arbitrary initial condition
#
# The initial condition we’ll start with is $ w(y) = 5 \ln (y) $

# %%
w = 5 * np.log(y_grid)  # An initial condition
n = 35

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(y_grid, w, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='Initial condition')

for i in range(n):
    w = T(w)  # Apply the Bellman operator
    ax.plot(y_grid, w, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

ax.plot(y_grid, v_star(y_grid, α, β, μ), 'k-', lw=2,
        alpha=0.8, label='True value function')

ax.legend(loc='lower right')
ax.set(ylim=(-40, 10), xlim=(np.min(y_grid), np.max(y_grid)))
plt.show()


# %% [markdown]
# The figure shows
#
# 1. the first 36 functions generated by the fitted value function iteration algorithm, with hotter colors given to higher iterates  
# 1. the true value function $ v^* $ drawn in black  
#
#
# The sequence of iterates converges towards $ v^* $
#
# We are clearly getting closer
#
# We can write a function that iterates until the difference is below a particular
# tolerance level

# %%
def compute_fixed_point(og,
                        w_init,
                        use_parallel=True,
                        tol=1e-4,
                        max_iter=1000,
                        verbose=True,
                        print_skip=25):

    T, _ = bellman_function_factory(og, parallel_flag=use_parallel)

    # Set up loop
    w = w_init
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        w_new = T(w)
        error = np.max(np.abs(w - w_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        w[:] = w_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return w_new


# %% [markdown]
# We can check our result by plotting it against the true value

# %%
initial_w = 5 * np.log(y_grid)
v_solution = compute_fixed_point(og, initial_w)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(y_grid, v_solution, lw=2, alpha=0.6,
        label='Approximate value function')

ax.plot(y_grid, v_star(y_grid, α, β, μ), lw=2,
        alpha=0.6, label='True value function')

ax.legend(loc='lower right')
ax.set_ylim(-35, -24)
plt.show()

# %% [markdown]
# The figure shows that we are pretty much on the money

# %% [markdown]
# ### The Policy Function
#
#
# <a id='index-8'></a>
# To compute an approximate optimal policy, we will use the second function
# return from bellman_function_factory that backs out the optimal policy
# from the optimal wage rate
#
# The next figure compares the result to the exact solution, which, as mentioned
# above, is $ \sigma(y) = (1 - \alpha \beta) y $

# %%
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(y_grid, get_greedy(v_solution), lw=2,
        alpha=0.6, label='Approximate policy function')

ax.plot(y_grid, σ_star(y_grid, α, β),
        lw=2, alpha=0.6, label='True policy function')

ax.legend(loc='lower right')
plt.show()


# %% [markdown]
# The figure shows that we’ve done a good job in this instance of approximating
# the true policy

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1
#
# Once an optimal consumption policy $ \sigma $ is given, income follows [(5)](#equation-firstp0_og2)
#
# The next figure shows a simulation of 100 elements of this sequence for three different discount factors (and hence three different policies)
#
# <img src="_static/figures/solution_og_ex2.png" style="width:100%;height:100%">
#
#   
# In each sequence, the initial condition is $ y_0 = 0.1 $
#
# The discount factors are discount_factors = (0.8, 0.9, 0.98)
#
# We have also dialed down the shocks a bit with s = 0.05
#
# Otherwise, the parameters and primitives are the same as the log linear model discussed earlier in the lecture
#
# Notice that more patient agents typically have higher wealth
#
# Replicate the figure modulo randomness

# %% [markdown]
# ## Solutions

# %% [markdown]
# ### Exercise 1
#
# Here’s one solution (assuming as usual that you’ve executed everything above)

# %%
def simulate_og(σ_func, og, α, y0=0.1, ts_length=100):
    '''
    Compute a time series given consumption policy σ.
    '''
    y = np.empty(ts_length)
    ξ = np.random.randn(ts_length-1)
    y[0] = y0
    for t in range(ts_length-1):
        y[t+1] = (y[t] - σ_func(y[t]))**α * np.exp(og.μ + og.s * ξ[t])
    return y


# %%
fig, ax = plt.subplots(figsize=(9, 6))

for β in (0.8, 0.9, 0.98):

    og = OptimalGrowthModel(f, np.log, β=β, s=0.05)
    y_grid = og.y_grid

    initial_w = 5 * np.log(y_grid)
    v_solution = compute_fixed_point(og, initial_w, verbose=False)

    σ_star = get_greedy(v_solution)
    σ_func = lambda x: interp(y_grid, σ_star, x)  # Define an optimal policy function
    y = simulate_og(σ_func, og, α)
    ax.plot(y, lw=2, alpha=0.6, label=rf'$\beta = {β}$')

ax.legend(loc='lower right')
plt.show()
