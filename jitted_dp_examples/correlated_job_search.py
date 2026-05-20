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
# ## Job Search
#
# #### John Stachurski

# %% [markdown]
# This notebook solves a McCall style job search model with persistent and transitory components to wages.  Dynamic programming is accelerated by JIT compilation using Numba, as well as shared-memory parallelization via `prange`.

# %% [markdown]
# Wages at each point in time are given by
#
# $$    w = \exp(z) + y $$
#
# $$    y \sim \exp(μ + s ζ)  $$
#
# $$    z' = d + ρ z + σ ε $$
#
# with ζ and ε both iid and N(0, 1).   
#
# The worker can either
#
# * accept an offer and work permanently at that wage, or
# * take unemployment compensation $c$ and wait till next period
#
# The value function satisfies the Bellman equation
#
# $$ v^*(w, z) = \max \left\{ \frac{u(w)}{1-β}, u(c) + β \, \mathbb E_z v^*(w', z') \right\} $$
#
# There's a way that we can reduce dimension in this problem, which massively accelerates compution.
#
# To see this, let $f^*$ be the continuation value function, defined by
#
# $$ f^*(z) := u(c) + β \, \mathbb E_z v^*(w', z') $$
#
# The Bellman equation can now be written
#
# $$ v^*(w, z) = \max \left\{ \frac{u(w)}{1-β}, \, f^*(z) \right\} $$
#
# Combining the last two expressions, we see that the continuation value function satisfies
#
# $$    f^*(z) = u(c) + β \, \mathbb E_z \max \left\{ \frac{u(w')}{1-β}, f^*(z') \right\} $$
#
# We'll solve this functional equation for $f^*$ by introducing the operator
#
# $$    Qf(z) = u(c) + β \, \mathbb E_z \max \left\{ \frac{u(w')}{1-β}, f(z') \right\} $$
#
#
# By construction, $f^*$ is a fixed point of $Q$
#
# It turns out that $Q$ is a contraction map, so $f^*$ is the unique fixed point and we can calculate it by iteration
#
# Once we have $f^*$, we can solve the search problem by stopping when the reward for excepting exceeds the continuation value, or
#
# $$    \frac{u(w)}{1-β} \geq f^*(z) $$
#
# For utility we take $u(c) = \ln(c)$.  The reservation wage is the wage where
# equality holds in the last expression.
#
# That is,
#
# $$    w^*(z) = \exp(f^*(z) (1-β)) $$
#
# One of our key aims is to solve for the reservation rule. 
#
# When we iterate, f is stored as a vector of values on a grid and these points
# are interpolated into a function as necessary.
#
# Interpolation is piecewise linear.
#
# The integral in the definition of $Qf$ is calculated by Monte Carlo.

# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numba import jit, njit, prange
from interpolation import interp


# %%
class JobSearch:

    def __init__(self,
                 μ=0.0,    # transient shock log mean
                 s=1.0,    # transient shock log variance
                 d=0.0,    # shift coefficient of persistent state
                 ρ=0.9,    # correlation coef. of persistent state
                 σ=0.1,    # state volatility
                 β=0.98,   # discount factor
                 c=5,      # unemployment compensation
                 mc_size=2000,
                 grid_size=200):

        self.μ, self.s, self.d,  = μ, s, d, 
        self.ρ, self.σ, self.β, self.c = ρ, σ, β, c 

        # Set up grid
        z_mean = d / (1 - ρ)
        z_sd = np.sqrt(σ / (1 - ρ**2))
        k = 3  # std devs from mean
        a, b = z_mean - k * z_sd, z_mean + k * z_sd
        self.z_grid = np.linspace(a, b, grid_size)

        # Store shocks
        self.mc_size = mc_size
        self.e_draws = randn(2, mc_size)

    def parameters(self):
        """
        Return all parameters as a tuple.
        """
        return self.μ, self.s, self.d, \
                self.ρ, self.σ, self.β, self.c
        


# %%
def generate_Q_operator(js, parallel_flag=True):
    """
    Build an efficient Q operator and return it.
    """
    μ, s, d, ρ, σ, β, c = js.parameters()
    e_draws = js.e_draws
    z_grid = js.z_grid
    M = e_draws.shape[1]

    @njit(parallel=parallel_flag)
    def Q(h):
        h_new = np.empty_like(h)
        for i in prange(len(z_grid)):
            z = z_grid[i]
            expectation = 0.0
            for m in range(M):
                e1, e2 = e_draws[:, m]
                z_next = d + ρ * z + σ * e1
                go_val = interp(z_grid, h, z_next)
                y_next = np.exp(μ + s * e2)             # y' draw
                w_next = np.exp(z_next) + y_next        # w' draw
                stop_val = np.log(w_next) / (1 - β)    
                expectation += max(stop_val, go_val)
            expectation = expectation / M 
            h_new[i] = np.log(c) + β * expectation
        return h_new
    
    return Q


# %%
def compute_fixed_point(js,
                        h_init,
                        use_parallel=True,
                        tol=1e-4, 
                        max_iter=1000, 
                        verbose=True,
                        print_skip=25): 

    Q = generate_Q_operator(js, parallel_flag=use_parallel)

    # Set up loop
    h = h_init
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        h_new = Q(h)
        error = np.max(np.abs(h - h_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        h[:] = h_new

    if i == max_iter: 
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return h_new


# %%
js = JobSearch()
h_init = np.log(js.c) * np.ones(len(js.z_grid))

# %%
# %%time

h_star = compute_fixed_point(js, h_init, use_parallel=False, verbose=True)

# %%
res_wage_function = np.exp(h_star * (1 - js.β))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(js.z_grid, res_wage_function, label="reservation wage given $z$")
ax.set(xlabel="$z$", ylabel="wage")
ax.legend()
plt.show()

# %% [markdown]
# ### Let's try changing unemployment compensation

# %%
c_vals = 1, 2, 3, 4

fig, ax = plt.subplots(figsize=(9, 6))

for c in c_vals:
    js = JobSearch(c=c)
    h_star = compute_fixed_point(js, h_init, verbose=False)
    res_wage_function = np.exp(h_star * (1 - js.β))
    ax.plot(js.z_grid, res_wage_function, label=f"$w^*$ at $c = {c}$")
    
ax.set(xlabel="$z$", ylabel="wage")
ax.legend()
plt.show()


# %% [markdown]
# ### Exercise: Unemployment Duration

# %% [markdown]
# Let's study how mean unemployment duration varies with unemployment compensation.
#
# For simplicity we'll fix the initial state at $z_t = 0$.

# %%
def compute_unemployment_duration(js, seed=1234):
    
    h_star = compute_fixed_point(js, h_init, verbose=False)
    μ, s, d, ρ, σ, β, c = js.parameters()
    z_grid = js.z_grid
    np.random.seed(seed)
        
    @njit
    def h_star_function(z):
        return interp(z_grid, h_star, z)

    @njit
    def draw_tau(t_max=10_000):
        z = 0
        t = 0

        unemployed = True
        while unemployed and t < t_max:
            # draw current wage
            y = np.exp(μ + s * np.random.randn())
            w = np.exp(z) + y
            res_wage = np.exp(h_star_function(z) * (1 - β))
            # if optimal to stop, record t
            if w >= res_wage:
                unemployed = False
                τ = t
            # else increment data and state 
            else:
                z = ρ * z + d + σ * np.random.randn()
                t += 1
        return τ

    @njit(parallel=True)
    def compute_expected_tau(num_reps=100_000):
        sum_value = 0
        for i in prange(num_reps):
            sum_value += draw_tau()
        return sum_value / num_reps

    return compute_expected_tau()
        


# %%
c_vals = np.linspace(1.0, 10.0, 8)
durations = np.empty_like(c_vals)
for i, c in enumerate(c_vals):
    js = JobSearch(c=c)
    τ = compute_unemployment_duration(js)
    durations[i] = τ
    
    

# %%
fig, ax = plt.subplots()
ax.plot(c_vals, durations)
ax.set_xlabel("unemployment compensation")
ax.set_ylabel("mean unemployment duration")
plt.show()

# %% [markdown]
# ### Exercise
#
# Investigate how mean unemployment duration varies with the discount factor $\beta$.  What is your prior?  Do your results match up?

# %%
for i in range(40):
    print("solution below!")

# %% [markdown]
# ### Solution

# %% [markdown]
# Here's one solution.  It shows, not surprisingly, that more patient individuals tend to wait longer before accepting an offer.

# %%
beta_vals = np.linspace(0.94, 0.99, 10)
durations = np.empty_like(beta_vals)
for i, β in enumerate(beta_vals):
    js = JobSearch(β=β)
    τ = compute_unemployment_duration(js)
    durations[i] = τ
    
    

# %%
fig, ax = plt.subplots()
ax.plot(beta_vals, durations)
ax.set_xlabel("$\\beta$")
ax.set_ylabel("mean unemployment duration")
plt.show()

# %%
