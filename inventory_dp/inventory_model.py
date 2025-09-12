# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inventory Management Model: Vectorization Practice
#
# This notebook demonstrates a stochastic dynamic inventory management model and compares different computational approaches for calculating transition probabilities.
#
# ## Problem Overview
#
# We have an inventory system with:
#
# - $K$: Maximum inventory capacity
# - $\beta$: Discount factor
# - $c$: Marginal cost (unit cost per order)
# - $\kappa$: Fixed cost of ordering
# - $p$: Parameter for demand shock distribution
#
# Inventory evolves according to 
#
# $$
#     X_{t+1} = \max(X_t - D_{t+1}, 0) + A_t
# $$
#
# where
#
# - $X_t$ is current inventory (number of units),
# - $D_{t+1}$ is an IID demand shock, and
# - $A_t$ is the current order (number of units).
#
# It will be convenient to work with the transition probability kernel
#
# $$P(x, a, y) := \mathbb P\{X_{t+1}=y \,|\, X_t = x, A_t = a \}$$
#
# With $\phi$ as the probability density function for demand,
# the transition probability kernel obeys
#
# \begin{align}
# P(x, a, y) &= \sum_{d \geq 0} \mathbb{1}\{\max(x - d, 0) + a = y\} \phi(d) \\
# &= \sum_{d < x} \mathbb{1}\{x - d + a = y\} \phi(d) + \sum_{d \geq x} \mathbb{1}\{a = y\} \phi(d) \\
# &= \sum_{d < x} \mathbb{1}\{d = x + a - y\} \phi(d) + \mathbb{1}\{y = a\} F(x) \\
# &= \mathbb{1}\{0 \leq x + a - y < x\} \phi(x + a - y) + \mathbb{1}\{y = a\} F(x)
# \end{align}
#
# Where $F(x) = P\{D \geq x\}$ is the survival function.
#

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt


class Model(NamedTuple):
    K: int = 20      # max inventory
    β: float = 0.97  # discount factor
    c: float = 0.2   # marginal cost
    κ: float = 0.8   # fixed cost
    p: float = 0.6   # demand shock parameter


def ϕ(p, d):
    """PDF for demand shock: ϕ(d) = (1-p)^d * p"""
    return (1 - p)**d * p


def F(p, x):
    """Survival function: F(x) = P{D ≥ x} = (1-p)^x"""
    return (1 - p)**x


def generate_kernel_vmap(model): 
    """
    Vmap-based computation of the transition probability kernel P(x, a, y).
    Uses JAX's vmap to vectorize the scalar function over all (x, a, y) combinations.
    """
    K, β, c, κ, p = model
    S = K + 1

    def P(x, a, y):
        """
        Scalar function to compute P(x, a, y) for a single (x, a, y) triple.
        See the mathematical derivation above for the complete formula.
        """
        d = x + a - y
        # Test 0 <= x + a - y < x (first term)
        valid_d = jnp.logical_and(0 <= d, d < x)
        # Test y = a (second term)
        y_eq_a = jnp.equal(y, a)
        # Combine: 1{0 ≤ x + a - y < x} φ(x + a - y) + 1{y = a} F(x)
        return valid_d * ϕ(p, d) + y_eq_a * F(p, x)

    # Create all combinations of (x, a, y) indices
    x_vals = jnp.arange(S)
    a_vals = jnp.arange(S)  
    y_vals = jnp.arange(S)
    
    # Use vmap to compute P(x,a,y) for all combinations
    vmap_y = jax.vmap(P,      (None, None, 0))
    vmap_a = jax.vmap(vmap_y, (None, 0, None))
    vmap_x = jax.vmap(vmap_a, (0, None, None))
    
    return vmap_x(x_vals, a_vals, y_vals)

model = Model()



