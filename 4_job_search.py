# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
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
# # Job Search
#
# **Prepared for the Bank of Portugal Computational Economics Course (Oct 2025)**
#
# **Author:** [John Stachurski](https://johnstachurski.net)
#

# %% [markdown]
# In this lecture we study a basic infinite-horizon job search problem with Markov wage
# draws 
#
# * For background on infinite horizon job search see, e.g., [DP1](https://dp.quantecon.org/).

# %% [markdown]
# In addition to what's in Anaconda, this lecture will need the QE library:

# %%
# !pip install quantecon  

# %% [markdown]
# We use the following imports.

import matplotlib.pyplot as plt
import quantecon as qe
import numpy as np
import time
from typing import NamedTuple, Callable


# %% [markdown]
# ## Model
#
# We study an elementary model where 
#
# * jobs are permanent 
# * unemployed workers receive current compensation $c$
# * the horizon is infinite
# * an unemployment agent discounts the future via discount factor $\beta \in (0,1)$
#
# ### Set up
#
# At the start of each period, an unemployed worker receives wage offer $W_t$.
#
# We assume that 
#
# $$
#     \ln W_{t+1} = \rho \ln W_t + \nu Z_{t+1}
# $$
#
# where $(Z_t)_{t \geq 0}$ is IID and standard normal.
#
# We then discretize this wage process using Tauchen's method to produce a stochastic matrix $P$.
#
# Successive wage offers are drawn from $P$.
#
# ### Rewards
#
# Since jobs are permanent, the return to accepting wage offer $w$ today is
#
# $$
#     w + \beta w + \beta^2 w + 
#     \cdots = \frac{w}{1-\beta}
# $$
#
# The Bellman equation is
#
# $$
#     v(w) = \max
#     \left\{
#             \frac{w}{1-\beta}, c + \beta \sum_{w'} v(w') P(w, w')
#     \right\}
# $$
#
# We solve this model using value function iteration.
