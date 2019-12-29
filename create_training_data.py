# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Create training data

import numpy as np
from pathlib import Path

top_dir = Path('/local/scratch/ssd2/jkahn/learned_order/')

# ## Start with trivial case
#
# Just generate a range of shuffled numbers and get the network to learn the permutation matrix to reorder them.

arange_dir = top_dir / 'arange'

num_elems = 10
num_samples = 100

orig = np.arange(num_elems)

ident = np.eye(num_elems)
ident.shape

y = np.asarray([np.random.permutation(ident) for _ in np.arange(num_samples)])

X = np.asarray([np.matmul(y[i].T, orig) for i in np.arange(y.shape[0])])

y.shape, orig.shape, X.shape

# Final reshape to make this match our problem (event, particles, features)
X = X.reshape(num_samples, num_elems, 1)

np.save(arange_dir / 'X.npy', X)
np.save(arange_dir / 'y.npy', y)
