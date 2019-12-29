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

# # Train ordering net

# Choose a GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
from tensorflow.keras import regularizers, callbacks, optimizers, backend
try:
    from tensorflow.keras import layers, activations
except ModuleNotFoundError:
    from tensorflow.python.keras import layers, activations

from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# # Load datum

dataset = 'arange'
top_dir = Path('/local/scratch/ssd2/jkahn/learned_order/')
dataset_dir = top_dir / dataset

# +
X = np.load(dataset_dir / 'X.npy')
y = np.load(dataset_dir / 'y.npy')

# Just for this dataset
#y = y - 1
X.shape, y.shape
# -

# ### Splitty brah

# +
# (
#     feat_X_train, feat_X_test,
#     adj_X_train, adj_X_test,
#     y_train, y_test
# ) = train_test_split(
#     feat_X, adj_X, y,
#     train_size=0.9,
# )

# feat_X_train.shape
# -

# ## Set up network

num_features = X.shape[1]

backend.clear_session()


# +
def make_prediction_model(l2_strength=1e-5):
    
    feat_input = layers.Input(shape=(num_features, 1), name='feat_input')

    num_maps = 7
    
    # Query
    q = layers.Conv1D(num_maps, 1)(feat_input)
    q = layers.LeakyReLU()(q)
    q = layers.Conv1D(num_maps, 1)(q)
    q = layers.LeakyReLU()(q)
    # Want one final permutation matrix
    q = layers.Conv1D(1, 1)(q)
    q = layers.LeakyReLU()(q)

    # Key
    k = layers.Conv1D(num_maps, 1)(feat_input)
    k = layers.LeakyReLU()(k)
    k = layers.Conv1D(num_maps, 1)(k)
    k = layers.LeakyReLU()(k)
    # Want one final permutation matrix
    k = layers.Conv1D(1, 1)(k)
    k = layers.LeakyReLU()(k)

    # Value
    v = layers.Conv1D(num_maps, 1)(feat_input)
    v = layers.LeakyReLU()(v)
    v = layers.Conv1D(num_maps, 1)(v)
    v = layers.LeakyReLU()(v)
    # Want one final permutation matrix
    v = layers.Conv1D(1, 1)(v)
    v = layers.LeakyReLU()(v)

    # Generate the permutation matrix
    # Make sure the dims are right
    q = layers.Permute(dims=(2, 1))(q)
    # Perform the matrix multiplication
    att = layers.Dot(axes=(1, 2))([q, k])
#     # Softmax rows and columns to turn them into probabilites?
#     att = layers.Softmax(axis=1)(att)
    att = layers.Softmax(axis=2)(att)
    
    # Finally get the self attention output
#     s_att = layers.Dot(axes=(1,2))([att, v])
    s_att = backend.batch_dot(att, v)

    # Create our permutation matrix as the dot of this with itself
    perm = layers.Dot(axes=(1,2))([layers.Permute(dims=(2, 1))(s_att), s_att])
    perm = layers.Softmax(axis=1)(perm)
    perm = layers.Softmax(axis=2)(perm)
    
    output_layer = perm
    
    model = tf.keras.Model(feat_input, output_layer)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae'],
    )
    model.summary()
    return model

pred_model = make_prediction_model()
# -

pred_model.fit(
    x={
        'feat_input': X,
    },
    y=y,
#     x={
#         'feat_input': X_train,
#     },
#     y=y_train,
    batch_size=1,
    epochs=10,
#     validation_data=(
#         {
#             'feat_input': feat_X_test, 
#         },
#         y_test,
#     )
)

pred_model.predict(X[0:1]), y[0]
