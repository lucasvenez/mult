# Copyright 2020 The MuLT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf


class StackedDenoisingAutoencoders(object):
    """
    Implementation of Stacked Denoising Autoencoders based on the paper 'Stacked Denoising Autoencoders: Learning Useful
    Representations in a Deep Network with a Local Denoising Criterion' enabled at
    <http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf>.
    """
    def __init__(self, n_features, neurons_per_hidden_layer=(500,), learning_rate=1E-5, keep_probability=0.5):

        assert n_features > 0 and len(neurons_per_hidden_layer) > 0

        self.input = tf.placeholder(dtype=tf.float32, shape=(None, n_features))
        self.neurons_per_hidden_layer=neurons_per_hidden_layer
        self.learning_rate=learning_rate
        self.keep_probability = keep_probability

    def __build(self):
        pass

    def optimize(self, x, batch_size=1000):
        pass

    def predict(self, x):
        pass

