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


class Model(object):

    @staticmethod
    def get_activation_function(name):

        if name == 'sigmoid':
            return tf.nn.sigmoid

        elif name == 'tanh':
            return tf.nn.tanh

        elif name == 'softmax':
            return tf.nn.softmax

        elif name == 'log_softmax':
            return tf.nn.log_softmax

        elif name == 'identity':
            return tf.identity

        elif name =='relu':
            return tf.nn.relu

        else:
            raise ValueError('Invalid activation function name: {}'.format(name))

    @staticmethod
    def get_optimizer(name):

        if name == 'adagrad':
            return tf.train.AdagradOptimizer

        elif name == 'adam':
            return tf.train.AdamOptimizer

        elif name == 'ftrl':
            return tf.train.FtrlOptimizer

        elif name == 'adadelta':
            return tf.train.AdadeltaOptimizer

        elif name == 'sgd':
            return tf.train.GradientDescentOptimizer

        elif name == 'psgd':
            return tf.train.ProximalGradientDescentOptimizer

        elif name == 'padagrad':
            return tf.train.ProximalAdagradOptimizer

        elif name == 'rmsprop':
            return tf.train.RMSPropOptimizer

        else:
            raise ValueError('Invalid optimizer name')

    @staticmethod
    def get_loss(name):

        if name == 'logloss':
            return tf.losses.log_loss

        elif name == 'huber':
            return tf.losses.huber_loss

        elif name == 'hinge':
            return tf.losses.hinge_loss

        elif name == 'softmax_cross_entropy':
            return tf.nn.softmax_cross_entropy_with_logits

        elif name == 'mse':
            return tf.losses.mean_squared_error

        else:
            raise ValueError('Invalid loss function name')