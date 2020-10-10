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

from sklearn.neural_network import MLPRegressor

import numpy as np


class DualAutoencoder(object):

    def __init__(self, model_name='DAF'):

        self.model_name = model_name

        self.stacks = {}

        self.steps = None

    def build(self, stack_size=2, n_hidden_neurons=100, activation_functions=('logistic', 'tanh'),
              l2_alpha=.1, batch_size='auto', steps=10, optimization_algorithm='lbfgs'):

        self.steps = steps

        for activation_function in activation_functions:

            for _ in range(stack_size):

                if activation_function not in self.stacks:
                    self.stacks[activation_function] = []

                self.stacks[activation_function].append(
                    MLPRegressor(hidden_layer_sizes=(n_hidden_neurons,),
                                 activation=activation_function,
                                 solver=optimization_algorithm, alpha=l2_alpha,
                                 batch_size=batch_size, max_iter=1,
                                 random_state=13))

                self.stacks[activation_function][-1].out_activation_ = 'logistic'

    def fit(self, x, noise_percentage=.25):

        for af in self.stacks:

            input = x

            for model in self.stacks[af]:

                for _ in range(self.steps):

                    input_with_noise = input * self.__binary_mask(shape=input.shape, zero_probability=noise_percentage)
                    print(input_with_noise)
                    model.fit(input_with_noise, x)

                input = np.add(np.matmul(input, model.coefs_[0]), model.intercepts_[0])

    def transform(self, x):

        result = None

        for af in self.stacks:

            input = x

            for model in self.stacks[af]:

                input = np.add(np.matmul(input, model.coefs_[0]), model.intercepts_[0])

            if result is None:
                result = input

            else:
                result = np.concatenate((result, input), axis=1)

        return result

    def fit_transform(self, x):

        self.fit(x)

        return self.transform(x)

    def predict(self, x):

        for af in self.stacks:

            return self.stacks[af][0].predict(x)

    def __binary_mask(self, shape, zero_probability=.25):

        return np.where(np.random.uniform(0.0, 1.0, shape) <= zero_probability,
                        np.zeros(shape=shape),
                        np.ones(shape=shape))
