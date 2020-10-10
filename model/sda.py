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

from model import Model
import tensorflow as tf


class StackedDenoisingAutoencoder(Model):

    def __init__(self, model_name='sda', summary_directory='../output/'):

        self.graph = tf.Graph()

        self.model_name = model_name

        with self.graph.as_default():

            self.session = tf.Session(graph=self.graph)

            self.summary_directory = summary_directory

            self.optimizers = []

            self.initial_weights, self.initial_biases = [], []

            self.corrupted_inputs = []

            self.weights, self.biases = [], []

            self.encoders, self.decoders = [], []

            self.tensorboard = []

            self.add_summaries = self.summary_directory is not None

            self.saver = None

    def build(self,
                n_input_features,
                units_per_hidden_layer,
                encoder_activation_function='sigmoid',
                decoder_activation_function='identity'
                ):

        with self.graph.as_default():

            self.n_input_features = n_input_features

            self.input = self.input = tf.placeholder(tf.float32, shape=(None, n_input_features), name='input')

            self.units_per_hidden_layer = units_per_hidden_layer

            self.encoder_activation_function = encoder_activation_function

            self.decoder_activation_function = decoder_activation_function

            self.keep_probability = tf.placeholder(tf.float32, name='keep_probability')

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            with tf.name_scope('stack'):

                n_inputs = self.n_input_features

                current_input = self.input

                for i, units in enumerate(self.units_per_hidden_layer):

                    mask = tf.random_uniform(shape=tf.shape(current_input), minval=0, maxval=1,
                                             dtype=tf.float32, seed=None,
                                             name='weight_initializer_stack_{}'.format(i + 1))

                    mask = tf.where(mask <= self.keep_probability,
                                    tf.ones_like(current_input, dtype=tf.float32),
                                    tf.zeros_like(current_input, dtype=tf.float32),
                                    name='random_mask_stack_{}'.format(i + 1))

                    self.corrupted_inputs.append(tf.multiply(current_input, mask,
                                                             name='corruped_input_stack_{}'.format(i + 1)))

                    with tf.name_scope('encoder_stack_{}'.format(i + 1)):

                        previous_input = self.corrupted_inputs[-1]

                        activation_function = self.get_activation_function(self.encoder_activation_function)

                        weights = tf.Variable(tf.truncated_normal([n_inputs, units]), dtype=tf.float32,
                                              name='weights_stack_{}'.format(i + 1))

                        self.initial_weights.append(weights)

                        bias = tf.Variable(tf.zeros([units], dtype=tf.float32),
                                           name='bias_stack_{}'.format(i + 1))

                        self.initial_biases.append(bias)

                        self.encoders.append(activation_function(tf.add(tf.matmul(previous_input, weights), bias),
                                                                 name='encoder_stack_{}'.format(i + 1)))

                        current_input = self.encoders[-1]

                    with tf.name_scope('decoder_stack_{}'.format(i + 1)):

                        weights = tf.Variable(tf.truncated_normal([units, n_inputs]), dtype=tf.float32,
                                              name='weights_stack_{}'.format(i + 1))

                        bias = tf.Variable(tf.zeros([n_inputs], dtype=tf.float32),
                                           name='bias_stack_{}'.format(i + 1))

                        activation_function = self.get_activation_function(self.decoder_activation_function)

                        self.decoders.append(activation_function(tf.add(tf.matmul(self.encoders[-1], weights), bias),
                                                                 name='decoder_stack_{}'.format(i + 1)))

                    n_inputs = units

            self.saver = tf.train.Saver()

    def __build_optimizers(self, loss, optimizer):

        with self.graph.as_default():

            self.loss = loss

            self.optimizer = optimizer

            label = self.input

            losses = []

            with tf.name_scope('optimization'):

                for i, decoder in enumerate(self.decoders):

                    lf = self.get_loss(self.loss)

                    losses.append(tf.reduce_mean(lf(label, decoder)))

                    optimizer = self.get_optimizer(self.optimizer)(learning_rate=self.learning_rate)

                    optimizer = optimizer.minimize(losses[-1])

                    self.optimizers.append(optimizer)

                    label = self.encoders[i]

                with tf.name_scope('loss'):
                    for i, loss in enumerate(losses):
                        self.tensorboard.append(tf.summary.scalar('stack_{}'.format(i + 1), loss))

    def fit(self, x, steps=1000, batch_size=None, learning_rate=1e-2, loss='mse', optimizer='sgd',
            keep_probability=0.75):

        with self.graph.as_default():

            self.__build_optimizers(loss, optimizer)

            self.batch_size = x.shape[0] if batch_size is None else batch_size

            current_input = self.input

            self.session.run(tf.global_variables_initializer())

            self.session.run(tf.local_variables_initializer())

            tensorboard_writer, tb = None, None

            if self.add_summaries:

                tensorboard_path = '{}/{}'.format( self.summary_directory, self.model_name)

                tensorboard_writer = tf.summary.FileWriter(tensorboard_path,
                                                    tf.get_default_graph())

            for i, optimizer in enumerate(self.optimizers):

                for step in range(steps):

                    start, end = 0, min(self.batch_size, x.shape[0])

                    while start < x.shape[0]:

                        tb, opt = self.session.run([self.tensorboard[i], optimizer],
                                                   feed_dict={current_input: x,
                                                              self.learning_rate: learning_rate,
                                                              self.keep_probability: keep_probability})

                        start, end = end, min(x.shape[0], end + self.batch_size)

                    if self.add_summaries:

                        log_path = '{0}/{1}/{1}'.format(self.summary_directory, self.model_name)

                        self.saver.save(self.session, log_path, global_step=step + 1)

                        tensorboard_writer.add_summary(tb, step + 1)

    def get_initial_weights(self):

        result = []

        for w in self.initial_weights:
            result.append(self.session.run(w))

        return result

    def get_initial_biases(self):

        result = []

        for b in self.initial_biases:
            result.append(self.session.run(b))

        return result
