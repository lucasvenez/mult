from model import Model

import tensorflow as tf
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dense(Model):
    """

    """
    def __init__(self, model_name, summaries_dir='output'):
        """

        :param model_name:
        :param summaries_dir:
        """
        self.graph = tf.Graph()

        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():

            self.summaries_dir = summaries_dir

            self.saver = None

            self.model_name = model_name

            #
            # Placeholders
            #
            self.raw_input = None

            self.expected_output = None

            self.model_path = None

            self.keep_prob = None

            self.training = None

            self.lr = None

            #
            #
            #
            self.test_writer = None

            self.tb_train = None

            self.tb_test = []

            #
            #
            #
            self.n_input_features = None

            self.abstraction_activation_functions = None

            self.n_hidden_nodes = None

            self.n_hidden_layers = None

            self.keep_probability = None

            self.n_outputs = None

            self.optimizer_algorithms = None

            self.cost_function = None

            self.add_summaries = None

            self.batch_normalization = None

            self.l2_regularizer = None

            #
            #
            #
            self.models = None

            self.cost_functions = None

            #
            #
            #
            self.optimizers = None

            self.correct_predictions = None

            self.accuracies = None

            #
            #
            #
            self.abstract_representation = None

            self.dense_layers = None

            self.l2_regularizers = None

            #
            #
            #
            self.initial_weights = None

            self.initial_bias = None

    def fit(self, x, y, x_test=None, y_test=None, learning_rate=1e-5, steps=1000, batch_size=1000, shuffle=True):

        assert steps > 0

        batch_size = x.shape[0] if batch_size is None else min(batch_size, x.shape[0])

        self.__build_optimizers()

        if batch_size is None:
            batch_size = x.shape[0]

        if x_test is None:
            x_test = x

        if y_test is None:
            y_test = y

        #
        # FIXME check if there is not uninitialized variables
        #
        with self.graph.as_default():

            self.session.run(tf.global_variables_initializer())

            self.session.run(tf.local_variables_initializer())

            tb_writer = tf.summary.FileWriter(
                self.summaries_dir + '/{}'.format(self.model_name), self.graph)

            n_rows = x.shape[0]

            index = np.array(list(range(n_rows)), dtype=np.int)

            j = 0

            for step in range(steps):

                current_block, train_log = 0, None

                while current_block < n_rows:

                    if shuffle:
                        np.random.shuffle(index)

                    batch = list(range(current_block, (min(current_block + batch_size, n_rows))))

                    train_log = self.session.run(
                                    [self.tb_train] + self.optimizers + self.models,
                                    feed_dict={self.raw_input: x[index[batch], :],
                                               self.expected_output: y[index[batch], :],
                                               self.keep_prob: self.keep_probability,
                                               self.lr: learning_rate,
                                               self.training: True})[0]

                    current_block += batch_size

                    j += 1

                if self.add_summaries:
                    tb_writer.add_summary(train_log, step)

                test_log = self.session.run(self.tb_test,
                                            feed_dict={self.raw_input: x_test,
                                                       self.expected_output: y_test,
                                                       self.keep_prob: 1.,
                                                       self.training: False})

                self.saver.save(self.session, '{0}/{1}/{1}'.format(
                    self.summaries_dir, self.model_name), global_step=step)

                if self.add_summaries:
                    for tl in test_log:
                        tb_writer.add_summary(tl, step)

    def transform(self, x):

        with self.graph.as_default():

            result = self.session.run(
                self.abstract_representation,
                feed_dict={self.raw_input: x,
                           self.keep_prob: 1.,
                           self.training: False})

            result = np.array(result)

            result = np.rollaxis(np.rollaxis(result, 2, 0), 3, 2)

            return np.reshape(result, (-1, result.shape[1], result.shape[2], result.shape[3], 1))

    def predict(self, x):

        with self.graph.as_default():

            result = self.session.run(
                self.models,
                feed_dict={self.raw_input: x,
                           self.keep_prob: 1.,
                           self.training: False})

            result = np.array(result)

            return result

    def load(self, model_path):

        if os.path.exists('{}.meta'.format(model_path)) and os.path.isfile('{}.meta'.format(model_path)):

            with self.graph.as_default():

                self.saver = tf.train.import_meta_graph('{}.meta'.format(model_path))

                self.saver.restore(self.session, tf.train.latest_checkpoint(os.path.dirname(model_path)))

                self.raw_input = tf.get_default_graph().get_tensor_by_name('raw_input:0')

                self.expected_output = tf.get_default_graph().get_tensor_by_name('expected_output:0')

                self.keep_prob = tf.get_default_graph().get_tensor_by_name('dropout_keep_probability:0')

                self.training = tf.get_default_graph().get_tensor_by_name('phase_ph:0')

                self.models = [tf.get_default_graph().get_tensor_by_name(n.name + ':0')
                               for n in tf.get_default_graph().get_operations()
                               if 'dense_model_' in n.name.split('/')[-1]]

                self.abstract_representation = []

                for model in self.models:

                    model_function = model.name.split('_')[-1].split(':')[0]

                    self.abstract_representation.append([tf.get_default_graph().get_tensor_by_name(op.name + ':0')
                                                         for op in tf.get_default_graph().get_operations()
                                                         if 'hidden_{0}_layer_'.format(model_function) in op.name and
                                                         '/{}'.format(model_function.title()) in op.name and
                                                         'grad' not in op.name])

                self.n_hidden_layers = len(self.abstract_representation[-1])

                self.n_hidden_nodes = self.abstract_representation[-1][-1].shape[1]

    def build(self, 
              n_input_features,
              n_outputs,
              abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
              n_hidden_layers=3,
              n_hidden_nodes=10,
              initial_weights=None,
              initial_bias=None,
              keep_probability=0.5,
              optimizer_algorithms=('sgd', 'sgd', 'sgd'),
              cost_function='logloss',
              add_summaries=True,
              batch_normalization=False,
              l2_regularizer=.1):

        with self.graph.as_default():

            assert isinstance(n_hidden_nodes, int) and isinstance(abstraction_activation_functions, tuple)

            assert 0. < keep_probability <= 1.

            assert n_hidden_nodes > 0 and n_hidden_layers > 0

            assert len(optimizer_algorithms) == len(abstraction_activation_functions)

            with self.graph.as_default():

                self.training = tf.placeholder(tf.bool, name='phase_ph')

                self.n_input_features = n_input_features

                self.abstraction_activation_functions = abstraction_activation_functions

                self.n_hidden_nodes = n_hidden_nodes

                self.n_hidden_layers = n_hidden_layers

                self.keep_probability = keep_probability

                self.n_outputs = n_outputs

                self.optimizer_algorithms = optimizer_algorithms

                self.cost_function = cost_function

                self.add_summaries = add_summaries

                self.batch_normalization = batch_normalization

                self.l2_regularizer = l2_regularizer

                #
                #
                #
                self.models = [None for _ in range(len(abstraction_activation_functions))]

                self.cost_functions = [None for _ in range(len(abstraction_activation_functions))]

                #
                #
                #
                self.optimizers = [None for _ in range(len(abstraction_activation_functions))]

                self.correct_predictions = [None for _ in range(len(abstraction_activation_functions))]

                self.accuracies = [None for _ in range(len(abstraction_activation_functions))]

                #
                #
                #
                self.abstract_representation = [
                    [None for _ in range(n_hidden_layers)] for _ in range(len(abstraction_activation_functions))
                ]

                self.dense_layers = [
                    [None for _ in range(n_hidden_layers)] for _ in range(len(abstraction_activation_functions))
                ]

                self.l2_regularizers = [None for _ in range(len(abstraction_activation_functions))]

                #
                #
                #
                self.initial_weights = initial_weights

                self.initial_bias = initial_bias

                #
                #
                #
                self.raw_input = tf.placeholder(
                    tf.float32, shape=(None, self.n_input_features), name='raw_input')

                self.expected_output = tf.placeholder(
                    tf.float32, shape=(None, self.n_outputs), name='expected_output')

                self.keep_prob = tf.placeholder(
                    tf.float32, name='dropout_keep_probability')

                with tf.name_scope('abstraction_layer'):

                    for i, activation_function in enumerate(self.abstraction_activation_functions):

                        with tf.name_scope('{}_model'.format(activation_function)):

                            previous_layer_size, previous_layer = self.n_input_features, self.raw_input

                            for j in range(self.n_hidden_layers):

                                layer_name = 'hidden_{}_layer_{}'.format(activation_function, j + 1)

                                with tf.name_scope(layer_name):
                                    #
                                    # TODO refactor code to define a function to create dense layers
                                    #
                                    af = self.get_activation_function(activation_function)

                                    weight_name = 'weight_{}_h{}{}'.format(activation_function, i + 1, j + 1)

                                    if self.initial_weights is None:
                                        initial_values = tf.truncated_normal(
                                            [previous_layer_size, self.n_hidden_nodes], stddev=.1)

                                    else:
                                        initial_values = self.initial_weights[i][j]

                                    w = tf.Variable(initial_values, name=weight_name)

                                    bias_name = 'bias_{}_h{}{}'.format(activation_function, i + 1, j + 1)

                                    if self.l2_regularizers[i] is None:
                                        self.l2_regularizers[i] = tf.nn.l2_loss(w)

                                    else:
                                        self.l2_regularizers[i] = self.l2_regularizers[i] + tf.nn.l2_loss(w)

                                    if self.initial_bias is None:
                                        initial_values = tf.zeros([self.n_hidden_nodes])

                                    else:
                                        initial_values = self.initial_bias[i][j]

                                    b = tf.Variable(initial_values, name=bias_name)

                                    abstraction_layer_name = 'abstraction_{}_layer_{}'.format(
                                        activation_function, j + 1)

                                    self.dense_layers[i][j] = af(tf.add(tf.matmul(previous_layer, w), b))

                                    self.abstract_representation[i][j] = \
                                        tf.nn.dropout(self.dense_layers[i][j], self.keep_prob,
                                                      name=abstraction_layer_name if not self.batch_normalization
                                                      else 'dropout_{}_{}'.format(activation_function, j + 1))

                                    if self.batch_normalization:
                                        self.abstract_representation[i][j] = \
                                            tf.layers.batch_normalization(
                                                self.abstract_representation[i][j], training=self.training,
                                                name=abstraction_layer_name)

                                    previous_layer = self.abstract_representation[i][j]

                                    previous_layer_size = self.n_hidden_nodes

                            with tf.name_scope('output_{}_layer'.format(activation_function)):

                                weight_name = 'weight_{}_out'.format(activation_function)

                                w = tf.Variable(tf.truncated_normal(
                                    [previous_layer_size, self.n_outputs], stddev=.1), name=weight_name)

                                self.l2_regularizers[i] += tf.nn.l2_loss(w)

                                bias_name = 'bias_{}_out'.format(activation_function)

                                b = tf.Variable(tf.zeros([self.n_outputs]), name=bias_name)

                                dense_name = 'dense_model_{}'.format(activation_function)

                                self.models[i] = tf.sigmoid(tf.add(tf.matmul(previous_layer, w), b, name=dense_name))

                self.saver = tf.train.Saver()

    def __build_optimizers(self):

        with self.graph.as_default():

            if self.lr is None:
                self.lr = tf.placeholder(tf.float32, name='learning_rate')

            with tf.name_scope('optimization'):

                for i, (model, optimizer, activation_function, l2) in enumerate(zip(
                        self.models, self.optimizer_algorithms,
                        self.abstraction_activation_functions, self.l2_regularizers)):

                    self.cost_functions[i] = tf.losses.log_loss(labels=self.expected_output, predictions=model)
                    self.cost_functions[i] = tf.reduce_mean(self.cost_functions[i]) + self.l2_regularizer * l2

                    self.optimizers[i] = self.get_optimizer(optimizer)(learning_rate=self.lr)
                    self.optimizers[i] = self.optimizers[i].minimize(self.cost_functions[i])

                if self.add_summaries:
                    with tf.name_scope('loss'):
                        for i, activation_function in enumerate(self.abstraction_activation_functions):
                            tf.summary.scalar('dense_{}'.format(activation_function), self.cost_functions[i])

            if self.add_summaries:
                #
                # Create summary tensors
                #
                self.tb_train = tf.summary.merge_all()

                with tf.name_scope('auc'):

                    for af, model in zip(self.abstraction_activation_functions, self.models):

                        _, op = tf.metrics.auc(self.expected_output, model)

                        self.tb_test.append(tf.summary.scalar('dense_{}'.format(af), op))
