import tensorflow as tf
import numpy as np
import os


class ConvDense:

    VALID_OPTIMIZERS = {'sgd': tf.train.GradientDescentOptimizer, 'ftrl': tf.train.FtrlOptimizer,
                        'adam': tf.train.AdamOptimizer, 'adadelta': tf.train.AdadeltaOptimizer,
                        'adagrad': tf.train.AdagradOptimizer, 'rmsprop': tf.train.RMSPropOptimizer}

    def __init__(self, abstraction_layer=None, session=None, model_name='C0000', summaries_dir='output/'):

        self.graph = tf.Graph()

        if session is None:
            self.sess = tf.Session(graph=self.graph)

        self.summaries_dir = summaries_dir
        self.saver = None
        self.abstraction_layer = abstraction_layer
        self.training = False

        #
        # Placeholders
        #
        self.input = None
        self.expected_output = None
        self.keep_prob = None
        self.lr = None

        #
        # Model Parameters
        #
        self.keep_probability = None
        self.optimizer_algorithm = None
        self.learning_rate = None
        #
        # Model meta-parameters
        #
        self.add_summaries = True if summaries_dir is not None else False
        self.model_name = model_name

        #
        # Accuracy tensors
        #
        self.correct_prediction = None
        
        #
        # Dashboard
        #
        self.tb_test = []

    def build(self, n_models=3, n_neurons_per_layer=100, n_layers=3, n_outputs=1, optimizer_algorithm='sgd', keep_probability=0.5):

        with self.graph.as_default():
            #
            #
            #
            self.n_outputs = n_outputs
            self.n_models = n_models
            self.n_neurons_per_layer = n_neurons_per_layer
            self.n_layers = n_layers
            self.keep_probability = keep_probability
            self.optimizer_algorithm = optimizer_algorithm

            #
            # Placeholders
            #
            self.keep_prob = tf.placeholder(tf.float32, name='keep_probability_ph')
            self.training = tf.placeholder(tf.bool, name='training_ph')

            self.input = tf.placeholder(tf.float32, shape=(None, n_models, n_neurons_per_layer, n_layers, 1), name='input')

            with tf.name_scope('conv1'):
                self.conv1_1 = tf.layers.conv3d(inputs=self.input, filters=16, kernel_size=[3, 8, 1], strides=1, padding='same', name="conv1_1",
                                                activation=tf.nn.relu)
                print(self.conv1_1.shape)

                self.conv1_2 = tf.layers.conv3d(self.conv1_1, filters=32, kernel_size=[3, 8, 2], strides=1, padding='same', name="conv1_2",
                                                activation=tf.nn.relu)
                print(self.conv1_2.shape)

                self.conv1_3 = tf.layers.conv3d(self.conv1_2, filters=32, kernel_size=[3, 8, 3], strides=1, padding='same', name="conv1_3",
                                                activation=tf.nn.relu)
                print(self.conv1_3.shape)

                self.pool1 = tf.layers.max_pooling3d(self.conv1_3, pool_size=[1, 4, 1], strides=[1, 4, 1], name='pool1')
                print(self.pool1.shape)

            with tf.name_scope('conv2'):
                self.conv2_1 = tf.layers.conv3d(inputs=self.pool1, filters=64, kernel_size=[3, 8, 1],
                                                padding='same', name="conv2_1", activation=tf.nn.relu)
                print(self.conv2_1.shape)

                self.conv2_2 = tf.layers.conv3d(self.conv2_1, filters=64, kernel_size=[3, 8, 2], padding='same', name="conv2_2",
                                                activation=tf.nn.relu)
                print(self.conv2_2.shape)

                self.pool2 = tf.layers.max_pooling3d(self.conv2_2, pool_size=[2, 2, 1], strides=[1, 2, 1], name='pool2')
                print(self.pool2.shape)

            with tf.name_scope('conv3'):
                self.conv3_1 = tf.layers.conv3d(inputs=self.pool2, filters=128, kernel_size=[2, 2, 1],
                                                padding='same', name="conv3_1", activation=tf.nn.relu)
                print(self.conv3_1.shape)

                self.conv3_2 = tf.layers.conv3d(self.conv3_1, filters=128, kernel_size=[2, 2, 2],
                                                padding='same', name="conv3_2", activation=tf.nn.relu)
                print(self.conv3_2.shape)

                self.pool3 = tf.layers.max_pooling3d(self.conv3_2, pool_size=[2, 8, 2], strides=[2, 8, 2], name='pool3')
                print(self.pool3.shape)

            with tf.name_scope('dense_layer'):
                _, w, x, y, z = self.pool3.shape

                self.fc1 = tf.layers.dense(tf.reshape(self.pool3, (-1, w * x * y * z)), 512, name='dense1', activation=tf.nn.relu)
                self.fc1 = tf.layers.dropout(self.fc1, rate=self.keep_prob, training=self.training)

                self.fc = tf.layers.dense(self.fc1, 1, name='output', activation=tf.nn.sigmoid)

            self.saver = tf.train.Saver()

    def build_optimizer(self):

        with self.graph.as_default():

            self.expected_output = tf.placeholder(tf.int64, shape=(None, self.n_outputs), name='expected_output')
            self.lr = tf.placeholder(tf.float32, name='learning_rate_ph')

            with tf.name_scope('optimization'):

                self.cost_function = tf.reduce_mean(tf.losses.log_loss(self.expected_output, self.fc))

                self.optimizer = self.VALID_OPTIMIZERS[self.optimizer_algorithm](learning_rate=self.lr).minimize(self.cost_function)

                if self.add_summaries:
                    tf.summary.scalar('cross_entropy', tf.reduce_mean(self.cost_function))
                    #
                    # Create summary tensors
                    #
                    self.merged = tf.summary.merge_all()

                    with tf.name_scope('auc'):

                        _, op = tf.metrics.auc(self.expected_output, self.fc)

                        self.tb_test.append(tf.summary.scalar('convdense', op))

    def fit(self, x, y, x_test=None, y_test=None, learning_rate=1e-5, steps=1000, batch_size=1000, shuffle=True):

        assert steps > 0

        assert 0 < batch_size <= x.shape[0]

        self.build_optimizer()

        print('Optimizing model')

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

            self.sess.run(tf.global_variables_initializer())

            self.sess.run(tf.local_variables_initializer())

            if self.add_summaries:
                self.test_writer = tf.summary.FileWriter(self.summaries_dir + '/{}'.format(self.model_name), tf.get_default_graph())

            n_rows = x.shape[0]

            index = np.array(list(range(n_rows)), dtype=np.int)

            j = 0

            if self.abstraction_layer is not None:
                x_test = self.abstraction_layer.transform(x_test)

            for step in range(steps):

                print('Step {} of {}'.format(step + 1, steps))

                current_block = 0

                while (current_block < n_rows):

                    if shuffle:
                        np.random.shuffle(index)

                    batch = list(range(current_block, (min(current_block + batch_size, n_rows))))

                    if self.abstraction_layer is not None:
                        x_transformed = self.abstraction_layer.transform(x[index[batch], :])
                    else:
                        x_transformed = x[index[batch], :]

                    run_list = [self.optimizer]

                    if self.add_summaries:
                        run_list = [self.merged] + run_list

                    self.sess.run(run_list,
                                  feed_dict={self.input: x_transformed, self.expected_output: y[index[batch], :],
                                             self.keep_prob: self.keep_probability, self.lr: learning_rate, self.training: True})

                    current_block += batch_size

                    j += 1

                self.saver.save(self.sess, 'output/{0}/{0}'.format(self.model_name), global_step=step)

                if self.add_summaries:

                    run_list = self.tb_test

                    test_results = self.sess.run(run_list,
                                   feed_dict={self.input: x_test, self.expected_output: y_test,
                                              self.keep_prob: 1., self.training: False})

                    self.test_writer.add_summary(test_results[0], step)

    def predict(self, x):

        if self.abstraction_layer is not None:
            x = self.abstraction_layer.tranform(x)

        with self.graph.as_default():
            return self.sess.run(self.fc, feed_dict={self.input: x, self.keep_prob: 1., self.training: False})


    def load(self, model_path):
        #
        # FIXME It should be tested
        #
        if os.path.exists('{}.meta'.format(model_path)) and os.path.isfile('{}.meta'.format(model_path)):

            with self.graph.as_default():

                self.saver = tf.train.import_meta_graph('{}.meta'.format(model_path))

                self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(model_path)))

                self.input = tf.get_default_graph().get_tensor_by_name('input:0')

                self.expected_output = tf.get_default_graph().get_tensor_by_name('expected_output:0')

                self.keep_prob = tf.get_default_graph().get_tensor_by_name('keep_probability_ph:0')

                self.training = tf.get_default_graph().get_tensor_by_name('training_ph:0')

                self.fc = tf.get_default_graph().get_tensor_by_name('output:0')
