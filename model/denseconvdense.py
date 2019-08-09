import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


class Deconvence(object):
    
    def __init__(self, model_name='C0000', summaries_dir='output'):

        self.graph = tf.Graph()

        self.sess = tf.Session(graph=self.graph)

        self.summaries_dir = summaries_dir
        self.saver = None
        self.abstraction_layer = None
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
        #
        #
        self.optimizer = None

    def get_activation_function(self, name):

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

    def get_optimizer(self, name):

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

    def get_loss(self, name):

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


    def build(self,
              n_features,
              n_outputs=1,
              abstraction_activation_functions=('relu', 'tanh', 'sigmoid'),
              abstraction_n_hidden_layers=3,
              abstraction_n_neurons_per_hidden_layer=128,
              processing_n_hidden_layers=3,
              processing_n_neurons_per_hidden_layer=1024,
              processing_activation_function='relu',
              output_activation_function='sigmoid',
              optimizer_algorithm='adadelta',
              keep_probability=0.5,
              loss_function='logloss',
              batch_normalization=True):
        """

        :param abstraction_activation_functions:
        :param abstraction_hidden_layers:
        :param abstraction_neurons_per_hidden_layer:
        :param processing_hidden_layers:
        :param processing_output_neurons:
        :param processing_activation_function:
        :param optimizer_algorithm:
        :param keep_probability:
        :return:
        """
        with self.graph.as_default():
            
            self.aucs = [tf.placeholder(tf.float32) for _ in range(n_outputs)]
            
            #
            #
            #
            self.n_features = self.n_outputs = n_outputs
            self.n_outputs = n_outputs

            self.abstraction_activation_functions = abstraction_activation_functions
            self.abstraction_n_hidden_layers = abstraction_n_hidden_layers
            self.abstraction_n_neurons_per_hidden_layer = abstraction_n_neurons_per_hidden_layer

            self.processing_n_hidden_layers = processing_n_hidden_layers
            self.processing_n_neurons_per_hidden_layer = processing_n_neurons_per_hidden_layer
            self.processing_activation_function = processing_activation_function

            self.keep_probability = keep_probability
            self.optimizer_algorithm = optimizer_algorithm
            self.loss_function = loss_function

            #
            # Placeholders
            #
            self.keep_prob = tf.placeholder(tf.float32, name='keep_probability_ph')
            self.training = tf.placeholder(tf.bool, name='training_ph')

            self.input = tf.placeholder(tf.float32, shape=(None, n_features), name='input')

            with tf.name_scope('abstraction_layer'):

                fc, abstraction_stack = None, []

                for i, af in enumerate(self.abstraction_activation_functions):

                    fc = self.input

                    abstraction_stack.append([])
                    
                    ll = 1
                    
                    for j in range(self.abstraction_n_hidden_layers):

                        layer_name = 'abstraction_layer_{}_{}'.format(af[:4], j + 1)
                        
                        if self.keep_prob is not None:
                            fc = tf.layers.dropout(fc, name='dropout_{}'.format(layer_name),
                                                   rate=self.keep_prob if ll > 1 else tf.constant(.25), training=self.training)
                            ll += 1

                            
                        if batch_normalization:
                            fc = tf.layers.batch_normalization(fc, name='bathnormalization_{}'.format(layer_name), 
                                                               training=self.training)
                        
                        fc = tf.layers.dense(fc, self.abstraction_n_neurons_per_hidden_layer,
                                             name=layer_name, activation=self.get_activation_function(af))

                        abstraction_stack[i].append(fc)

                for i in range(len(abstraction_stack)):
                    abstraction_stack[i] = tf.stack(abstraction_stack[i], 0)

                abstraction_stack = tf.stack(abstraction_stack, 2)

                self.abstraction_layer = tf.expand_dims(tf.transpose(abstraction_stack, [1, 0, 3, 2]), axis=4)

            with tf.name_scope('convolutional_layer'):

                with tf.name_scope('conv1'):

                    self.conv1_1 = tf.layers.conv3d(inputs=self.abstraction_layer, filters=2,
                                                    kernel_size=[abstraction_n_hidden_layers, 8, 1],
                                                    strides=1, padding='same', name="conv1_1", activation=tf.nn.relu)
                    print(self.conv1_1.shape)

                    self.conv1_2 = tf.layers.conv3d(inputs=self.conv1_1, filters=4,
                                                    kernel_size=[abstraction_n_hidden_layers, 8, 2], strides=1,
                                                    padding='same', name="conv1_2", activation=tf.nn.relu)
                    print(self.conv1_2.shape)

                    self.conv1_3 = tf.layers.conv3d(inputs=self.conv1_2, filters=16,
                                                    kernel_size=[abstraction_n_hidden_layers, 8, 3], strides=1,
                                                    padding='same', name="conv1_3", activation=tf.nn.relu)
                    print(self.conv1_3.shape)

                    self.pool1 = tf.layers.average_pooling3d(inputs=self.conv1_3, pool_size=[1, 4, 1],
                                                             strides=[1, 4, 1], name='pool1')
                    print(self.pool1.shape)
                    
                    if self.keep_prob is not None:
                        self.pool1 = tf.layers.dropout(self.pool1, name='dropout_pool1', rate=self.keep_prob, training=self.training)
                        
                    if batch_normalization:
                        self.pool1 = tf.layers.batch_normalization(self.pool1, name='bathnormalization_pool1', 
                                                                   training=self.training)

                with tf.name_scope('conv2'):
                    self.conv2_1 = tf.layers.conv3d(inputs=self.pool1, filters=32,
                                                    kernel_size=[abstraction_n_hidden_layers, 8, 1],
                                                    padding='same', name="conv2_1", activation=tf.nn.tanh)
                    print(self.conv2_1.shape)

                    self.conv2_2 = tf.layers.conv3d(inputs=self.conv2_1, filters=64,
                                                    kernel_size=[abstraction_n_hidden_layers, 8, 2],
                                                    padding='same', name="conv2_2", activation=tf.nn.relu)
                    print(self.conv2_2.shape)

                    self.pool2 = tf.layers.average_pooling3d(inputs=self.conv2_2, pool_size=[2, 2, 1],
                                                             strides=[1, 2, 1], name='pool2')
                    print(self.pool2.shape)
                    
                    if self.keep_prob is not None:
                        self.pool2 = tf.layers.dropout(self.pool2, name='dropout_pool2', rate=self.keep_prob, training=self.training)
                        
                    if batch_normalization:
                        self.pool2 = tf.layers.batch_normalization(self.pool2, name='bathnormalization_pool2', 
                                                                   training=self.training)

                with tf.name_scope('conv3'):

                    self.conv3_1 = tf.layers.conv3d(inputs=self.pool2, filters=128, kernel_size=[2, 2, 1],
                                                    padding='same', name="conv3_1", activation=tf.nn.relu)
                    print(self.conv3_1.shape)

                    self.conv3_2 = tf.layers.conv3d(inputs=self.conv3_1, filters=256, kernel_size=[2, 2, 2],
                                                    padding='same', name="conv3_2", activation=tf.nn.relu)
                    print(self.conv3_2.shape)

                    self.pool3 = tf.layers.average_pooling3d(inputs=self.conv3_2, pool_size=[2, 2, 2],
                                                             strides=[2, 2, 2], name='pool3')
                    print(self.pool3.shape)
                    
                    if self.keep_prob is not None:
                        self.pool3 = tf.layers.dropout(self.pool3, name='dropout_pool3', rate=self.keep_prob, training=self.training)
                        
                    if batch_normalization:
                        self.pool3 = tf.layers.batch_normalization(self.pool3, name='bathnormalization_pool3', 
                                                                   training=self.training)

            with tf.name_scope('processing_layer'):

                shape = 1

                for i in self.pool3.shape[1:]:
                    shape *= int(i)

                fcp = tf.reshape(self.pool3, (-1, shape))

                for i in range(processing_n_hidden_layers):

                    layer_name = 'processing_hidden_layer_{}'.format(i + 1)

                    fcp = tf.layers.dense(inputs=fcp,
                                          units=processing_n_neurons_per_hidden_layer,
                                          name=layer_name,
                                          activation=self.get_activation_function(processing_activation_function))

                    if self.keep_prob is not None:
                        fcp = tf.layers.dropout(inputs=fcp, name='dropout_{}'.format(layer_name),
                                                rate=self.keep_prob, training=self.training)
                        
                    if batch_normalization:
                        fcp = tf.layers.batch_normalization(fcp, name='bathnormalization_{}'.format(layer_name), 
                                                                   training=self.training)

                self.fc = tf.layers.dense(inputs=fcp, units=n_outputs, name='output_layer',
                                          activation=self.get_activation_function(output_activation_function))

            self.saver = tf.train.Saver()

    def build_optimizer(self):

        with self.graph.as_default():
            
            self.feed_aucs = []

            self.expected_output = tf.placeholder(tf.int64, shape=(None, self.n_outputs), name='expected_output')
            self.lr = tf.placeholder(tf.float32, name='learning_rate_ph')

            with tf.name_scope('optimization'):

                self.cost_function = self.get_loss(self.loss_function)(self.expected_output, self.fc)

                self.cost_function = tf.reduce_mean(self.cost_function)

                self.optimizer = self.get_optimizer(self.optimizer_algorithm)(learning_rate=self.lr).minimize(self.cost_function)

                if self.add_summaries:
                    tf.summary.scalar('cross_entropy', self.cost_function)
                    tf.summary.histogram('output_histogram', self.fc)

            if self.add_summaries:
                #
                # Create summary tensors
                #
                self.merged = tf.summary.merge_all()

                with tf.name_scope('auc'):
                    self.feed_aucs.append(tf.summary.scalar('denseconvdense', self.aucs[-1]))

    def fit(self, x, y, x_test=None, y_test=None, learning_rate=1e-5, steps=1000, batch_size=1000, shuffle=True):

        assert steps > 0

        assert 0 < batch_size <= x.shape[0]

        self.batch_size = batch_size
        
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

            n_rows = x.shape[0]
            
            index = np.array(list(range(n_rows)), dtype=np.int)

            for step in range(steps):

                print('Step {} of {}'.format(step + 1, steps))

                current_block = 0

                while current_block < n_rows:

                    if shuffle:
                        np.random.shuffle(index)

                    batch = list(range(current_block, min(current_block + batch_size, n_rows)))

                    run_list = [self.optimizer]

                    self.sess.run(run_list,
                                  feed_dict={self.input: x[index[batch], :], self.expected_output: y[index[batch], :],
                                             self.keep_prob: self.keep_probability, self.lr: learning_rate, self.training: True})

                    current_block += batch_size
                
                if step % 100 == 0 or step == steps - 1:
                    self.saver.save(self.sess, self.summaries_dir + '/{0}/{0}'.format(self.model_name), global_step=step)

    def predict(self, x):

        with self.graph.as_default():
            
            start, end = 0, self.batch_size
            
            y_hat = []
            
            while start < x.shape[0]:
                
                y_ = self.sess.run(self.fc, feed_dict={self.input: x[start:end],
                                                       self.keep_prob: 1.,
                                                       self.training: False})
                
                y_hat += list(y_)
                
                start, end = end, min(end + self.batch_size, x.shape[0])

            return y_hat

