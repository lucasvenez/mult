from model import Model
import tensorflow as tf
import numpy as np
import os


class DenoisingAutoencoder(Model):

    def __init__(self, model_name=None, summaries_dir='../output/'):

        self.graph = tf.Graph()

        with self.graph.as_default():

            self.session = tf.Session(graph=self.graph)

            self.model_name = model_name

            self.input = None

            self.corrupted_input = None

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            self.add_summaries = summaries_dir is not None

            self.summaries_dir = summaries_dir

            self.keep_probability = tf.placeholder(tf.float32, name='keep_probability')

            self.batch_size = None
            
            self.layer_index = None

    def build(self, n_inputs, encoder_units=(128,), decoder_units=(128,), encoder_activation_function='sigmoid', decoder_activation_function='identity'):

        assert isinstance(encoder_units, tuple) and len(encoder_units) > 0, 'encoder_units should tuple with at least one element'
        
        assert isinstance(decoder_units, tuple) and len(decoder_units) > 0, 'decoder_units should tuple with at least one element'
        
        with self.graph.as_default():
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='input')

            with tf.name_scope('random_noise'):

                mask = tf.random_uniform(shape=tf.shape(self.input), minval=0, maxval=1, dtype=tf.float32, seed=None, name=None)

                mask = tf.where(mask <= self.keep_probability, tf.ones_like(self.input, dtype=tf.float32), tf.zeros_like(self.input, dtype=tf.float32))

                self.corrupted_input = tf.multiply(self.input, mask)

            with tf.name_scope('encoder'):

                self.encoder = self.corrupted_input
                
                for layer_index, units in enumerate(encoder_units):
                
                    name = 'last_encoder' if layer_index == len(encoder_units) - 1 else 'encoder_{}'.format(layer_index + 1)
               
                    self.encoder = tf.layers.dense(self.encoder, units, kernel_initializer=tf.truncated_normal_initializer())
                    
                    self.encoder = self.get_activation_function(encoder_activation_function)(self.encoder, name=name)

            with tf.name_scope('decoder'):

                self.decoder = self.encoder
                
                for layer_index, units in enumerate(decoder_units):
                
                    self.decoder = tf.layers.dense(self.encoder, n_inputs, kernel_initializer=tf.truncated_normal_initializer())

                    self.decoder = self.get_activation_function(encoder_activation_function)(self.decoder, name='decoder_{}'.format(layer_index + 1))
                    
                self.decoder = tf.layers.dense(self.encoder, n_inputs, kernel_initializer=tf.truncated_normal_initializer())

                self.decoder = self.get_activation_function(decoder_activation_function)(self.decoder, name='last_encoder')

            self.layer_index = layer_index + 1
                
            self.saver = tf.train.Saver()

            
    def fit(self, x, keep_probability=0.75, learning_rate=1e-4, steps=10000, batch_size=None, shuffle=True, optimizer='sgd', loss='mse'):
        '''
        
        '''
        assert steps > 0, 'steps should be an integer greater than zero'

        assert batch_size is None or 0 < batch_size <= x.shape[0], 'bath should be none or an integer between zero (exclusive) and number of input features (inclusive)'

        self.best_error = np.inf
        
        with self.graph.as_default():

            self.__build_optimizer(optimizer, loss)

            test_writer, test_results = None, None

            if batch_size is None:
                batch_size = x.shape[0]

            self.batch_size = batch_size

            with self.graph.as_default():

                self.session.run(tf.global_variables_initializer())

                self.session.run(tf.local_variables_initializer())

                if self.add_summaries:
                    test_writer = tf.summary.FileWriter(self.summaries_dir + '/{}'.format(self.model_name), tf.get_default_graph())

                n_rows = x.shape[0]

                index = np.array(list(range(n_rows)), dtype=np.int)

                j, logdata = 0, None

                logs = []
                
                for step in range(steps):

                    current_block = 0

                    while current_block < n_rows:

                        if shuffle:
                            np.random.shuffle(index)

                        batch = list(range(current_block, (min(current_block + batch_size, n_rows))))

                        loss = self.session.run([self.optimizer, self.loss],
                                                        feed_dict={self.input: x[index[batch], :],
                                                                   self.learning_rate: learning_rate,
                                                                   self.keep_probability: keep_probability})[1]
                        
                        logs.append(loss)
                        
                        current_block += batch_size

                        j += 1
                        
                        break

                    if self.add_summaries:
                        test_writer.add_summary(self.session.run(tf.summary.scalar('mean ' + loss, np.mean(logs))), step)

                    if step == steps - 1:
                        self.saver.save(self.session, '{0}/{1}/{1}'.format(self.summaries_dir, self.model_name), global_step=step)

                    if self.best_error > np.mean(logs):
                        self.best_error = np.mean(logs)
                        self.saver.save(self.session, '{0}/{1}/{1}__BESTONE__'.format(self.summaries_dir, self.model_name), global_step=step)

    def predict(self, x):

        if self.batch_size is None:
            self.batch_size = 1000

        x_line = None

        start, end = 0, min(self.batch_size, x.shape[0])

        while start < x.shape[0]:

            with self.graph.as_default():
                x_ = self.session.run([self.decoder], feed_dict={self.input: x[start:end, :], self.keep_probability: 1.0})[0]

            if x_line is None:
                x_line = x_

            else:
                x_line = np.concatenate((x_line, x_), axis=0)

            start, end = end, min(x.shape[0], end + self.batch_size)

        return x_line

    def encode(self, x):

        if self.batch_size is None:
            self.batch_size = 1000

        x_line = None

        start, end = 0, min(self.batch_size, x.shape[0])

        with self.graph.as_default():

            while start < x.shape[0]:

                x_ = self.session.run([self.encoder], feed_dict={self.input: x[start:end, :],
                                                                 self.keep_probability: 1.0})[0]

                if x_line is None:
                    x_line = x_

                else:
                    x_line = np.concatenate((x_line, x_), axis=0)

                start, end = end, min(x.shape[0], end + self.batch_size)

        return x_line

    def transform(self, x):

        if self.batch_size is None:
            self.batch_size = 1000

        x_line = None

        start, end = 0, min(self.batch_size, x.shape[0])

        while start < x.shape[0]:

            with self.graph.as_default():
                x_ = self.session.run([tf.reduce_sum(tf.square(self.input - self.decoder), axis=1)],
                                      feed_dict={self.input: x[start:end, :], self.keep_probability: 1.0})[0]

            if x_line is None:
                x_line = x_

            else:
                x_line = np.concatenate((x_line, x_), axis=0)

            start, end = end, min(x.shape[0], end + self.batch_size)

        return x_line

    def get_error(self, x):

        if self.batch_size is None:
            self.batch_size = 1000

        x_line = None

        start, end = 0, min(self.batch_size, x.shape[0])

        while start < x.shape[0]:

            with self.graph.as_default():
                x_ = self.session.run([self.input - self.decoder],
                                      feed_dict={self.input: x[start:end, :], self.keep_probability: 1.0})[0]

            if x_line is None:
                x_line = x_

            else:
                x_line = np.concatenate((x_line, x_), axis=0)

            start, end = end, min(x.shape[0], end + self.batch_size)

        return x_line

    def fit_encode(self, x, keep_probability=0.75, learning_rate=1e-4, steps=1000, batch_size=None, shuffle=True):

        self.fit(x, keep_probability, learning_rate, steps, batch_size, shuffle)

        return self.encode(x)

    def fit_transform(self, x, keep_probability=0.75, learning_rate=1e-4, steps=1000, batch_size=None, shuffle=True):

        self.fit(x, keep_probability, learning_rate, steps, batch_size, shuffle)

        return self.transform(x)

    def load(self, model_path):

        if os.path.exists('{}.meta'.format(model_path)) and os.path.isfile('{}.meta'.format(model_path)):

            with self.graph.as_default():

                self.saver = tf.train.import_meta_graph('{}.meta'.format(model_path))

                self.saver.restore(self.session, tf.train.latest_checkpoint(os.path.dirname(model_path)))

                self.input = tf.get_default_graph().get_tensor_by_name('input:0')

                self.keep_probability = tf.get_default_graph().get_tensor_by_name('keep_probability_1:0')

                self.encoder = tf.get_default_graph().get_tensor_by_name('encoder/last_encoder:0')

                self.decoder = tf.get_default_graph().get_tensor_by_name('decoder/last_decoder:0')

    def __build_optimizer(self, optimizer, loss):

        with tf.name_scope('optimization'):

            with tf.name_scope('loss'):

                self.loss = self.get_loss(loss)(self.input, self.decoder)

                self.loss = tf.reduce_mean(self.loss)

                tf.summary.scalar('dae', self.loss)

            self.optimizer = self.get_optimizer(optimizer)(learning_rate=self.learning_rate)

            self.optimizer = self.optimizer.minimize(self.loss, name='optimizer')

        if self.add_summaries:
            #
            # Create summary tensors
            #
            self.merged = tf.summary.merge_all()