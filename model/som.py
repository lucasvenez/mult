import tensorflow as tf
import numpy as np


class SOM(object):
    
    '''
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    '''

    def __init__(self, model_name, output_dir='output'):
        
        #
        # To check if the SOM has been trained
        #
        self._trained = False
        
        #
        # INITIALIZE GRAPH
        #
        self.graph = tf.Graph()
        
        self.sessoin = tf.Session(graph=self.graph)
        
        #
        #
        #
        self.output_dir = output_dir
        
        #
        #
        #
        self.model_name = model_name
        
        #
        #
        #
        self.add_summaries = output_dir is not None
        
        #
        #
        #
        self.tensorboard = None
        
        self.tensorboard_writer = None
            
    def build(self, m, n, dim, alpha=None, sigma=None):
        '''
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        '''
        
        #
        # Assign required variables first
        #
        self._m, self._n = m, n
        
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
            
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)

        #
        # POPULATE GRAPH WITH NECESSARY COMPONENTS
        #
        with self.graph.as_default():
            
            with tf.name_scope('SOM'):
                
                # VARIABLES AND CONSTANT OPS FOR DATA STORAGE

                #
                # Randomly initialized weightage vectors for all neurons,
                # stored together as a matrix Variable of size [m*n, dim]
                #
                self._weightage_vects = tf.Variable(tf.random_normal([m * n, dim]))

                #
                # Matrix of size [m*n, 2] for SOM grid locations
                # of neurons
                #
                self._location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
            
            #
            #
            #
            with tf.name_scope('optimization'):
                
                # PLACEHOLDERS FOR TRAINING INPUTS
                # We need to assign them as attributes to self, since they
                # will be fed in during training

                #
                # The training vector
                #
                self._vect_input = tf.placeholder(tf.float32, [dim])
                
                #
                #
                #
                self._n_interations_ph = tf.placeholder(tf.float32)

                #
                # Iteration number
                #
                self._iter_input = tf.placeholder(tf.float32)

                #
                # CONSTRUCT TRAINING OP PIECE BY PIECE
                # Only the final, 'root' training op needs to be assigned as
                # an attribute to self, since all the rest will be executed
                # automatically during training
                #

                #
                # To compute the Best Matching Unit given a vector
                # Basically calculates the Euclidean distance between every
                # neuron's weightage vector and the input, and returns the
                # index of the neuron which gives the least value
                #
                bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                    tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                        [self._vect_input for i in range(m * n)])), 2), 1)), 0)

                #
                # This will extract the location of the BMU based on the BMU's index
                #
                slice_input = tf.cast(tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]])), tf.int32)
                
                bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.constant(np.array([1, 2]))), [2])

                #
                # To compute the alpha and sigma values based on iteration
                # number
                #
                learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_interations_ph))
                
                _alpha_op = tf.multiply(alpha, learning_rate_op)

                _sigma_op = tf.multiply(sigma, learning_rate_op)

                #
                # Construct the op that will generate a vector with learning
                # rates for all neurons, based on iteration number and location
                # wrt BMU.
                #
                bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                    self._location_vects, tf.stack([bmu_loc for i in range(m * n)])), 2), 1)

                neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                    bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))

                learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

                #
                # Finally, the op that will use learning_rate_op to update
                # the weightage vectors of all neurons based on a particular
                # input
                #
                learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                    learning_rate_op, np.array([i]), 
                    np.array([1])), [dim]) for i in range(m * n)])

                weightage_delta = tf.multiply(
                    learning_rate_multiplier,
                    tf.subtract(tf.stack([self._vect_input for i in range(m * n)]), self._weightage_vects))

                new_weightages_op = tf.add(self._weightage_vects, weightage_delta)

                self._training_op = tf.assign(self._weightage_vects, new_weightages_op)
                
                if self.add_summaries:
                    with tf.name_scope('distance'):
                           self.tensorboard = tf.summary.scalar('som', tf.reduce_mean(self._training_op))
            
            with tf.name_scope('session'):
                
                #
                # Saver node
                #
                self.saver = tf.train.Saver()

                #
                # INITIALIZE SESSION
                #
                self.session = tf.Session()

                #
                # INITIALIZE VARIABLES
                #
                self.session.run(tf.global_variables_initializer())

    def _neuron_locations(self, m, n):
        '''
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        '''
        #
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        #
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def fit(self, input_vects, steps):
        '''
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        '''
        with self.graph.as_default():
            
            #
            #
            #
            if self.add_summaries:
                
                file_path = '{}/{}'.format(self.output_dir, self.model_name)
                
                self.tensorboard_writer = tf.summary.FileWriter(file_path, tf.get_default_graph())
            
            #
            #
            #
            self._n_iterations = abs(int(steps))

            #
            # Training iterations
            #
            
            runs = [self._training_op]
            
            if self.add_summaries:
                runs += [self.tensorboard]
            
            for step in range(self._n_iterations):
                
                #
                # Train with each vector one by one
                #
                for input_vect in input_vects:
                    output = self.session.run(runs, feed_dict={self._vect_input: input_vect, 
                                                               self._iter_input: step,
                                                               self._n_interations_ph: steps})
                    
                if self.add_summaries:

                    file_path = '{0}/{1}/{1}'.format(self.output_dir, self.model_name)

                    self.saver.save(self.session, file_path, global_step=step)
                    self.tensorboard_writer.add_summary(output[-1], step)

            #
            # Store a centroid grid for easy retrieval later on
            #
            centroid_grid = [[] for i in range(self._m)]

            self._weightages = list(self.session.run(self._weightage_vects))

            self._locations = list(self.session.run(self._location_vects))

            for i, loc in enumerate(self._locations):
                centroid_grid[loc[0]].append(self._weightages[i])

            self._centroid_grid = centroid_grid

            self._trained = True

    def get_centroids(self):
        '''
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        '''
        
        if not self._trained:
            raise ValueError("SOM not trained yet")
            
        return self._centroid_grid

    def map_vects(self, input_vects):
        '''
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        '''

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            
            to_return.append(self._locations[min_index])

        return to_return
