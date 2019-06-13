import tensorflow as tf


class PearsonCorrelationCoefficient(object):

    def __init__(self, sess=tf.Session()):
        """
        Compute the pairwise Pearson Correlation Coefficient (https://bit.ly/2ipHb9y)
        using TensorFlow (http://www.tensorflow.org) framework.

        Graph improved using the Patwie's tips described at https://bit.ly/2tbuPqX

        Author: Lucas Venezian Povoa

        :param sess a Tensorflow session

        :usage
        >>> import numpy as np
        >>> x = np.array([[1,2,3,4,5,6], [5,6,7,8,9,9]]).T
        >>> pcc = PearsonCorrelationCoefficient()
        >>> pcc.compute_score(x)
        """

        self.x_ph = tf.placeholder(tf.float16, shape=(None, None))

        x_mean, x_var = tf.nn.moments(self.x_ph, axes=0)

        x_op = self.x_ph - x_mean

        self.w_a = tf.placeholder(tf.int32)

        self.h_b = tf.placeholder(tf.int32)

        self.w_b = tf.placeholder(tf.int32)

        x_sd = tf.sqrt(x_var)

        self.x_sds = tf.reshape(tf.einsum('i,k->ik', x_sd, x_sd), shape=(-1,))

        c = tf.einsum('ij,ik->ikj', x_op, x_op)

        c = tf.reshape(c, shape=tf.stack([self.h_b, self.w_a * self.w_b]))

        self.op = tf.reshape(tf.reduce_mean(c, axis=0) / self.x_sds, shape=tf.stack([self.w_a, self.w_b]))

        self.sess = sess

    def compute_score(self, x):
        """
        Compute the Pearson Correlation Coeffiient of the x matrix. It is equivalent to `numpy.corrcoef(x.T)`
        :param x: a numpy matrix containing a variable per column
        :return: Pairwise Pearson Correlation Coefficient of the x matrix.
        """

        if len(x.shape) == 1:
            x = x.reshape((-1, 1))

        assert len(x.shape) == 2 and x.shape[1] > 0

        self.sess.run(tf.global_variables_initializer())

        return self.sess.run(self.op, feed_dict={self.x_ph: x, self.h_b: x.shape[0],
                                                 self.w_a: x.shape[1], self.w_b: x.shape[1]})


class PearsonOneVsAll(object):

    def __init__(self,sess=tf.Session()):

        self.x_ph = tf.placeholder(tf.float32, shape=(None, 1))

        self.y_ph = tf.placeholder(tf.float32, shape=(None, None))

        x_mean, x_var = tf.nn.moments(self.x_ph, axes=[0, 1])

        y_mean, y_var = tf.nn.moments(self.y_ph, axes=0)

        x_op = self.x_ph - x_mean

        y_op = self.y_ph - y_mean

        cov = tf.reduce_mean(tf.multiply(x_op, y_op), axis=0)

        self.op = cov / (tf.sqrt(x_var) * tf.sqrt(y_var))

        self.sess = sess
        
        self.sess.run(tf.global_variables_initializer())

    def compute_score(self, x, y):

        if len(x.shape) == 1:
            x = x.reshape((-1, 1))

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        assert len(x.shape) == 2 and x.shape[1] == 1

        assert len(y.shape) == 2 and y.shape[1] > 0

        assert x.shape[0] == y.shape[0]

        return self.sess.run(self.op, feed_dict={self.x_ph: x, self.y_ph: y})


def pearson_half_memory(x):

    h_x, w_x = x.shape

    x = tf.constant(x, tf.float16)

    x_mean, x_var = tf.nn.moments(x, axes=0)

    x_op = x - x_mean

    x_sds = tf.sqrt(x_var)

    x_op_t = tf.transpose(x_op)

    inner_ops = []
    x_sd = []

    for i in range(w_x - 1):
        for j in range(i + 1, w_x):
            inner_ops.append(tf.multiply(x_op_t[i], x_op_t[j]))
            x_sd.append(tf.multiply(x_sds[i], x_sds[j]))

    op = tf.reduce_mean(tf.transpose(tf.stack(inner_ops)), axis=0) / tf.stack(x_sd)

    with tf.Session() as sess:
        return sess.run(op)
