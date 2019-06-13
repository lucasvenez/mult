import tensorflow as tf


class KMeans(object):

    def __init__(self, num_clusters, verbose=0):

        self.num_clusters = num_clusters

        self.kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=self.num_clusters, use_mini_batch=False)

        self.x = None

        self.cluster_centers = None

        self.verbose = verbose

    def fit(self, x, num_iterations=10):

        def input_fn():
            return tf.train.limit_epochs(tf.convert_to_tensor(x, dtype=tf.float32), num_epochs=1)

        previous_centers = None

        for _ in range(num_iterations):

            self.kmeans.train(input_fn)

            self.cluster_centers = self.kmeans.cluster_centers()

            if previous_centers is not None and self.verbose > 0:
                print('delta:', self.cluster_centers - previous_centers)

            previous_centers = self.cluster_centers

            if self.verbose > 0:
                print('score:', self.kmeans.score(input_fn))

        if self.verbose > 0:
            print('cluster centers:', self.cluster_centers)

        return self.kmeans.score(input_fn)

    def predict(self, x):

        def input_fn():
            return tf.train.limit_epochs(tf.convert_to_tensor(x, dtype=tf.float32), num_epochs=1)

        #
        # map the input points to their clusters
        #
        cluster_indices = list(self.kmeans.predict_cluster_index(input_fn))

        clusters = []

        for i, point in enumerate(x):

            clusters.append(cluster_indices[i])

        return clusters

    def transform(self, x):

        def input_fn():
            return tf.train.limit_epochs(tf.convert_to_tensor(x, dtype=tf.float32), num_epochs=1)

        #
        # map the input points to their clusters
        #
        cluster_indices = list(self.kmeans.predict_cluster_index(input_fn))

        centers = []

        for i, point in enumerate(x):

            cluster_index = cluster_indices[i]

            centers.append(tuple(self.cluster_centers[cluster_index]))

        return centers