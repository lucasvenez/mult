import unittest
from model import StackedDenoisingAutoencoder, Dense


class TestSDA(unittest.TestCase):

    def test_sda(self):

        import numpy as np

        np.random.seed(13)

        w, b = [], []

        x = np.random.uniform(0, 1, (100, 10))

        y = np.where(np.random.uniform(0, 1, (100, 1)) < .5, np.ones((100, 1)), np.zeros((100, 1)))

        for af in ['sigmoid', 'tanh', 'relu']:

            sda = StackedDenoisingAutoencoder(model_name='sda_test_5_{}'.format(af))

            sda.build(10, (20, 20, 20), encoder_activation_function=af)

            sda.fit(x, steps=10)

            w.append(sda.get_initial_weights())

            b.append(sda.get_initial_biases())

        dense = Dense(model_name='dense_test_5')

        dense.build(10, 1, n_hidden_layers=3, n_hidden_nodes=20, initial_weights=w, initial_bias=b)

        dense.fit(x, y, steps=10)
