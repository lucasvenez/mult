from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import OneClassSVM
from model import ConvDense, Dense, DenseConvDense


def dcd(train_x, train_y, test_x, test_y=None, selector='', fold=0, n_neurons=128):

    model = Dense(model_name='MMDEN_{}_{}_{}'.format(selector.upper(), n_neurons, fold))

    model.build(n_input_features=train_x.shape[1], n_outputs=train_y.shape[1],
                abstraction_activation_functions=('sigmoid', 'tanh', 'relu'), n_hidden_layers=3, n_hidden_nodes=n_neurons,
                keep_probability=.5, initialization=None, optimizer_algorithms=('adagrad', 'rmsprop', 'rmsprop'),
                l2_regularizer=0.)

    model.fit(train_x, train_y, test_x, test_y, steps=1000, batch_size=100)

    train_x = model.transform(train_x).reshape((-1, 3, n_neurons, 3, 1))

    test_x = model.transform(test_x).reshape((-1, 3, n_neurons, 3, 1))

    del model

    conv = ConvDense(model_name='MMDCD_{}_{}'.format(selector.upper(), fold))

    conv.build(n_models=3, n_neurons_per_layer=n_neurons)

    conv.fit(x=train_x, y=train_y, x_test=test_x, y_test=test_y, learning_rate=1e-4, steps=1000, batch_size=100)

    y_hat = conv.predict(test_x)

    return y_hat


def dcd2(train_x, train_y, test_x, test_y=None, selector='', fold=0, n_neurons=64):

    conv = DenseConvDense(model_name='MMDCD_FULL_MMX_LOG_{}_{}_{}'.format(selector.upper(), n_neurons, fold if fold > 9 else '0{}'.format(fold)))

    conv.build(n_features=train_x.shape[1], loss_function='log', abstraction_n_neurons_per_hidden_layer=n_neurons)

    conv.fit(x=train_x, y=train_y, x_test=test_x, y_test=test_y, learning_rate=1e-3, steps=1000, batch_size=100)

    y_hat = conv.predict(test_x)

    return y_hat


def mlp(train_x, train_y, test_x):

    model = MLPClassifier(hidden_layer_sizes=(128, 128, 128), solver='adam', max_iter=1000)

    model.fit(train_x, train_y)

    return model.predict(test_x)


def bnl(train_x, train_y, test_x):

    model = BernoulliRBM()

    model.fit(train_x, train_y)

    return model.score_samples(test_x)


def rfc(train_x, train_y, test_x):

    model = RandomForestClassifier()

    model.fit(train_x, train_y)

    return model.predict(test_x)


def abc(train_x, train_y, test_x):

    model = AdaBoostClassifier()

    model.fit(train_x, train_y)

    return model.predict(test_x)


def etc(train_x, train_y, test_x):

    model = ExtraTreesClassifier()

    model.fit(train_x, train_y)

    return model.predict(test_x)


def bgc(train_x, train_y, test_x):

    model = BaggingClassifier()

    model.fit(train_x, train_y)

    return model.predict(test_x)


def svm(train_x, train_y, test_x):

    model = OneClassSVM()

    model.fit(train_x, train_y)

    return model.predict(test_x)


def xgb(train_x, train_y, test_x):

    model = None #ExtremeGradientBoosting()

    model.fit(train_x, train_y)

    return model.predict(test_x)