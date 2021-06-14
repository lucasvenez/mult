from pipeline import SelectMarker

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

from imblearn.over_sampling import SMOTE

from optimization import LightGBMOptimizer

from model import GeneticClustering, GeneticProfiling, DenoisingAutoencoder

from util import to_data_frame

import tensorflow as tf
import pandas as pd
import numpy as np
import lightgbm
import os


class MuLT(SelectMarker):
    """
    """

    def __init__(self,
                 experiment_number=0,
                 number_of_experiments=1,
                 n_gene_limit=None,
                 output_path='./output',
                 random_state=None,
                 verbose=None,
                 export_metadata=True):
        """
        """

        self.experiment_number = experiment_number
        self.number_of_experiments = number_of_experiments
        self.output_path = output_path
        self.random_state = random_state

        self.dae_model_name = 'data_augmentation_adadelta_{0:03d}'.format(self.experiment_number)

        if n_gene_limit is not None:
            self.dae_model_name += '_{0:03d}_genes'.format(n_gene_limit)

        self.dae_output_path = '{0}/dae/'.format(self.output_path)
        self.dae_experiment_path = '{0}/{1}/graph/'.format(self.dae_output_path, self.dae_model_name)
        self.dae_model_path = '{0}/{1}/graph/{1}'.format(self.dae_output_path, self.dae_model_name)

        self.embed_path = os.path.join(self.output_path, 'embed', 'model_{0:03d}'.format(self.experiment_number))

        self.n_gene_limit = n_gene_limit

        self.verbose = verbose

        # scalers
        self.min_max_embed = None
        self.genes_min_max_scaler = None
        self.clinical_min_max_scaler = None
        self.dda_minmax_scaler = None
        self.raw_genes_min_max_scaler = None

        # selected markers
        self.selected_clinical = None
        self.selected_genes = None

        # models
        self.embed = None
        self.genetic_profiling_model = None
        self.genetic_clustering_model = None

        self.lgb_models = []

        # limits
        self.lgb_mins = []
        self.lgb_maxs = []

        # performance metrics
        self.predictor_train_losses = []
        self.predictor_train_aucs = []

        self.predictor_valid_losses = []
        self.predictor_valid_aucs = []

        # params
        self.lgb_optimized_params = None

        # data augmentation
        self.minor_class_augmentation = None

        # export hyperparameters and selected genes?
        self.export_metadata = export_metadata

        # optimizer
        self.predictor_optimizer = None

        # creating required paths
        for subdir in ['selected_markers', 'dae']:
            path = os.path.join(self.output_path, subdir)
            if not os.path.exists(path):
                os.makedirs(path)

    def __reset__(self):

        # scalers
        self.min_max_embed = None
        self.genes_min_max_scaler = None
        self.clinical_min_max_scaler = None
        self.dda_minmax_scaler = None
        self.raw_genes_min_max_scaler = None

        # selected markers
        self.selected_clinical = None
        self.selected_genes = None

        # models
        self.embed = None
        self.genetic_profiling_model = None
        self.genetic_clustering_model = None

        self.lgb_models = []

        # limits
        self.lgb_mins = []
        self.lgb_maxs = []

        # performance metrics
        self.predictor_train_losses = []
        self.predictor_train_aucs = []

        self.predictor_valid_losses = []
        self.predictor_valid_aucs = []

        # params
        self.lgb_optimized_params = None

        # optimizer
        self.predictor_optimizer = None

    def fit_genetic_profiling(self, markers, early_stopping_rounds=50):
        """
        """

        self.genetic_profiling_model = GeneticProfiling(
            random_state=self.random_state,
            early_stopping_rounds=early_stopping_rounds)

        self.genetic_profiling_model.fit(markers)

    def predict_genetic_profiling(self, markers):
        """
        """

        assert self.genetic_profiling_model is not None

        return to_data_frame(self.genetic_profiling_model.transform(markers),
                             prefix='PV', index=markers.index)

    def fit_gene_clustering(self, markers, early_stopping_rounds=50):
        """
        """

        self.genetic_clustering_model = GeneticClustering(
            random_state=self.random_state,
            early_stopping_rounds=early_stopping_rounds,
            verbose=self.verbose)

        self.genetic_clustering_model.fit(markers)

    def predict_gene_clustering(self, markers):
        """
        """

        assert self.genetic_clustering_model is not None

        return to_data_frame(self.genetic_clustering_model.transform(markers),
                             prefix='GC', index=markers.index)

    def fit_embed(self, treatments, outcome, output_dim=5):

        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_random_seed(self.random_state)

        tf.keras.backend.clear_session()

        self.embed = tf.keras.Sequential()
        self.embed.add(tf.keras.layers.Embedding(5, output_dim=output_dim, input_length=1))
        self.embed.add(tf.keras.layers.Dense(20, activation='tanh'))
        self.embed.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.embed.compile(loss='binary_crossentropy', optimizer='adam')

        self.embed.fit(treatments, outcome, epochs=20, batch_size=100, verbose=0)

        for _ in range(2):
            self.embed.pop()

        self.embed.save(self.embed_path)

        del self.embed

        tf.keras.backend.clear_session()

    def predict_embed(self, treatments):

        tf.keras.backend.clear_session()

        self.embed = tf.keras.models.load_model(self.embed_path)

        result = self.embed.predict(treatments)[:, 0, :]

        result = pd.DataFrame({'E{}'.format(h): l for h, l in enumerate(result.T)},
                              index=treatments.index)

        del self.embed

        tf.keras.backend.clear_session()

        return result

    def fit_dae(self,
                markers,
                keep_probability=.7,
                decay_rate=0.1,
                learning_rate=1e-4,
                steps=50000,
                early_stopping_rounds=1000):
        """
        """

        dae = DenoisingAutoencoder(model_name=self.dae_model_name,
                                   summaries_dir=self.dae_output_path,
                                   random_state=self.random_state,
                                   verbose=1)

        dae.build(
            n_inputs=markers.shape[1],
            encoder_units=(int(markers.shape[1] * .5),
                           int(markers.shape[1] * .4),
                           int(markers.shape[1] * .3)),
            decoder_units=(int(markers.shape[1] * .4),
                           int(markers.shape[1] * .5)),
            encoder_activation_function='tanh',
            decoder_activation_function='sigmoid',
            l2_scale=0.01)

        dae.fit(x=markers.values,
                keep_probability=keep_probability,
                steps=steps,
                optimizer='adadelta',
                loss='mse',
                learning_rate=learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                decay_rate=decay_rate)

        dae.close()

        del dae

        for file in os.listdir(self.dae_experiment_path):
            if 'tmp' in file:
                try:
                    os.remove(os.path.join(self.dae_model_path, file))
                except:
                    pass

    def predict_dae(self, markers):
        """
        """

        assert os.path.exists(self.dae_model_path + ".index")

        dae = DenoisingAutoencoder(
            model_name=self.dae_model_name,
            summaries_dir=self.dae_output_path,
            verbose=1,
            random_state=self.random_state)

        dae.load(self.dae_model_path)

        inference = dae.encode(markers.values)

        dae.close()

        del dae

        return pd.DataFrame(inference,
                            index=markers.index,
                            columns=['DDA{:04d}'.format(col) for col in range(inference.shape[1])])

    def fit(self,
            genes,
            outcome,
            clinical=None,
            treatments=None,

            optimization_n_call=50,
            optimization_n_folds=2,
            optimization_early_stopping_rounds=1,

            clinical_marker_selection_threshold=.05,
            gene_selection_threshold=.05,

            dae_early_stopping_rounds=1000,
            dae_decay_rate=0.1,
            dae_learning_rate=1e-4,
            dae_steps=50000,
            dae_keep_probability=.75,
            
            use_predictor=True,

            lgb_fixed_parameters=None,
            lgb_early_stopping_rounds=10,

            predictor_n_folds=5,
            minor_class_augmentation=False):
        """
        """

        self.__reset__()

        self.minor_class_augmentation = minor_class_augmentation

        x = None

        ############################################################################################
        # Select gene expressions
        ############################################################################################

        if clinical is not None:
            self.selected_clinical = self.select_markers(
                clinical, outcome, threshold=clinical_marker_selection_threshold, random_state=self.random_state)

        self.selected_genes = self.select_markers(
            genes, outcome, threshold=gene_selection_threshold, random_state=self.random_state)

        # if self.n_gene_limit is None:
        #    self.n_gene_limit = self.select_k_top_markers(self.selected_genes[2])

        if self.n_gene_limit is not None:
            if 4 <= self.n_gene_limit < len(self.selected_genes[0]):
                self.selected_genes = (self.selected_genes[0][:self.n_gene_limit],
                                       self.selected_genes[1][:self.n_gene_limit],
                                       self.selected_genes[2][:self.n_gene_limit])

        if self.export_metadata:

            if clinical is not None:
                pd.DataFrame({'clinical_marker': self.selected_clinical[0],
                              'pvalue': self.selected_clinical[1],
                              'entropy': self.selected_clinical[2]}).to_csv(
                    os.path.join(
                        self.output_path, 'selected_markers',
                        'clinical_{0:03}_{1:03}.csv'.format(
                            self.experiment_number, self.number_of_experiments)),
                    index=False)

            pd.DataFrame({'gene': self.selected_genes[0],
                          'pvalue': self.selected_genes[1],
                          'entropy': self.selected_genes[2]}).to_csv(
                os.path.join(
                    self.output_path, 'selected_markers',
                    'genes_{0:03}_{1:03}.csv'.format(
                        self.experiment_number, self.number_of_experiments)),
                index=False)

        if clinical is not None:
            if len(self.selected_clinical[0]) > 0:
                x = clinical.loc[:, self.selected_clinical[0]]

        if treatments is not None:
            if treatments.shape[1] > 0:
                x = treatments if x is None else x.join(treatments)

        assert len(self.selected_genes[0]) >= 4, \
            'At least 4 genes are required for MuLT approach. You can increase the threshold.'

        genes = genes.loc[:, self.selected_genes[0]]

        #
        self.raw_genes_min_max_scaler = MinMaxScaler()

        genes_norm = self.raw_genes_min_max_scaler.fit_transform(genes)

        genes_norm = pd.DataFrame(genes_norm, columns=genes.columns, index=genes.index)

        ############################################################################################
        # Normalizing Gene Expression Data
        ############################################################################################

        self.genes_min_max_scaler = MinMaxScaler()

        genes = pd.DataFrame(self.genes_min_max_scaler.fit_transform(np.log1p(genes)),
                             index=genes.index, columns=genes.columns)

        ############################################################################################
        # Genetic Profiling
        ############################################################################################

        self.fit_genetic_profiling(genes_norm)
        profiling = self.predict_genetic_profiling(genes_norm)

        x = pd.concat([x, profiling], axis=1) if x is not None else profiling

        ############################################################################################
        # Gene Clustering
        ############################################################################################

        self.fit_gene_clustering(genes_norm)

        gene_clusters = self.predict_gene_clustering(genes_norm)

        x = pd.concat([x, gene_clusters], axis=1)

        ############################################################################################
        # Denoising Autoencoder
        ############################################################################################

        self.fit_dae(markers=genes,
                     keep_probability=dae_keep_probability,
                     decay_rate=dae_decay_rate,
                     learning_rate=dae_learning_rate,
                     steps=dae_steps,
                     early_stopping_rounds=dae_early_stopping_rounds)

        if use_predictor:
        
            dda = self.predict_dae(genes)

            ############################################################################################
            # Joining all features
            ############################################################################################

            x = x.join(genes_norm).join(dda, how='inner').fillna(0)

            x, y = x.values, outcome.values

            if minor_class_augmentation:
                smote = SMOTE(sampling_strategy='minority', random_state=self.random_state, n_jobs=-1)
                x, y = smote.fit_resample(x, y)
                del smote

            ############################################################################################
            # LightGBM Hyper parameter Optimization
            ############################################################################################

            if lgb_fixed_parameters is None:
                lgb_fixed_parameters = dict()

            self.predictor_optimizer = LightGBMOptimizer(
                n_calls=optimization_n_call,
                n_folds=optimization_n_folds,
                fixed_parameters=lgb_fixed_parameters,
                early_stopping_rounds=optimization_early_stopping_rounds,
                random_state=self.random_state)

            lgb_params = self.predictor_optimizer.optimize(x, y)

            self.lgb_optimized_params = lgb_params

            lgb_params = {**lgb_params, **lgb_fixed_parameters}

            ############################################################################################
            # Training
            ############################################################################################

            if predictor_n_folds > 1:
                kkfold = StratifiedKFold(predictor_n_folds, random_state=self.random_state)
                splits = kkfold.split(x, y)

            else:
                splits = [(list(range(0, x.shape[0])), None)]

            for iii, (t_index, v_index) in enumerate(splits):

                x_train, y_train = x[t_index, :], y[t_index]

                if v_index is not None:
                    x_valid, y_valid = x[v_index, :], y[v_index]

                ###############################################################################
                # Light GBM
                ###############################################################################

                lgb = lightgbm.LGBMModel(**lgb_params)

                lgb.fit(
                    X=x_train, y=y_train,
                    eval_set=[(x_valid, y_valid)] if v_index is not None else None,
                    early_stopping_rounds=lgb_early_stopping_rounds if v_index is not None else None,
                    verbose=self.verbose is not None and self.verbose > 0)

                y_train_hat_lgb = lgb.predict(x_train)

                self.lgb_models.append(lgb)

                if v_index is not None:
                    y_valid_hat_lgb = lgb.predict(x_valid)
                    self.lgb_mins.append(min(np.min(y_train_hat_lgb), np.min(y_valid_hat_lgb)))
                    self.lgb_maxs.append(max(np.max(y_train_hat_lgb), np.max(y_valid_hat_lgb)))

                else:
                    self.lgb_mins.append(min(y_train_hat_lgb))
                    self.lgb_maxs.append(max(y_train_hat_lgb))

                ###############################################################################
                # Performance metrics
                ###############################################################################

                y_train_hat = (y_train_hat_lgb - self.lgb_mins[-1]) / (self.lgb_maxs[-1] - self.lgb_mins[-1])

                if v_index is not None:
                    y_valid_hat = (y_valid_hat_lgb - self.lgb_mins[-1]) / (self.lgb_maxs[-1] - self.lgb_mins[-1])

                self.predictor_train_losses.append(log_loss(y_train, y_train_hat))
                self.predictor_train_aucs.append(roc_auc_score(y_train, y_train_hat))

                if v_index is not None:
                    self.predictor_valid_losses.append(log_loss(y_valid, y_valid_hat))
                    self.predictor_valid_aucs.append(roc_auc_score(y_valid, y_valid_hat))

            if self.verbose:

                print('Train mean log loss: {0:03}'.format(np.mean(self.predictor_train_losses)))
                print('Train mean AUC: {0:03}'.format(np.mean(self.predictor_train_aucs)))

                if v_index is not None:
                    print('Valid mean log loss: {0:03}'.format(np.mean(self.predictor_valid_losses)))
                    print('Valid mean AUC: {0:03}'.format(np.mean(self.predictor_valid_aucs)))

    def transform(self, genes, clinical=None, treatments=None, add_gc=True, add_pc=True, add_dda=True,
                  as_data_frame=False):
        """

        :param genes:
        :param clinical:
        :param treatments:
        :param add_gc:
        :param add_pc:
        :param add_dda:
        :param as_data_frame:
        :return:
        """

        assert len(self.lgb_models) > 0

        X = None

        ############################################################################################
        # Feature Selection
        ############################################################################################

        if clinical is not None:
            X = clinical[self.selected_clinical[0]]

        if treatments is not None:
            X = X.join(treatments) if X is not None else treatments

        genes = genes[self.selected_genes[0]]

        #
        genes_norm = self.raw_genes_min_max_scaler.transform(genes)

        genes_norm = pd.DataFrame(genes_norm, columns=genes.columns, index=genes.index)

        ############################################################################################
        # Normalizing Gene Expression Data
        ############################################################################################

        genes = pd.DataFrame(self.genes_min_max_scaler.transform(np.log1p(genes)),
                             index=genes.index, columns=genes.columns)

        ############################################################################################
        # Patient Clustering
        ############################################################################################

        if add_pc is True:
            profiling = self.predict_genetic_profiling(genes_norm)
            X = pd.concat([X, profiling], axis=1) if X is not None else profiling

        ############################################################################################
        # Gene Clustering
        ############################################################################################

        if add_gc is True:
            gene_clusters = self.predict_gene_clustering(genes_norm)
            X = pd.concat([X, gene_clusters], axis=1)

        ############################################################################################
        # Denoising Autoencoder
        ############################################################################################

        if add_dda is True:
            dda = self.predict_dae(genes)

        ############################################################################################
        # Joining all features
        ############################################################################################

        X = X.join(genes_norm, how='inner')

        if add_dda is True:
            X = X.join(dda, how='inner')

        X = X.fillna(0)

        return X.values if as_data_frame is False else X

    def predict(self, genes, clinical=None, treatments=None):
        """
        """

        X = self.transform(genes, clinical, treatments)

        ############################################################################################
        # Predicting
        ############################################################################################

        values = None

        for i, lgb in enumerate(self.lgb_models):

            ############################################################################################
            # LightGBM
            ############################################################################################

            lgb_y_hat = lgb.predict(X, num_iteration=lgb.best_iteration_)

            # lgb_y_hat = (lgb_y_hat - self.lgb_mins[i]) / (self.lgb_maxs[i] - self.lgb_mins[i])

            # lgb_y_hat = np.maximum(0., np.minimum(1., lgb_y_hat))

            ############################################################################################
            # Final score
            ############################################################################################

            values = lgb_y_hat if values is None else values + lgb_y_hat

        return values / len(self.lgb_models)
