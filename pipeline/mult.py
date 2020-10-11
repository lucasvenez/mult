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
                 verbose=None):
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

    def __reset__(self):

        # scalers
        self.min_max_embed = None
        self.genes_min_max_scaler = None
        self.clinical_min_max_scaler = None
        self.dda_minmax_scaler = None

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

    def fit_embed(self, treatments, outcome):

        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_random_seed(self.random_state)

        tf.keras.backend.clear_session()

        self.embed = tf.keras.Sequential()
        self.embed.add(tf.keras.layers.Embedding(5, 3, input_length=1))
        self.embed.add(tf.keras.layers.Dense(20, activation='relu'))
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
            encoder_activation_function='relu',
            decoder_activation_function='relu',
            l2_scale=0.01)

        dae.fit(x=markers.values,
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
                os.remove(os.path.join(self.dae_model_path, file))

        dae = DenoisingAutoencoder(model_name=self.dae_model_name,
                                   summaries_dir=self.dae_output_path,
                                   random_state=self.random_state,
                                   verbose=1)

        dae.load(self.dae_model_path)

        self.dda_minmax_scaler = MinMaxScaler()

        self.dda_minmax_scaler.fit(dae.predict(markers.values))

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

        inference = dae.predict(markers.values)

        dae.close()

        del dae

        return pd.DataFrame(self.dda_minmax_scaler.transform(inference),
                            index=markers.index,
                            columns=[str(col) + '_DDA' for col in markers.columns])

    def fit(self,

            clinical,
            genes,
            treatments,
            outcome,

            optimization_n_call=50,
            optimization_n_folds=2,
            optimization_early_stopping_rounds=1,

            clinical_marker_selection_threshold=.05,
            gene_selection_threshold=.05,

            dae_early_stopping_rounds=1000,
            dae_decay_rate=0.1,
            dae_learning_rate=1e-4,
            dae_steps=50000,

            lgb_fixed_parameters=dict(),
            lgb_early_stopping_rounds=10,

            predictor_n_folds=5):
        """
        """

        self.__reset__()

        ############################################################################################
        # Select gene expressions
        ############################################################################################

        self.selected_clinical = self.select_markers(
            clinical, outcome, threshold=clinical_marker_selection_threshold)

        self.selected_genes = self.select_markers(
            genes, outcome, threshold=gene_selection_threshold)

        if self.n_gene_limit is not None:
            if 4 <= self.n_gene_limit < len(self.selected_genes[0]):
                self.selected_genes = (self.selected_genes[0][:self.n_gene_limit],
                                       self.selected_genes[1][:self.n_gene_limit],
                                       self.selected_genes[2][:self.n_gene_limit])

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

        clinical = clinical.loc[:, self.selected_clinical[0]]
        genes = genes.loc[:, self.selected_genes[0]]

        ############################################################################################
        # Normalizing Gene Expression Data
        ############################################################################################

        self.genes_min_max_scaler = MinMaxScaler()

        genes = pd.DataFrame(self.genes_min_max_scaler.fit_transform(genes),
                             index=genes.index, columns=genes.columns)

        ############################################################################################
        # Embedding Treatments
        ############################################################################################

        self.fit_embed(treatments, outcome)
        clinical = clinical.join(self.predict_embed(treatments))

        ############################################################################################
        # Genetic Profiling
        ############################################################################################

        self.fit_genetic_profiling(genes)
        profiling = self.predict_genetic_profiling(genes)

        clinical = pd.concat([clinical, profiling], axis=1)

        ############################################################################################
        # Gene Clustering
        ############################################################################################

        self.fit_gene_clustering(genes)
        gene_clusters = self.predict_gene_clustering(genes)

        clinical = pd.concat([clinical, gene_clusters], axis=1)

        ############################################################################################
        # Normalizing Clinical Data
        ############################################################################################

        self.clinical_min_max_scaler = MinMaxScaler()

        clinical = pd.DataFrame(self.clinical_min_max_scaler.fit_transform(clinical),
                                index=clinical.index, columns=clinical.columns)

        clinical = clinical.fillna(0)

        ############################################################################################
        # Denoising Autoencoder
        ############################################################################################

        self.fit_dae(markers=genes,
                     decay_rate=dae_decay_rate,
                     learning_rate=dae_learning_rate,
                     steps=dae_steps,
                     early_stopping_rounds=dae_early_stopping_rounds)

        dda = self.predict_dae(genes)

        ############################################################################################
        # Joining all features
        ############################################################################################

        join = clinical.join(genes, how='inner').join(dda, how='inner')

        x = join.values
        y = outcome.values

        smote = SMOTE(sampling_strategy='all', random_state=self.random_state, n_jobs=-1)

        x, y = smote.fit_resample(x, y)

        del smote

        ############################################################################################
        # LightGBM Hyperparameter Optimization
        ############################################################################################

        lgb_params = LightGBMOptimizer(
            n_calls=optimization_n_call,
            n_folds=optimization_n_folds,
            fixed_parameters=lgb_fixed_parameters,
            early_stopping_rounds=optimization_early_stopping_rounds,
            random_state=self.random_state).optimize(x, y)

        self.lgb_optimized_params = lgb_params

        lgb_params = {**lgb_params, **lgb_fixed_parameters}

        ############################################################################################
        # Training
        ############################################################################################

        kkfold = StratifiedKFold(predictor_n_folds, random_state=self.random_state)

        for iii, (t_index, v_index) in enumerate(kkfold.split(x, y)):

            x_train, y_train = x[t_index, :], y[t_index]
            x_valid, y_valid = x[v_index, :], y[v_index]

            ###############################################################################
            # Light GBM
            ###############################################################################

            lgb = lightgbm.LGBMModel(**lgb_params)

            lgb.fit(
                X=x_train, y=y_train,
                eval_set=[(x_valid, y_valid)],
                early_stopping_rounds=lgb_early_stopping_rounds,
                verbose=self.verbose is not None and self.verbose > 0)

            y_train_hat_lgb = lgb.predict(x_train)
            y_valid_hat_lgb = lgb.predict(x_valid)

            self.lgb_models.append(lgb)

            self.lgb_mins.append(min(np.min(y_train_hat_lgb), np.min(y_valid_hat_lgb)))
            self.lgb_maxs.append(max(np.max(y_train_hat_lgb), np.max(y_valid_hat_lgb)))

            ###############################################################################
            # Performance metrics
            ###############################################################################

            # y_train_hat = (y_train_hat_lgb - self.lgb_mins[-1]) / (self.lgb_maxs[-1] - self.lgb_mins[-1])
            # y_valid_hat = (y_valid_hat_lgb - self.lgb_mins[-1]) / (self.lgb_maxs[-1] - self.lgb_mins[-1])
            y_train_hat = y_train_hat_lgb
            y_valid_hat = y_valid_hat_lgb

            self.predictor_train_losses.append(log_loss(y_train, y_train_hat))
            self.predictor_train_aucs.append(roc_auc_score(y_train, y_train_hat))

            self.predictor_valid_losses.append(log_loss(y_valid, y_valid_hat))
            self.predictor_valid_aucs.append(roc_auc_score(y_valid, y_valid_hat))

        print('TRAIN mean log loss: {0:03}'.format(np.mean(self.predictor_train_losses)))
        print('TRAIN mean AUC: {0:03}'.format(np.mean(self.predictor_train_aucs)))

        print('VALID mean log loss: {0:03}'.format(np.mean(self.predictor_valid_losses)))
        print('VALID mean AUC: {0:03}'.format(np.mean(self.predictor_valid_aucs)))

    def predict(self, clinical, genes, treatments):
        """
        """

        assert len(self.lgb_models) > 0

        ############################################################################################
        # Feature Selection
        ############################################################################################

        clinical = clinical[self.selected_clinical[0]]
        genes = genes[self.selected_genes[0]]

        ############################################################################################
        # Normalizing Gene Expression Data
        ############################################################################################

        genes = pd.DataFrame(self.genes_min_max_scaler.transform(genes),
                             index=genes.index, columns=genes.columns)

        ############################################################################################
        # Embedding Treatments
        ############################################################################################

        clinical = clinical.join(self.predict_embed(treatments))

        ############################################################################################
        # Genetic Profiling
        ############################################################################################

        profiling = self.predict_genetic_profiling(genes)
        clinical = pd.concat([clinical, profiling], axis=1)

        ############################################################################################
        # Gene Clustering
        ############################################################################################

        gene_clusters = self.predict_gene_clustering(genes)
        clinical = pd.concat([clinical, gene_clusters], axis=1)

        ############################################################################################
        # Normalizing Clinical Data
        ############################################################################################

        clinical = pd.DataFrame(
            self.clinical_min_max_scaler.transform(clinical),
            index=clinical.index, columns=clinical.columns)

        clinical = clinical.fillna(0)

        ############################################################################################
        # Denoising Autoencoder
        ############################################################################################

        dda = self.predict_dae(genes)

        ############################################################################################
        # Joining all features
        ############################################################################################

        x = clinical.join(genes, how='inner').join(dda, how='inner').values

        ############################################################################################
        # Predicting
        ############################################################################################

        values = None

        for i, lgb in enumerate(self.lgb_models):

            ############################################################################################
            # LightGBM
            ############################################################################################

            lgb_y_hat = lgb.predict(x)

            lgb_y_hat = np.maximum(0, np.minimum(1.,
                    (lgb_y_hat - self.lgb_mins[i]) / (self.lgb_maxs[i] - self.lgb_mins[i])))

            ############################################################################################
            # Final score
            ############################################################################################

            values = lgb_y_hat if values is None else values + lgb_y_hat

        return values / len(self.lgb_models)
