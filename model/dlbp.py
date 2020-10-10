# Copyright 2020 The MuLT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from model import PreProcessing, FeatureSelector


class DeepLearningBasedProcess(object):
    """
    author: Lucas Venezian Povoa
    """

    def __init__(self):

        self.clinical_pre_processing = PreProcessing()

        self.mutation_pre_processing = PreProcessing()

        self.gene_expression_pre_processing = PreProcessing()

        self.feature_selector = FeatureSelector()

        self.gene_profiling = GeneExpressionProfiling()

        self.gene_clustering = GeneExpressionClustering()

        self.autoencoer = Autoencoder()

        self.classifier = Deconvense()

    def fit(self, clinical_vars, mutation_vars, gene_expression_vars, y):

        #
        #
        #
        clinical_vars = self.clinical_pre_processing.fit_transform(clinical_vars)

        mutation_vars = self.mutation_pre_processing.fit_transform(mutation_vars)

        gene_expression_vars = self.gene_expression_pre_processing.fit_transform(gene_expression_vars)

        #
        #
        #
        gene_expression_vars = self.feature_selector.fit_transform(gene_expression_vars, y)

        #
        #
        #
        all_vars = clinical_vars.join(mutation_vars, how='outer').join(gene_expression_vars, how='inner').fillna(0)

        #
        #
        #
        gene_clustering = self.gene_clustering.fit_transform(gene_expression_vars)

        gene_profiling = self.gene_profiling.fit_transform(gene_expression_vars)

        #
        #
        #
        noise = self.autoencoer.fit_transform(all_vars)

        #
        #
        #
        all_vars = all_vars.join(gene_clustering, how='inner').join(gene_profiling, how='inner').join(noise, how='inner')

        #
        #
        #
        self.classifier.fit(all_vars, y)

    def predict(self, clinical_vars, mutation_vars, gene_expression_vars, y):

        #
        #
        #
        clinical_vars = self.clinical_pre_processing.transform(clinical_vars)

        mutation_vars = self.mutation_pre_processing.transform(mutation_vars)

        gene_expression_vars = self.gene_expression_pre_processing.transform(gene_expression_vars)

        #
        #
        #
        gene_expression_vars = self.feature_selector.transform(gene_expression_vars)

        #
        #
        #
        all_vars = clinical_vars.join(mutation_vars, how='outer').join(gene_expression_vars, how='inner')

        #
        #
        #
        gene_clustering = self.gene_clustering.fit_transform(gene_expression_vars)

        gene_profiling = self.gene_profiling.fit_transform(gene_expression_vars)

        #
        #
        #
        noise = self.autoencoer.fit_transform(all_vars)

        #
        #
        #
        all_vars = all_vars.join(gene_clustering, how='inner')\
            .join(gene_profiling, how='inner').join(noise, how='inner')

        #
        #
        #
        return self.classifier.predict(all_vars, y)

    def fit_predict(self, clinical_vars, mutation_vars, gene_expression_vars, y):

        self.fit(clinical_vars, mutation_vars, gene_expression_vars, y)

        return self.predict(clinical_vars, mutation_vars, gene_expression_vars)
