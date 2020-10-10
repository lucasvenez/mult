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

library(ggplot2)

df <- read.csv('data/output/pca_with_resp.csv')


p <- ggplot(df) + geom_point(aes(PCA1, PCA2, colour=as.factor(Greater18Months.Days.to.Progression))) + 
  scale_colour_discrete(name = "Risk", labels=c('Hight', 'Low'))

ggsave('images/pca_all_vars.pdf', p, units='cm', width=16, height=10)

df <- read.csv('data/output/pca_with_resp_and_reduction.csv')

p <- ggplot(df) + geom_point(aes(PCA1, PCA2, colour=as.factor(Greater18Months.Days.to.Progression))) + 
  scale_colour_discrete(name = "Risk", labels=c('Hight', 'Low'))

ggsave('images/pca_reduced_gens.pdf', p, units='cm', width=16, height=10)
