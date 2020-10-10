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
library(scales)

mic <- read.csv('data/output/correlation_delgain_x_response_vars_mic.csv', stringsAsFactors = FALSE)
mic[is.na(mic)] <- 0.0
mic['METHOD'] <- 'Maximal Information Coefficient'
colnames(mic) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

dis <- read.csv('data/output/correlation_delgain_x_response_vars_distcorr.csv', stringsAsFactors = FALSE)
dis[is.na(dis)] <- 0.0
dis['METHOD'] <- 'Distance Correlation'
colnames(dis) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

pea <- read.csv('data/output/correlation_delgain_x_response_vars_pearson.csv', stringsAsFactors = FALSE)
pea[is.na(pea)] <- 0.0
pea['METHOD'] <- 'Pearson Correlation Coefficient'
colnames(pea) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

df <- rbind(mic, dis, pea)

#
#
#
df[df == "Maximal Information Coefficient"] <- "MIC"
df[df == "Distance Correlation"] <- "DCR"
df[df == "Pearson Correlation Coefficient"] <- "PCC"

#
#
#
df[df == "Days-to-Progression"] <- "M2DP"
df[df == "Greater18Months-Days-to-Progression"] <- "M2DP_GT18M"
df[df == "Group1-From-Best-Response-FirstLine"] <- "BRFT_G1"
df[df == "Group2-From-Best-Response-FirstLine"] <- "BRFT_G2"
df[df == "Group3-From-Best-Response-FirstLine"] <- "BRFT_G3"
df[df == "Group4-From-Best-Response-FirstLine"] <- "BRFT_G4"
df[df == "Group5-From-Best-Response-FirstLine"] <- "BRFT_G5"
df[df == "Best-Response-FirstLine-ID"] <- "BRFT"

df[df == 'First-line-Therapy'] <- 'Therapy'

p <- 
  ggplot(df) + 
  geom_bar(stat='identity', aes(x=IndependentVariable, y=Value)) + 
  coord_flip() +
  facet_grid(DependentVariable ~ Method,  scales='free_x') +
  ylab('Correlation Metric') + xlab('Independent Variable') +
  theme(text=element_text(size=11, family='serif'), 
        axis.text.x=element_text(angle = 45, hjust = 1))

ggsave('images/histogram_delgain_correlation_all_dep_vars.pdf', p, units='cm', width=16, height=24)
