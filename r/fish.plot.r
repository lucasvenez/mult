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

df <- read.csv('data/clinical.tsv', sep='\t', stringsAsFactors = F)


df <- df[c('response_best_response_first_line', 'X13q14', 'X13q34', 'X17p13', 'X1q21', 'X11p15', 'X15q15', 'X19q13', 'X20q13', 'X21q22', 'X3q21', 'X5q31', 'X7q22', 
     'X9q33', 't_11_14_ccnd1', 't_12_14_ccnd2', 't_14_16_maf', 't_14_20_mafb', 't_4_14_whsc1', 't_6_14_ccnd3', 't_8_14_mafa', 't_8_14_myc')]

require(ggplot2)

resp = df$response_best_response_first_line

df = stack(df, select=-response_best_response_first_line)

df$resp <- resp

df[is.na(df)] <- 'NA'
df[df==''] <- 'NA'
df[df=='Detected'] <- 1
df[df=='Not Detected'] <- 0

df$ind <- gsub("X", "", df$ind)

p <- ggplot(df) + 
  geom_bar(aes(values, fill=resp, colour=resp), alpha=.6) +
  xlab('FISH') + ylab('Count') +
  scale_fill_grey('BRFT_G2', labels = c("High Risk", "Low Risk", "NA")) + 
  scale_colour_grey('BRFT_G2', labels = c("High Risk", "Low Risk", "NA")) +
  theme(legend.position = c(0.81, 0.1), 
        legend.direction = 'horizontal', 
        legend.box.background = element_rect(colour = "black"),
        legend.background = element_blank()) +
  facet_wrap(ind  ~ ., ncol = 8)

ggsave('images/fish_plot2.png', p, width=25, height=9, units='cm')