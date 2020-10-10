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

df <- read.csv('data/output/derived_dependent_variables.csv', sep=',', stringsAsFactors = FALSE)

df[df == 0] <- 'High Risk'

df[df == 1] <- 'Low Risk'

colnames(df) <- c('M2DP_GT18M', 'BRFT_G1', 'BRFT_G2', 'BRFT_G3', 'BRFT_G4', 'BRFT_G5')

df <- stack(df)

p <- 
  ggplot(df) + 
  geom_bar(aes(values), alpha=.75, colour='black') + 
  facet_wrap(ind ~ .) + 
  xlab('Risk Group') + 
  ylab('Count') +
  theme(text=element_text(size=11, family='serif'))

ggsave('images/derivated_dep_vars.pdf', p, units='cm', width=16, height=8)
