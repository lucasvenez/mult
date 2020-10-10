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

library('ggplot2')

df <- read.csv('data/iss_fish_therapy_response.csv', sep='\t')

p1 <- ggplot(df) + geom_histogram(aes(Days.to.Progression/30), alpha=.7, colour='black') +
          xlab('Months to Disease Progression') + ylab('Count') + 
          theme(text = element_text(size=11, family='serif'))


p2 <- ggplot(df[!is.na(df$Best.Response.FirstLine),]) + geom_bar(aes(Best.Response.FirstLine), alpha=.7, colour='black', stat='count') +
        xlab('Best Response at First Line Treat.') + ylab('Count') + 
        theme(text = element_text(size=11, family='serif'))


ggsave('images/histogram_days_to_desiase_progression.pdf', p1, units='cm', width=8, height=6)

ggsave('images/histogram_best_response_firstline.pdf', p2, units='cm', width=8, height=6)

