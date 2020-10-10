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

df <- df[!is.na(df$Days.to.Progression),]

print(nrow(df))

p1 <- ggplot(df) + geom_bar(stat='count', aes(as.factor(ifelse(Days.to.Progression/30 > 18, 1, 0))), alpha=.7, colour='black') +
        xlab('Months to Disease Progression') + ylab('Count') + 
        theme(text = element_text(size=11, family='serif')) + coord_flip()

ggsave('images/histogram_risk_from_days_to_progression.pdf', p1, units='cm', width=16, height=4)
