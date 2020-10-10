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

df <- read.csv('../output/fill/accuracy_fish_aucs.csv', sep=',')
df$category <- c(rep('Copy Number Deletions', 3), c('Copy Number Gains'), rep('Hyperdiploid Copy Number Gains', 9), rep('Translocations', 8), c('Flag'))

fish.levels <- c("Copy Number Deletions", "Copy Number Gains", "Hyperdiploid Copy Number Gains", "Translocations", "Flag")
df$category <- factor(df$category, levels = fish.levels)
#df$category <- fish.levels
df <- df[order(df$category),]
# df$category <- factor(df$category, levels = fish.levels)

df$var <- factor(df$var, levels = df$var[rev(order(df$category))])

graph <- ggplot(df, 
                aes(x = var, 
                    y = auc, 
                    colour = category,
                    fill = category)) + 
  geom_bar(stat = "identity", alpha = .6, size = 0.3) +
  ylim(0,1) + 
  scale_fill_hue(name    = "FISH Legend", 
                 labels  = fish.levels) +
  scale_colour_hue(guide = "none") +
  scale_y_continuous(labels = scales::percent) +
  xlab(NULL) + ylab("AUC") +
  theme(legend.background  = element_rect(colour = "black", size = .2),
        panel.grid.major.y = element_line(size = .15),
        text               = element_text(size = 7),
        legend.key.size    = unit(8, "pt"),
        legend.position    = 'right',
        axis.text.x        = element_text(size = 6, angle = 45),
        axis.title.x       = element_text(vjust = 1)) + 
  coord_flip()
