library(ggplot2)
library(scales)

mic <- read.csv('data/output/correlation_clinical_x_response_vars_mic.csv', stringsAsFactors = FALSE)
mic[is.na(mic)] <- 0.0
mic['METHOD'] <- 'Maximal Information Coefficient'
colnames(mic) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

dis <- read.csv('data/output/correlation_clinical_x_response_vars_distcorr.csv', stringsAsFactors = FALSE)
dis[is.na(dis)] <- 0.0
dis['METHOD'] <- 'Distance Correlation'
colnames(dis) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

pea <- read.csv('data/output/correlation_clinical_x_response_vars_pearson.csv', stringsAsFactors = FALSE)
pea[is.na(pea)] <- 0.0
pea['METHOD'] <- 'Pearson Correlation Coefficient'
colnames(pea) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

df <- rbind(mic, dis, pea)

df[df == 'First-line-Therapy'] <- 'Therapy'

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

p <- 
  ggplot(df) + 
  geom_bar(stat='identity', aes(x=IndependentVariable, y=Value)) + 
  coord_flip() +
  facet_grid(DependentVariable ~ Method,  scales='free_x') +
  ylab('Correlation Metric') + xlab('Independent Variable') +
  theme(text=element_text(size=11, family='serif'), 
        axis.text.x=element_text(angle = 45, hjust = 1))

ggsave('images/histogram_clinical_correlation_all_dep_vars.pdf', p, units='cm', width=16, height=24)
