library(ggplot2)
library(scales)

mic <- read.csv('data/output/correlation_gene_x_response_vars_mic.csv', stringsAsFactors = FALSE)
mic['METHOD'] <- 'Maximal Information Coefficient'
colnames(mic) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

dis <- read.csv('data/output/correlation_gene_x_response_vars_distcorr.csv', stringsAsFactors = FALSE)
dis['METHOD'] <- 'Distance Correlation'
colnames(dis) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

pea <- read.csv('data/output/correlation_gene_x_response_vars_pearson.csv', stringsAsFactors = FALSE)
pea['METHOD'] <- 'Pearson Correlation Coefficient'
colnames(pea) <- c('IndependentVariable', 'DependentVariable', 'Value', 'Method')

df <- rbind(mic, dis, pea)

df[is.na(df)] <- 0.0

#
#
#
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
  geom_histogram(aes(Value, y = (..count..)/sum(..count..)), bins=50) + 
  facet_grid(DependentVariable ~ Method,  scales='free_x') +
  xlab('Correlation Metric') + ylab('Frequency') +
  scale_y_continuous(labels=percent) + 
  theme(text=element_text(size=11, family='serif'))

ggsave('images/histogram_correlation_selected_genes.pdf', p, units='cm', width=16, height=24)


