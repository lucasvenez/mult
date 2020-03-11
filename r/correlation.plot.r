##

clinical_mic <- read.csv('output/correlation_clinical_x_m2dp.csv', sep=',', header=T)
clinical_mic$resp <- 'M2DP (MIC)'

clinical_pva <- read.csv('output/correlation_clinical_x_brft.csv', sep=',', header=T)
clinical_pva$metric = 1 - clinical_pva$metric 
clinical_pva$resp <- 'BRFT_G3 (1 - pvalue)'

clinical <- rbind(clinical_mic, clinical_pva)
clinical[is.na(clinical)] = 0
clinical$feature = sapply(clinical$feature, function(x) toupper(substr(x, 0, 25)))

##


fish_mic <- read.csv('output/correlation_fish_x_m2dp.csv', sep=',', header=T)
fish_mic$resp <- 'M2DP (MIC)'

fish_pva <- read.csv('output/correlation_fish_x_brft.csv', sep=',', header=T)
fish_pva$metric = 1 - fish_pva$metric 
fish_pva$resp <- 'BRFT_G3 (1 - pvalue)'

fish <- rbind(fish_mic, fish_pva)
fish[is.na(fish)] = 0
fish$feature = sapply(fish$feature, function(x) toupper(substr(x, 0, 25)))

##

genexp_mic <- read.csv('output/correlation_geneexp_x_m2dp.csv', sep=',', header=T)
genexp_mic$resp <- 'M2DP (MIC)'

genexp_pva <- read.csv('output/correlation_geneexp_x_brft.csv', sep=',', header=T)
genexp_pva$metric = 1 - genexp_pva$metric 
genexp_pva$resp <- 'BRFT_G3 (1 - pvalue)'

genexp <- rbind(genexp_mic, genexp_pva)

##
require(ggplot2)

p <- ggplot(clinical) + 
  geom_bar(stat='identity', aes(feature, metric), colour='gray28', alpha=.6) + 
  facet_grid(. ~ resp) + 
  theme(text=element_text(size=11, family='serif')) +
  xlab('Feature') + ylab('Correlation Metric Value') +
  coord_flip()

ggsave('images/correlation_clinical.png', p, width=16, height=20, units='cm')

p <- ggplot(fish) + 
  geom_bar(stat='identity', aes(feature, metric), colour='gray28', alpha=.6) + 
  facet_grid(. ~ resp) + 
  theme(text=element_text(size=11, family='serif')) +
  xlab('Feature') + ylab('Correlation Metric Value') +
  coord_flip()

ggsave('images/correlation_fish.png', p, width=16, height=14, units='cm')

##

p <- ggplot(genexp) + 
  geom_histogram(aes(metric), colour='gray28', alpha=.6) + 
  facet_grid(. ~ resp, scales = "free_x") + 
  theme(text=element_text(size=15, family='serif')) +
  xlab('Correlation Metric Value') + ylab('Count')

ggsave('images/correlation_genexp.png', p, width=16, height=8, units='cm')
