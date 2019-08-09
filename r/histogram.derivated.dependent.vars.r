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

p

ggsave('images/derivated_dep_vars.pdf', p, units='cm', width=16, height=8)
