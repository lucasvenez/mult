library(ggplot2)

df <- read.csv('data/iss_fish_therapy_response.csv', sep='\t')

df['M2DP_GT18M'] <- df$Days.to.Progression / 30 > 18

df$M2DP_GT18M[df$M2DP_GT18M == 1] = 'Low Risk'
df$M2DP_GT18M[df$M2DP_GT18M == 'FALSE'] = 'High Risk'
df$M2DP_GT18M[is.na(df$M2DP_GT18M)] = 'NA'

p <- ggplot(df) + geom_bar(aes(First.line.Therapy, fill=M2DP_GT18M), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('First line therapy') +
     theme(text=element_text(size=11, family='serif')) 

ggsave('images/indep_var_histogram_first_line_therapy.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_bar(aes(First.line.Therapy.Class, fill=M2DP_GT18M), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('First line therapy class') +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_first_line_therapy_class.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_histogram(aes(Age, fill=M2DP_GT18M), alpha=.6) +
     ylab('Count') + xlab('Age') +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_age.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_bar(aes(ISS, fill=M2DP_GT18M), alpha=.6) +
     ylab('Count') + xlab('ISS') +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_iss.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_bar(aes(DPRT, fill=M2DP_GT18M), alpha=.6) +
     ylab('Count') + xlab('DPRT') +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_dprt.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_bar(aes(Race, fill=M2DP_GT18M), alpha=.6) +
  ylab('Count') + xlab('DPRT') +
  theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_race.pdf', p, units='cm', width=16, height=8)