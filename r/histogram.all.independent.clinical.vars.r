library(ggplot2)

df <- read.csv('data/clinical.tsv', sep='\t')

df$response_best_response_first_line[df$response_best_response_first_line == 1] = 'Low Risk'
df$response_best_response_first_line[df$response_best_response_first_line == 0] = 'High Risk'
df[is.na(df)] = 'NA'
df[df == ''] = 'NA'

df$iss[df$iss == 1] = '   Stage I'
df$iss[df$iss == 2] = '   Stage II'
df$iss[df$iss == 3] = '   Stage III'

p <- ggplot(df) + geom_bar(aes(therapy_first_line, fill=response_best_response_first_line), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('First line therapy') +
     scale_fill_grey(name = "BRFL_G2") +
     theme(text=element_text(size=11, family='serif')) 

ggsave('images/indep_var_histogram_first_line_therapy.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_bar(aes(therapy_first_line_class, fill=response_best_response_first_line), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('First line therapy class') +
     scale_fill_grey(name = "BRFL_G2") +
     theme(legend.position = c(-.5, .05), text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_first_line_therapy_class.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_histogram(aes(age, fill=response_best_response_first_line), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('Age') +
     scale_fill_grey(name = "BRFL_G2") +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_age.pdf', p, units='cm', width=16, height=8)

p <- ggplot(df) + geom_bar(aes(iss, fill=response_best_response_first_line), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('ISS') +
     scale_fill_grey(name = "BRFL_G2") +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_iss.pdf', p, units='cm', width=16, height=5)

p <- ggplot(df) + geom_bar(aes(first_line_transplant, fill=response_best_response_first_line), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('First Line Transplant') +
     scale_fill_grey(name = "BRFL_G2") +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_dprt.pdf', p, units='cm', width=16, height=5.5)

p <- ggplot(df) + geom_bar(aes(race, fill=response_best_response_first_line), alpha=.6) + coord_flip() + 
     ylab('Count') + xlab('Race') +
     scale_fill_grey(name = "BRFL_G2") +
     theme(text=element_text(size=11, family='serif'))

ggsave('images/indep_var_histogram_race.pdf', p, units='cm', width=16, height=7)