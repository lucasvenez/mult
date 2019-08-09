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

