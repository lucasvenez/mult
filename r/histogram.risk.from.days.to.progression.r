library('ggplot2')

df <- read.csv('data/iss_fish_therapy_response.csv', sep='\t')

df <- df[!is.na(df$Days.to.Progression),]

print(nrow(df))

p1 <- ggplot(df) + geom_bar(stat='count', aes(as.factor(ifelse(Days.to.Progression/30 > 18, 1, 0))), alpha=.7, colour='black') +
        xlab('Months to Disease Progression') + ylab('Count') + 
        theme(text = element_text(size=11, family='serif')) + coord_flip()

ggsave('images/histogram_risk_from_days_to_progression.pdf', p1, units='cm', width=16, height=4)
