library(ggplot2)

df <- read.csv('data/output/pca_with_resp.csv')


p <- ggplot(df) + geom_point(aes(PCA1, PCA2, colour=as.factor(Greater18Months.Days.to.Progression))) + 
  scale_colour_discrete(name = "Risk", labels=c('Hight', 'Low'))

ggsave('images/pca_all_vars.pdf', p, units='cm', width=16, height=10)

df <- read.csv('data/output/pca_with_resp_and_reduction.csv')

p <- ggplot(df) + geom_point(aes(PCA1, PCA2, colour=as.factor(Greater18Months.Days.to.Progression))) + 
  scale_colour_discrete(name = "Risk", labels=c('Hight', 'Low'))

ggsave('images/pca_reduced_gens.pdf', p, units='cm', width=16, height=10)
