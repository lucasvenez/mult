library(ggplot2)
library(dplyr)
library(MESS)

df <- read.csv('data/output/best_model_estimations.csv', sep=',')

simple_roc <- function(labels, scores){
  labels <- labels[order(scores, decreasing=TRUE)]
  list(data.frame(TPR=cumsum(labels)/sum(labels), FPR=cumsum(!labels)/sum(!labels), labels))
}

p <- 
  ggplot(df, aes(y_hat, y - y_hat)) + geom_point(aes(colour=as.factor(y))) + ylim(-1, 1) + 
  geom_hline(yintercept = 0, lty=2,col="grey") + xlab('Predicted Values') + ylab('Residuals') + 
  geom_smooth() + theme(legend.position = c(0.35, .1)) + 
  scale_colour_discrete(name = "Risk", labels=c('High', 'Low')) +
  theme(text=element_text(size=11, family='serif'), legend.direction = "horizontal")

ggsave('images/residuals_best_model.pdf', p, units='cm', heigh=7, width=8)

result <- 
  df %>%
    group_by(fold) %>%
    summarize(roc = simple_roc(y, y_hat))

final_df = data.frame()

for (i in result$roc) {
  
  i['INDEX'] = 1:nrow(i)
  
  final_df <- rbind(final_df, i)
  
}

final_df <- as.data.frame(cbind(aggregate(TPR ~ INDEX, final_df, mean)$TPR,
      aggregate(TPR ~ INDEX, final_df, sd)$TPR,
      aggregate(FPR ~ INDEX, final_df, mean)$FPR,
      aggregate(FPR ~ INDEX, final_df, sd)$FPR))

colnames(final_df) <- c('TPR.mean', 'TPR.sd', 'FPR.mean', 'FPR.sd')

p <- 
  ggplot(final_df, aes(FPR.mean, TPR.mean)) + geom_line(size=1.2) + 
  geom_ribbon(aes(x=FPR.mean, ymax=TPR.mean + TPR.sd, ymin=TPR.mean - TPR.sd), alpha=0.2) +
  xlab('False Positive Ratio') + ylab('True Positive Ratio') +
  geom_label(aes(label=paste('AUC of', round(auc(final_df$FPR.mean, final_df$TPR.mean), 2)), x=.75, y=.25), size=3, family='serif', fill = "gray", alpha=.1) +
  theme(text=element_text(size=11, family='serif'))

ggsave('images/roc_curve_best_model.pdf', p, units='cm', heigh=7, width=8)
