library(pROC)
p <- c(0.1, 0.2, 0.3, 0.4, 0.5,
	    0.6, 0.7, 0.8, 0.9)
r <- c(0, 1, 0, 1, 0, 1, 1, 1, 1)

dat1 <- data.frame(cbind(p,r))
auc(dat1$r, dat1$p)
plot(roc(dat1$r, dat1$p))
# abline에 v나 h 옵션을 사용하면 세로선이나 가로선을 그린다.
abline(v=0)
abline(h=0)