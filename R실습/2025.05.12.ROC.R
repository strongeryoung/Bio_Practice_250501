p <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
r <- c(0, 1, 0, 1, 0, 1, 1, 1, 1)
dat1 <- data.frame(cbind(p, r))

threshold <- 0.5  # threshold를 0.95에서 0.1씩 빼 가면서 계산해보자

dat1$p2 <- ifelse(dat1$p > threshold, 1, 0)

# 범주가 하나로만 나오는 경우를 방지하기 위한 factor 설정
t2 <- table(
  factor(dat1$r, levels = c(0, 1)),
  factor(dat1$p2, levels = c(0, 1))
)

# 혼동되지 않도록 주의
sens <- t2[2, 2] / (t2[2, 2] + t2[2, 1])  # 민감도 = TP / (TP + FN)
spec <- t2[1, 1] / (t2[1, 1] + t2[1, 2])  # 특이도 = TN / (TN + FP)

# 혼동행렬, threshold, 결과지표 출력
print(t2)
print(threshold)
print(c(1 - spec, sens))  # 1 - 특이도(FPR), 민감도(TPR)

# 2. ROC 곡선 그리기

x <- c(0, 0, 0, 0.3333, 0.3333, 0.6667, 0.6667, 1)
y <- c(0, 0.1667, 0.6667, 0.6667, 0.8333, 0.8333, 1, 1)

plot(x, y, type = "l", lwd = 2,  # xlab = "x", ylab = "y",
     xlim = c(-0.2, 1.1), ylim = c(-0.2, 1.1))

points(x, y, pch = 19)

text(x, y, labels = paste0("(", x, ",", y, ")"),
     adj = c(1.1, -1.1), cex = 0.7)

abline(0, 1)         # 대각선
abline(0, 0)         # 원점 기준선

abline(v = 0)        # 수직선 (x=0)
abline(v = 1)        # 수직선 (x=1)
abline(v = 0.3333)   # 수직선 (x=0.3333)
abline(v = 0.6667)   # 수직선 (x=0.6667)
abline(h = 0.6667)   # 수평선 (y=0.6667)
