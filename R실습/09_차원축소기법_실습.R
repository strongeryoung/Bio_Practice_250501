install.packages("plotly")
library(plotly)
library(MASS)

set.seed(42)
mean_vec <- c(0,0,0)
cov_matrix <- matrix(c(3, 1, 1,
                       1, 2, 1,
                       1, 1, 1), nrow = 3)
data3D <- mvrnorm(n=100, mu=mean_vec, Sigma=cov_matrix)
df3D <- as.data.frame(data3D)
colnames(df3D) <- c("X1", "X2", "X3")

set.seed(42)
clusters <- kmeans(df3D[, c("X1", "X2", "X3")], centers = 3)
df3D$cluster <- as.factor(clusters$cluster)

plot_ly(df3D, x=~X1, y=~X2, z=~X3, color=~cluster,
	  colors = c("red", "blue", "green"), type = "scatter3d", mode="makers",
	  makers = list(size=5)) %>%
	layout(title = "3D데이터 (K-Means 군집 기반 색상 구분)")



# 주성분 분석(Principle Component Analysis)
# Swiss 데이터
install.packages("factoextra")
library(ggplot2)
library(FacoMineR)
library(factoextra)
library(dplyr)

data("swiss")
df <- swiss
df %>% str
? swiss

# 주성분 분석 수행
pca_result <- prcomp(df, center = TRUE, scale.=TRUE)
summary(pca_result)
fviz_eig(pca_result, addlabels=TRUE, ylim = c(0, 75))

## 기존의 변수는 새로운 주성분들의 조합으로 표현 가능
pca_result$rotation

## 새로운 주성분들의 직교 확인
round(t(pca_result$rotation) %*% pca_result$rotation, 5)
# - 행렬곱 (%*%)함수를 통해 벡터의 내적을 확인 가능. 주성분 간 행렬곱 결과가 0인 단위벡터가 나오는 것을 확인

# ** 파이프함수로도 사용 가능
t(pca_result$rotation) %*% pca_result$rotation %>% round(4)


# Biplot
fviz_pca_biplot(pca_result, repel=TRUE, col.var="red", col.ind="blue")

# - 파란점(개별데이터), 빨간 화살표(변수 로딩), 화살표 방향이 비슷한 변수들은 상관관계가 있음


# 변수 기여도 시각화
fviz_pca_var(pca_result, col.var="contrib", gradient.cols = c("blue", "red"))
# 화살표는 각 변수들이 주성분 1,2에 기여하는 정도를 나타냄
# 변수 로딩 벡터는 원 내부에 위치
# 색상은 각 기여도의 정도를 나타냄



# 요인분석 
# 요인분석 예제 실습
# mtcars 사용
install.packages("psych")
library(psych)
str(df)
which(is.na(df))

KMO(df)
cortest.bartlett(df)

# 요인분석 전 데이터 체크
KMO(df)
