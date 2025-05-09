# x,y 자료 입력
# 회귀분석은 서로 대응되는 쌍의 자료로 이루어지므로 순서가 바뀌면 안됨
x<-c(168,160,170,158,176,161,180,183,180,167,179,171,166)
y<-c(179,169,180,160,178,170,183,187,179,172,181,173,165)

hist(x, freq=F)
lines(density(x), col="red", lwd=2)

plot(x,y)

boxplot(x,y)

length(x)  # x, y 자료의 길이 비교
length(y)

lm(y~x)  # y를 종속변수로, x를 독립변수로 하는 선형회귀 모형 적합
summary(lm(y~x))  #해당 모형의 내용 요약


# 다른 방법 (데이터프레임 이용)
x<-c(168,160,170,158,176,161,180,183,180,167,179,171,166)
y<-c(179,169,180,160,178,170,183,187,179,172,181,173,165)

exl<-cbind(x,y) #x,y자료 합치기
str(exl)
exl<-data.frame(exl)  #데이터프레임 형태로 변환, 생략한 경우 오류 발생

modell<-lm(y~x, data=exl)  #exl데이터 내부에 있는 y, x를 불러오는 경우 data= 항목이 필요
summary(modell)

modell  #결과해석

summary(modell)

install.packages("writexl")
library(writexl)

write_xlsx(data.frame(cor(mtcars)), "C:/cor.xlsx")
getwd()
setwd("C:/")

write_xlsx(data.frame(cor(mtcars)), "0228.xlsx")

model2<-lm(mpg ~.,data=mtcars)
summary(model2)

shapiro.test(model2$residuals)

