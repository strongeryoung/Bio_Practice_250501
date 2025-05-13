
#############################
####### 01. R 이해하기 ########
#############################


########### 변수와 데이터 타입 #########
 x<-10		                 #실수형 숫자, numeric 		
 y<-as.integer(10)                 #정수형 숫자, integer
 z<-FALSE                          #논리값(TRUE, FALSE), logical 
 w<-"Hello, World!"                #문자열, character
 f<-factor(c("one","two","three")) #요인, factor

 x
 y
 z
 w
 f

 str(x)
 str(y)
 str(z)
 str(w)
 str(f)

########## 벡터 ###########

 v1<-c(1,2,3,4,5)           #숫자형 벡터
 v2<-c("a","b","c")         #문자형 벡터
 v3<-c(TRUE, FALSE, TRUE)   #논리형 벡터
 v4<-c(1,2,"c", FALSE)      #혼합형 벡터(?) 

 v1
 v2
 v3
 v4

 str(v1)
 str(v2)
 str(v3)
 str(v4)


 x<-c(1,2,3)
 y<-c(4,5,6)
 
 x+y         #벡터 덧셈
 x*2         #스칼라 연산
 x^2         #스칼라 연산
 sum(y)      #벡터 합계 계산

########## 리스트 ##########

 list2<- list(name="R", age=25, 
 scores=c(90,85,80))
 str(list2)
 list2$name
 list2$age
 list2$scores

 str(list2$scores)

########## 행렬 #########

 1:9
 seq(1,9)
 mat2<-matrix(1:9, nrow=3) 
 mat3<-matrix(seq(1,9), 
 nrow=3, byrow=T)
 mat2
 mat3

######### 데이터프레임 ##########

 data2<- data.frame(name=c("A","B"),
 score=c(30,40))
 str(data2)
 data2

######### 행렬을 데이터 프레임으로 ##########
 
 datmat3<-as.data.frame(mat3)
 datmat3
 str(datmat3)

 datmat3<-data.frame(mat3)
 datmat3
 str(datmat3)

######### 데이터 프레임을 행렬로 ##########

 matdata2<-matrix(data2)
 matdata2
 str(matdata2)

######### 조건문 if ##########

 x<-6
 if (x>=5) {            #조건 1
 print("x는 5 이상이다")} #조건 1이 참이면 실행

######### 조건문 if else ##########
 
 x<-6
 if (x>=5) {           
 print("x는 5 이상이다")}  #조건문 실행, 결과 반환
 else {                  #else 인식 오류
 print("x는 5 미만이다")   #print문 수행
 }                       #} 인식 오류
 
 x<-2
 if (x>=5) {                  #조건 1
 print("x는 5 미만이다")        #조건 1이 참이면 실행
 } else if (x>=3) {           #조건 2 
 print("x는 5미만, 3 이상이다")  #조건 1이 거짓이면 실행
 } else {
 print("x는 3미만이다")         #조건 1,2 전부 거짓이면 실행
 }

########## 조건문 for ##########

 for (i in 1:5)
 {
 print(i)
 }

 i

 k<-c(1,3,5,7,9)
 for (i in k)
 {
 print(i)
 }

 i

########## 조건문 while ########## 

 i<-1             #초기값 설정
 while(TRUE)       
 {
  print(i)
  i<-i+1
  if(i>10) break  #While문 종료 조건
 }

 i<-1                 #초기값 설정
 while(i<=20)         #while문 종료 조건
 { if(i %% 2 ==0) {   #2로 나누었을때 나머지가 0
   print(i)          
  }
  i<-i+1              #i값 1 증가
 }

########## 조건문 ifelse ##########

x<-seq(1,10)
x
ifelse(x>5, "B","A")

########## 사용자 정의 함수 ##########

 even_number <- function(a){
  if(a %% 2 ==0) { 
  return(TRUE)
  } else { return(FALSE) }
  }

 even_number(2)
 even_number(13)


 
 find_even_number <-function(a,b) {
 for (i in a:b)
 { 
  if (i %% 2 == 0) {return(i)}
 }
 }
 find_even_number(10,20)
 find_even_number(11,20)


 hist

 hist<-c("3","4","5")

 hist

 rm(list=c("hist"))

 hist


################ 실습 ##################

 find_even_number <-function(a,b) {
 m<-NA
 k<-0
 for (i in a:b)
 { 
 if (i %% 2 == 0) {k<-k+1} 
 if (i %% 2 == 0) {m[k]<-i}
 }
 return(print(m))
 }

 find_even_number(10,112)


#######################################
####### 02. 기초 통계분석 기법 이해 ########
#######################################

########## 홍차 예제 해설 ##########

1/1024

10*1/1024+1/1024

sum(choose(10,0),
choose(10,1),
choose(10,2),
choose(10,3),
choose(10,4),
choose(10,5),
choose(10,6),
choose(10,7),
choose(10,8),
choose(10,9),
choose(10,10))


############ 카이제곱 검정 예제 ##############
 
 chisq.test(c(89,41,22,8), p=c(9/16, 3/16, 3/16, 1/16)) 

 chisq.test(matrix(c(8,4,2,6), nrow=2), correct=F)

########## Fisher's exact test ###########

 fisher.test(matrix(c(8,4,2,6), nrow=2))

########### t-검정 예제 ##############  

 x<- c(31, 27, 35, 35, 32, 35, 29, 30, 33, 37, 30, 28, 34, 33, 28 ,36, 33, 30)
 t.test(x, mu=30)
 
 y<- c(28, 32, 37, 35, 28, 23, 35, 40, 33, 41, 35, 33, 31, 33, 33, 28, 30)
 var.test(x,y) 
 t.test(x,y, var.equal=T)

 
###########################################
####### 03. 임상연구에서의 회귀분석 활용 ########
###########################################

 # 아버지와 아들의 키 예제 

 x<-c(168,160,170,158,176,161,180,183,180,167,179,171,166)
 y<-c(179,169,180,160,178,170,183,187,179,172,181,173,165)

 #히스토그램과 density plot 
 
 hist(x, freq=F) 
 lines(density(x), col="red", lwd=2) 

 # 상자그림boxplot

 boxplot(x,y) 

 #상관관게
  
 cor(x,y)

 # 선형 회귀분석 
 
 ex1<- cbind(x,y)
 str(ex1)
 ex1<- data.frame(ex1)

 model1<-lm(y~x, data=ex1)
 summary(model1)

 ex1<-rbind(x,y)
 str(ex1)
 ex1<-data.frame(ex1)

 model1<-lm(y~x, data=ex1)
 summary(model1)

 summary(aov(model1))

 # 잔차분석 

 plot(model1)
 library(dplyr)
 par(mfrow=c(2,1))

 # 총 제곱합 예제 
 
 x<-c(1,1,1,3,3,3,5,5,5,7,7,7,9,9,9)
 y<-c(0,1,2,2,3,4,4,5,6,6,7,8,8,9,10)
 
 plot(x,y, ylim=c(-6,16))
 abline(lm(y~x), col="red") 
 lm(y~x) %>% summary
 lm(y~x) %>% aov %>% summary
 
 u<-c(1,1,1,3,3,3,5,5,5,7,7,7,9,9,9)
 v<-c(-6,1,8,-4,3,10,-2,5,12,0,7,14,2,9,16) 
 
 plot(u,v, ylim=c(-6,16))
 abline(lm(v~u),col="red")
 lm(v~u) %>% summary
 lm(v~u) %>% aov %>% summary


 # 제곱형태의 자료 예제 

 set.seed(1)

 x<- seq(-3,4, by=0.1)
 y<- 3+x^2+rnorm(length(x),0,1)
 
 plot(x,y)

 summary(lm(y~x))

 abline(lm(y~x), col="red", lwd=2)
 
 
 plot(lm(y~x))
 shapiro.test(lm(y~x)$residuals)

 summary(lm(y~I(x^2)))

 plot(lm(y~I(x^2)))

 plot(x^2,y)
 abline(lm(y~I(x^2)), col="blue", lwd=1)
 shapiro.test(lm(y~I(x^2))$residuals)

 # 다중 회귀분석 

 mtcars
 ?mtcars
 cor(mtcars)
 data.frame(cor(mtcars))
 
 model2<-lm(mpg ~., data=mtcars)
 summary(model2)

 par(mfrow=c(2,2))
 plot(model2)

 library(car)
 vif(model2)

 # 변수 선택법 

 null<-lm(mpg ~ 1, data=mtcars)
 full<-lm(mpg ~ ., data=mtcars)

 summary( step(null, direction= "forward", scope=list(lower=null, upper=full)) )

 summary( step(full, direction= "backward") )

 summary( step(full, direction= "both") )


############################################
####### 04. 분류 모델을 이용한 질병 예측 ########
############################################
 
 # 로지스틱 회귀분석 

 # https://www.kaggle.com/code/abdmental01/heart-disease-prediction-binary-classification/notebook

 #setwd("file directory") ## R 디폴트는 내 문서
 data<-read.csv("heart_disease_uci.csv")
 str(data)
 is.na(data)
 dim(data)
 data2<-na.omit(data)
 dim(data2)
 data2$num

 # 결측치 제거 
 
 data2<-filter(data, if_all(everything(), ~!is.na(.)& .!=""))

 # 변수 변환

 data2$target<-ifelse(data2$num>0,1,0)

 data2$target

 #모형 적합
 model<-glm(target ~ age+sex+dataset+cp+trestbps+chol+
                     fbs+restecg+thalch+exang+oldpeak+slope+
                     ca, family=binomial, data=data2)

 data2sub<-subset(data2, select=c(-id,-num))
 colnames(data2sub)
 
 model2<-glm(target ~. , family="binomial", data=data2sub)

 summary(model2)

 table(data2$sex,data2$target)

 model<-glm(target ~ sex, family=binomial, data=data2)
 summary(model)

 #오즈비 계산
 
 exp(1.2919)

 (72/25)/(91/115)

 (72*115)/(91*25)

 exp(1.546)

 model<-glm(target ~ age+sex+dataset+cp+trestbps+chol+
                     fbs+restecg+thalch+exang+oldpeak+slope+
                     ca+thal, family=binomial, data=data2)
 summary(model)
 
 
 # 로지스틱 회귀분석 적합

 data<-read.csv("heart_disease_uci.csv")
 data$target<-ifelse(data$num>0,1,0)
 data2<-filter(data, if_all(everything(), ~!is.na(.)& .!=""))
 data2sub<-subset(data2, select=c(-id, -num))
 colnames(data2sub)

 model2<-glm(target ~ ., family=binomial, data=data2sub)
 summary(model2)

 factor(data2$dataset)

 table(data2sub$dataset)
 table(data2sub$cp)
 table(data2sub$restecg)
 table(data2sub$slope)
 table(data2sub$thal)

 #로지스틱 회귀 모형 해석

 model_step<-step(model2, direction="both", k=2) 
 model_step<-summary

 str(data) 
 hist(table(data$fbs))

 str(table(data$fbs))

 hist(data$chol)
 barplot(table(data$restecg))

 library(caret)
 library(randomForest)
 library(ggplot2)

 # 변수 중요도

 vi<-varImp(step(model2, direction="both"))

 plot(vi)

 barplot(names.arg=rownames(vi), 
         vi$Overall,  cex.names=0.7)


 var_imp_df <- data.frame(Variable = rownames(vi), 
                         Importance = vi$Overall)

  ggplot(var_imp_df, aes(x = Variable, y = Importance)) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_minimal()

 varImp_df<-data.frame(X=rownames(varImp(model2)),Y=varImp(model2)$Overall)
 varImp_df

 #detach("package:caret", unload=TRUE)
 #detach("package:ggplot2", unload=TRUE)


 # 훈련 데이터셋 로지스틱 회귀 모형 적합
 set.seed(42)
 library(caret)
 train_index<-createDataPartition(data2sub$target, p=0.7, list=FALSE)
 train_data<-data2sub[train_index, ]
 test_data<-data2sub[-train_index, ]
 
 dim(data2sub)
 dim(train_data)
 dim(test_data)

 train_model<-glm(target ~.,
                  family="binomial",
                  data=train_data)
 summary(step(train_model))

 table(train_data$thal)
 
 #다중공선성 

 vif(train_model)

 str(train_data)

 train_model<-lm(target ~.,data=train_data)
 summary(step(train_model))

 vif(step(train_model))

 #적합 모형을 검증데이터에 적용

 test_data$prob<-predict(step(train_model), test_data, type="response") 
 
 test_data$pred<-ifelse(test_data$prob>0.5, 1,0)

 #혼동행렬과 성능평가

 confusionMatrix(test_data$pred, test_data$target)

 confusionMatrix(factor(test_data$pred), factor(test_data$target))
 
 (58)/(50+23+8+8) ## No Information Rate

#################################################
########## 05. 군집화를 통한 환자 그룹 분류 ##########
#################################################


 # 군집화 예시

 set.seed(42)
 a1<- rnorm(100, 1,1)
 a2<- rnorm(100, 3,0.5)
 a<-cbind(a1,a2)
 b1<- rnorm(100, 3,1)
 b2<- rnorm(100, 1,0.5)
 b<-cbind(b1,b2)

 data<-data.frame(rbind(a,b))
 colnames(data)<-c("X","Y")
 plot(data)

 plot(data, xlim=c(-2,6),ylim=c(-2,6), lwd=2)
 abline(lm(Y~X, data=data), lwd=2)

 plot(a, xlim=c(-2,6), ylim=c(-2,6), xlab="X", ylab="Y",
      col="red", lwd=2)
 lines(b, col="blue", type="p", lwd=2) # 또는 points(b, col="blue")

 abline(lm(a2~a1,data=data.frame(a)), col="red", lwd=2)
 abline(lm(b2~b1,data=data.frame(b)), col="blue", lwd=2)


 # 데이터 전처리

 data<-data.frame(data)
 data$real<-c(rep(1,100),rep(2,100))

 #k-means

 data_km <- kmeans(data, centers = 2)
 data$kmcluster<-data_km$cluster
 data$kmcluster<-ifelse(data$kmcluster==1,2,1)


 data_scale_km <- kmeans(scale(data), centers =2)
 data$scale_kmcluster<-data_scale_km$cluster
 data$scale_kmcluster<-ifelse(data$scale_kmcluster==1,2,1)

 table(data$real,
       data$kmcluster)

 table(data$real,
       data$scale_kmcluster)


 library(cluster)
 library(factoextra)

 kmeans(scale(data), centers=2, nstart= 25)$cluster
 fviz_cluster(kmeans(scale(data), 2, nstart= 10), data = scale(data))


 # 거리 기반 계층적 군집화

 hc <- hclust(dist(scale(data)), method = "ward.D2")  
 plot(hc)


 library(dbscan)

 db <- dbscan(scale(data), 
              eps = 0.5, minPts = 5 )
 data$dbcluster<-db$cluster
 table(data$real, data$dbcluster)
 
 fviz_cluster(db, data=scale(data), geom="point")

 
 # GMM(Gaussian mixture model)

 library(mclust)
 gmm <- Mclust(scale(data), G=2)
 data$gmmcluster <- gmm$classification

 table(data$real, data$gmmcluster)



 #군집화 실습

 #이전 데이터 불러오기 

 data<-read.csv("heart_disease_uci.csv")
 data$target<-ifelse(data$num>0,1,0)
 data2<-filter(data, if_all(everything(), ~!is.na(.)& .!=""))
 data2sub<-subset(data2, select=c(-id, -num))
 colnames(data2sub)

 library(dplyr)

 str(data2sub)
 data2sub2<-subset(data2sub, select=c(
               age, trestbps,chol, thalch, oldpeak, ca))
 fviz_cluster(Mclust(data2sub2, G=2))

 Mclust(data2sub, G=2)$classification
 data2sub$class<-Mclust(data2sub2, G=2)$classification
 
 model3<-glm(target ~ ., family="binomial", data=data2sub)
 summary(step(model3)) 

 # 오류 발생

 model3_2<-glm(target ~ .-class, family="binomial",
               data=filter(data2sub,data2sub$class==2))

 # 데이터 재탐색

 filter(data2sub,data2sub$class==2) %>% str
 filter(data2sub,data2sub$class==2) %>% summary

 table(filter(data2sub,data2sub$class==2)$dataset)
 table(filter(data2sub,data2sub$class==2)$fbs)
 table(filter(data2sub,data2sub$class==2)$exang)

 #모델 재설정

 model3_2<-glm(target ~ .-dataset , family="binomial",
               data=filter(data2sub,data2sub$class==2))

 data2sub$class
 str(data2sub)
 str(filtered)

?select
 library(dplyr)
 detach("package:MASS", unload=TRUE)
 filtered <- filter(data2sub, data2sub$class==2)
 filtered <- dplyr::select(filtered, -c(dataset)) 

 model3_2<-glm(target ~., family="binomial", data=filtered)

 filtered2 <- filter(data2sub, data2sub$class==1)
 filtered2 <- dplyr::select(filtered2, -c(dataset)) 

 model3_3<-glm(target ~., family="binomial", data=filtered2)

 summary(step(model3_2)) ## class ==2
 summary(step(model3_3)) ## class ==1
  

#######################################################
########## 06. 특성 중요도 분석을 통한 의학적 해석 ##########
#######################################################

 library(dplyr)
 library(caret)
 library(randomForest)
 library(xgboost)
 library(SHAPforxgboost)
 library(Matrix)
 library(ggplot2) 
  
 #로지스틱 회귀분석 특성 중요도 분석 실습

 set.seed(42)
 data <- read.csv("heart_disease_uci.csv")
 data$target<-ifelse(data$num>0, 1,0)
 data2 <- filter(data, if_all(everything(), ~!is.na(.) & .!=""))
 data2sub<- subset(data2, select=c(-id,-num))
 train_index<-createDataPartition(data2sub$target, p=0.8, list=FALSE)
 train_data<-data2sub[train_index, ]
 test_data<-data2sub[-train_index, ]
 
 logit_model <- glm(target ~. , data=train_data, family="binomial")

 summary(step(logit_model))

 logit_importance <- as.data.frame(summary(step(logit_model))$coefficients)
 logit_importance$Variable <- rownames(logit_importance)
 colnames(logit_importance) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)", "Variable")

 ggplot(logit_importance, 
      aes(x = reorder(Variable, abs(Estimate)), 
      y = abs(Estimate))) +
 geom_bar(stat = "identity", fill = "steelblue") +
 coord_flip() +
 labs(title = "Logistic Regression Feature Importance", 
      x = "Variable", 
      y = "Absolute Coefficient") +
 theme_minimal()

 # 랜덤 포레스트 모델 학습 

 train_data$target <- as.factor(train_data$target)  

 test_data$target <- as.factor(test_data$target)

 rf_model <- randomForest(target ~ ., data = train_data, 
                         importance = TRUE, ntree = 500)

 varImpPlot(rf_model)
 
 # XGboost 모델 학습

 train_matrix<- model.matrix(target~.-1, data=train_data)
 train_label <- as.numeric(train_data$target)
 dtrain <- xgb.DMatrix(data=train_matrix, label=train_label)
 params <- list(
		objective = "binary:logistic",
		eval_metric = "logloss",
		tree_method = "auto"
		)

 xgb_model <-xgb.train(params, dtrain, nrounds = 100, verbose = 0)
 importance <- xgb.importance(model= xgb_model)
 importance
 xgb.plot.importance(importance)

 #shap 

 shap_values <- predict(xgb_model, dtrain, predcontrib = TRUE)
 shap_values <- as.data.frame(shap_values)
 colnames(shap_values) <- c(colnames(train_matrix), "bias")  # 컬럼명 정리, bias 추가
 shap_values <- shap_values[, !colnames(shap_values) %in% "bias"]
 shap_long <- shap.prep(shap_contrib = shap_values, X = as.data.frame(train_matrix))

 #shap 그림
 shap.plot.summary(shap_long)

 #상호작용 확인
 shap.plot.dependence(data_long=shap_long, x="age", color_feature="ca")


###########################################
########## 07. ROC 곡선과 모델 평가 ##########
###########################################

 # ROC 직접 그려보기
 
 library(pROC)
 p<-c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
 r<-c(0, 1, 0, 1, 0, 1, 1, 1, 1)
 dat1<-data.frame(cbind(p,r))
 auc(dat1$r, dat1$p)
 plot(roc(dat1$r, dat1$p))

 threshold<-0.95
 dat1$p2<-ifelse(dat1$p>threshold,1,0)

 t2<-table(factor(dat1$r, levels=c(0,1)), 
 		factor(dat1$p2, levels=c(0,1)))

 sens<-t2[2,2]/(t2[2,2]+t2[2,1])
 spec<-t2[1,1]/(t2[1,1]+t2[1,2])

 print(t2)
 print(threshold)
 print(c(1-spec,sens))

 x<-c(0, 0, 0, 0.3333, 0.3333, 0.6667, 0.6667, 1)
 y<-c(0, 0.1667, 0.6667, 0.6667, 0.8333, 0.8333, 1, 1)
 plot(x,y, type="l", lwd=2, xlim=c(-0.2, 1.1), ylim=c(-0.2, 1.1))
 abline(0,0)
 abline(0,1)
 abline(v=0)
 abline(v=1)
 abline(v=0.3333)
 abline(v=0.6667)
 abline(h=0.6667)



 # heart 검증 데이터로 모형적합

 data <- read.csv("heart_disease_uci.csv")
 data$target <- ifelse(data$num > 0, 1, 0)
 data2 <- filter(data, if_all(everything(), ~!is.na(.) & .!=""))
 data2sub <- subset(data2, select=c(-id, -num, -dataset))
 set.seed(42)
 train_index <- createDataPartition(data2sub$target, p = 0.8, list = FALSE)
 train_data <- data2sub[train_index, ]
 test_data <- data2sub[-train_index, ]

 ## 로지스틱
 logit_model <- glm(target ~ ., data = train_data, family = binomial)
 logit_pred <- predict(logit_model, newdata=test_data, type = "response")
 
 ## 랜덤포레스트
 train_data$target <- as.factor(train_data$target)  
 rf_model <- randomForest(target ~ ., data = train_data, importance = TRUE, ntree = 500)
 rf_pred <- predict(rf_model, newdata=test_data, type = "prob")[,2]

 ## XGboost
 train_matrix <- model.matrix(target ~ . - 1, data = train_data)     
 train_label <- as.numeric(as.character(train_data$target))
 dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
 params <- list(
   objective = "binary:logistic",
   eval_metric = "logloss",
   tree_method = "auto"
 )
 xgb_model <- xgb.train(params, dtrain, nrounds = 100, verbose = 0)

 test_matrix <- model.matrix(target ~ . - 1, data = test_data)
 test_label <- as.numeric(as.character(test_data$target))
 dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

 # 또 오류발생 

 xgb_pred <- predict(xgb_model, dtest)

 colnames(train_matrix)
 colnames(test_matrix) 
 
 table(train_data$restecg, test_data$restecg)
 table(test_data$restecg)

 setdiff(colnames(train_matrix), colnames(test_matrix))

 levels(factor(train_data$restecg))
 levels(factor(test_data$restecg))

 test_data$restecg <- factor(test_data$restecg, levels = levels(factor(train_data$restecg)))

 test_matrix <- model.matrix(target ~ . - 1, data = test_data)
 test_label <- as.numeric(as.character(test_data$target))
 dtest <- xgb.DMatrix(data = test_matrix, label = test_label)
 
 # 오류 해결
 xgb_pred <- predict(xgb_model, dtest)

 # 모델 별 AUC 확인
 logit_auc <- roc(test_data$target, logit_pred)
 logit_pred_target<- ifelse(logit_pred>0.5, 1,0)
 table(test_data$target, logit_pred_target)
 auc(logit_auc)

 rf_auc <- roc(test_data$target, rf_pred)
 rf_pred_target<- ifelse(rf_pred>0.5, 1,0)
 table(test_data$target, rf_pred_target)
 auc(rf_auc)

 xgb_auc <- roc(test_data$target, xgb_pred)
 xgb_pred_target<- ifelse(xgb_pred>0.5, 1,0)
 table(test_data$target, xgb_pred_target)
 auc(xgb_auc)


 # 모델 별 AUC 그리기 
 plot(logit_auc, col = "blue", main = "ROC Curves")
 legend("bottomright", legend = c("Logistic"),
        col = c("blue"), lwd = 2)

 plot(rf_auc, col = "red", main = "ROC Curves")
 legend("bottomright", legend = c("Random Forest"),
        col = c("red"), lwd = 2)

 plot(xgb_auc, col = "green", main = "ROC Curves")
 legend("bottomright", legend = c("XGBoost"),
        col = c("green"), lwd = 2)

 # 한 번에 그리기 
 plot(logit_auc, col = "blue", main = "ROC Curves")
 plot(rf_auc, col = "red", add = TRUE)
 plot(xgb_auc, col = "green", add = TRUE)
 legend("bottomright", 
        legend = c("Logistic", "Random Forest", "XGBoost"),
        col = c("blue", "red", "green"), lwd = 2)

 #혼동행렬 및 성과지표 

 TN<-table(test_data$target, rf_pred_target)[1,1]
 FP<-table(test_data$target, rf_pred_target)[1,2]
 FN<-table(test_data$target, rf_pred_target)[2,1]
 TP<-table(test_data$target, rf_pred_target)[2,2]

 accuracy <- (TP + TN) / (TN+FP+FN+TP)  
 sensitivity <- TP / (TP + FN) 
 specificity <- TN / (TN + FP)  
 precision <- TP / (TP + FP)  
 f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)  

 print(c(accuracy, sensitivity,specificity,precision, f1_score))

 #또는 


 performance_metrics<- data.frame(
     Accuracy= accuracy,
     Sensitivity=sensitivity,
     Specificity=specificity,
     Precision=precision,
     F1_score=f1_score
     )
 performance_metrics

################ 실습 #################

 threshold<-seq(0.1, 0.9, by=0.1)
 logit_result<-matrix(NA,length(threshold),6)
 colnames(logit_result)<-c("threshold","Accuracy","Sensitivity",
                           "Specificity","Precision","F1_score")
 logit_result<-data.frame(logit_result)

 for ( i in 1:length(threshold))
  {
  logit_pred_target<- ifelse(logit_pred>threshold[i],1,0)
  TN<-table(test_data$target, logit_pred_target)[1,1]
  FP<-table(test_data$target, logit_pred_target)[1,2]
  FN<-table(test_data$target, logit_pred_target)[2,1]
  TP<-table(test_data$target, logit_pred_target)[2,2]

  accuracy <- (TP + TN) / (TN+FP+FN+TP)  
  sensitivity <- TP / (TP + FN) 
  specificity <- TN / (TN + FP)  
  precision <- TP / (TP + FP)  
  f1_score <- 2 * (precision * sensitivity) /
                  (precision + sensitivity)  

  logit_result$threshold[i]<-threshold[i]
  logit_result$Accuracy[i]<-accuracy
  logit_result$Sensitivity[i]<-sensitivity
  logit_result$Specificity[i]<-specificity
  logit_result$Precision[i]<-precision
  logit_result$F1_score[i]<-f1_score
  }
 
  logit_result



######################################
########## 08. 생존 분석 기법 ##########
######################################

 library(survival)

 # 달리기 예제 로그랭크 검정

 run_data <- data.frame(
   group = c("A", "A", "A", "A", "B", "B", "B", "B"),
   time = c(15.2, 17.5, 14.0, 20.0, 20.0, 18.0, 17.0, 19.5),  
   event = c(1,1,1,0,0,1,1,1))

 log_rank_test <- survdiff(Surv(time, event) ~ group, 
                           data = run_data)

 # 폐암 환자 생존률 데이터
  
 library(survival)
 library(survminer)
 df<-lung
 ?lung
 str(lung)
 df$event<-ifelse(df$status==2, 1,0)

 # 카플란 마이어 
 
 km_fit<- survfit(Surv(time, event) ~ 1, data=df)
 plot(km_fit)
 ggsurvplot(km_fit, data=df) 

 str(df)

 km_fit_sex<- survfit(Surv(time, event) ~ sex, data=df)

 
 # 카플란 마이어 생존 곡선 (strata)  
 
 ggsurvplot(km_fit_sex, data=df, pval=T, pval.coord=c(500,1),
            risk.table=T,
            title="성별에 따른 K-M 생존 곡선") 

 # 카플란 마이어 실습
 df$age %>% summary
 df$age_cat<-NA
 hist(df$age)
 df$age_cat<-ifelse(df$age<55,1,df$age_cat)
 df$age_cat<-ifelse(df$age>=55 & df$age<70,2,df$age_cat)
 df$age_cat<-ifelse(df$age>=70,3,df$age_cat)
 df$age_cat %>% table


 df$age_cat<-NA
 df$age_cat<-ifelse(df$age<70,1,df$age_cat)
 df$age_cat<-ifelse(df$age>=70,2,df$age_cat)
 df$age_cat %>% table
 km_fit_age<- survfit(Surv(time, event) ~ age_cat, data=df)
 ggsurvplot(km_fit_age, data=df, pval=T, pval.coord=c(500,1),
            risk.table=T,
            title="연령대에 따른 K-M 생존 곡선")  

 # 콕스 비례 위험 모형
 
 cox_model <- coxph(Surv(time, event) ~ 
                    sex+age+ph.ecog, data=df)
 cox_model %>% summary

 # forest plot
 
 ggforest(cox_model,data=df)

 # 비례 위험 가정 검정 

 cox.zph(cox_model)



 cox_model <- coxph(Surv(time, event) ~ 
                    sex+age+ph.ecog+ph.karno, data=df)
 cox_model %>% summary

 cox.zph(cox_model)



 cox_model <- coxph(Surv(time, event) ~ 
                    sex+age+ph.ecog+strata(ph.karno), data=df)
 cox_model %>% summary

 cox.zph(cox_model)


 fit <- survfit(Surv(time, event) ~ ph.karno, data = df)
 ggsurvplot(fit, data=df, pval=T, pval.coord=c(500,1))
 
 cox_model <- coxph(Surv(time, event) ~ 
                    sex+age+ph.ecog+ph.karno, data=df)
 cox_model %>% summary
 cox.zph(cox_model)
 
 ggforest(cox_model, data=df)


######################################
########## 09. 차원 축소 기법 ##########
######################################

library(plotly)
library(MASS)

 # 3D interactive plot

 set.seed(42)
 mean_vec <- c(0, 0, 0)
 cov_matrix <- matrix(c(3, 1, 1,
                        1, 2, 1,
                        1, 1, 1), nrow = 3)

 data3D <- mvrnorm(n = 100, mu = mean_vec, Sigma = cov_matrix)

 df3D <- as.data.frame(data3D)
 colnames(df3D) <- c("X1", "X2", "X3")

 set.seed(42)
 clusters <- kmeans(df3D[, c("X1", "X2", "X3")], centers = 3)
 df3D$cluster <- as.factor(clusters$cluster)

 plot_ly(df3D, x = ~X1, y = ~X2, z = ~X3, color = ~cluster,
         colors = c("red", "blue", "green"), type = "scatter3d",
         mode = "markers", marker = list(size = 5)) %>%
   layout(title = "3D 데이터 (K-Means 군집 기반 색상 구분)")



 install.packages("ggplot2")
 install.packages("FactoMineR")  # PCA 분석 패키지
 install.packages("factoextra")  # PCA 시각화 패키지

 library(ggplot2)
 install.packages("FactoMineR")
 library(FactoMineR)
 install.packages("factoextra")
 library(factoextra)
 library(dplyr)

 #Swiss 데이터 

 data("swiss")
 df <- swiss  
 df %>% str
 ? swiss

 # PCA 수행 (표준화 포함)

 pca_result <- prcomp(df, center = TRUE, scale. = TRUE)
 str(pca_result)
 pca_result$eigen
 summary(pca_result)

 pca_result$sdev^2  %>% sum

 3.1997/6 + 1.1883/6
 
 fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 75))

 pca_result$rotation  
 pca_result$x[1:5, ]

 t(pca_result$rotation) %*% pca_result$rotation %>% round(4)

 t(pca_result$rotation) %*% pca_result$rotation


 # Biplot (데이터 포인트 + 변수 벡터)

 fviz_pca_biplot(pca_result, repel = TRUE, 
                 col.var = "red", col.ind = "blue")

 # 변수 기여도 시각화

 fviz_pca_var(pca_result, col.var = "contrib", 
              gradient.cols = c("blue", "red"))

 # 개별 데이터의 주성분 분포

 fviz_pca_ind(pca_result, col.ind = "cos2", gradient.cols = c("blue", "red"))

 # 요인분석 (mtcars)

 #install.packages("psych")
 library(psych)
 library(car)
 df<-mtcars

 str(df)
 which(is.na(df))

 cor(df)
 library(car)
 lm(mpg ~., data=df)
 vif( lm(mpg ~., data=df))

 # KMO 검사
 
 KMO(df)


 # Bartlett 구형성 검정
 
 cortest.bartlett(cor(df), n=nrow(df))


 #요인 수 설정 

 eigen(cor(scale(df)))
 eigen(cor(scale(df)))$value %>% plot
 plot(eigen(cor(scale(df)))$value, type="l")
 
 #또는 
 VSS.scree(scale(df))

 # 요인 분석 결과 해석
 
 ?fa
 fa_result <- fa(df, nfactors=3, rotate="varimax", fm="ml")  
 print(fa_result, digits=5) 





