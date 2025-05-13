library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(SHAPforxgboost)
library(Matrix)
library(ggplot2)

# 1. target 변수의 형식을 범주형으로 변환 (랜덤 포렛스트 분류 문제이기 때문에)
train_data$target <- as.factor(train_data$target)

# 2. 랜덤 포레스트 모델 생성
#    target ~ . : target을 제외한 모든 변수 사용
#    data = train_data : 학습 데이터 지정
#    importance = TRUE : 변수 중요도를 계산하겠다는 옵션
#    ntree = 500 : 트리를 500개 생성

rf_model <- randomForest(target ~ ., data = train_data,
						   importance = TRUE, ntree =500)

# 3. 학습된 모델에서 변수 중요도 추출
rf_importance <- importance(rf_model)
rf_importance