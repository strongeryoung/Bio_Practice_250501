library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(SHAPforxgboost)
library(Matrix)
library(ggplot2)

# 1. model.matrix를 사용해 데이터의 범주형 변수들을 
# one-hot encoding으로 변경 (xgboost는 문자형 못 씀)
train_matrix <- model.matrix(target ~ . -1, data=train_data)

# 2. 라벨 벡터 생성. factor -> character -> numeric 변환
# (xgboost는 label도 숫자여야 한다.)
train_label <- as.numeric(as.character(train_data$target))

# 3. XGBoost 전용 데이터 형식으로 변환 (메모리 최적화 및 빠른 학습)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

# 4. 모델 학습에 사용할 파라미터
params <- list(
	objective = "binary:logistic",  # 이진분류
	eval_metric = "logloss",        # 로스 함수로 로그 손실 사용
	tree_method = "auto"            # 트리 생성 방식 자동 선택
)

# 5. XGBoost 모델 학습 (100회 반복, verbose = 0 : 학습 로그 생략)
xgb_model <- xgb.train(params, dtrain, nrounds = 100, verbose = 0)

# 6. 변수 중요도 추출 (Gain, Cover, Frequency 중에서 기본값 Gain)
importance <- xgb.importance(model = xgb_model)

# 7. 중요도 출력
importance

# 8. 중요도 시각화
xgb.plot.importance(importance)