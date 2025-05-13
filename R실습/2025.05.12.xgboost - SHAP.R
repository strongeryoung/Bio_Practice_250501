library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(SHAPforxgboost)
library(Matrix)
library(ggplot2)

# 1. SHAP 값을 예측값 대신 반환하도록 설정 (predcontrib = TRUE)
#    각 샘플에 대해 각 변수들이 예측에 기여한 정도 반환
shap_values <- predict(xgb_model, dtrain, predcontrib = TRUE)

# 2. 결과를 데이터프레임 형태로 변환
shap_values <- as.data.frame(shap_values)

# 3. 컬럼 이름 지정: 기존 feature 이름 + "bias" (마지막 열은 bias term)
colnames(shap_values) <- c(colnames(train_matrix), "bias")

# 4. bias term을 분석 대상에서 제외하고 제거 (원하면 남겨도 됨)
shap_values <- shap_values[, !colnames(shap_values) %in% "bias"]

# 5. shap.prep 함수를 이용해 shap 값을 long format으로 변환
# shap_contrib에는 SHAP 값, x에는 원본 입력값(train_matrix) 전달
shap_long <- shap.prep(shap_contrib = shap_values,
						 X = as.data.frame(train_matrix))

# 6. SHAP summary plot 생성 (변수별 평균 영향력 + 밀도 시각화)
shap.plot.summary(shap_long)

shap.plot.dependence(data_long = shap_long, x= "age")

shap.plot.dependence(data_long = shap_long, x="age",
					   color_feature = "ca")