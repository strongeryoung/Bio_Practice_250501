library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(SHAPforxgboost)
library(Matrix)
library(ggplot2)
library(pROC)


data <- read.csv("heart_disease_uci.csv")
data$target <- ifelse(data$num > 0, 1, 0)

data2 <- filter(data, if_all(everything(), ~!is.na(.) & . != ""))
data2sub <- subset(data2, select = c(-id, -num, -dataset))  # 상술한 Contrast 문제로 dataset 변수는 제외

set.seed(42)
train_index <- createDataPartition(data2sub$target, p = 0.8, list = FALSE)
train_data <- data2sub[train_index, ]
test_data <- data2sub[-train_index, ]

# 1. 로지스틱 회귀
logit_model <- glm(target ~ ., data = train_data, family = binomial)
logit_pred <- predict(logit_model, newdata = test_data, type = "response")  # 예측값을 확률로 반환

## 2. 랜덤포레스트
train_data$target <- as.factor(train_data$target)

rf_model <- randomForest(target ~ ., data = train_data, importance = TRUE, ntree = 500)
rf_pred <- predict(rf_model, newdata = test_data, type = "prob")[, 2]  # 클래스 1의 확률만 추출

## 3. XGboost
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


# 에러 fix!! : train_data의 restecg 변수의 factor level을 
# 			   test data의 restecg 변수에 옮겨 적용하면 에러가 발생하지 않는다.
test_data$restecg <- factor(test_data$restecg,
						  	  levels = levels(factor(train_data$restecg)))

xgb_pred <- predict(xgb_model, dtest)


# 로지스틱 회귀 모델 ROC & AUC
logit_auc <- roc(test_data$target, logit_pred)
logit_pred_target <- ifelse(logit_pred > 0.5, 1, 0)
table(test_data$target, logit_pred_target)
auc(logit_auc)

# 랜덤 포레스트 모델 ROC & AUC
rf_auc <- roc(test_data$target, rf_pred)
rf_pred_target <- ifelse(rf_pred > 0.5, 1, 0)
table(test_data$target, rf_pred_target)
auc(rf_auc)

# XGBoost 모델 ROC & AUC
xgb_auc <- roc(test_data$target, xgb_pred)
xgb_pred_target <- ifelse(xgb_pred > 0.5, 1, 0)
table(test_data$target, xgb_pred_target)
auc(xgb_auc)

# Logistic ROC 커브
plot(logit_auc, col = "blue", main = "ROC Curves")
legend("bottomright", legend = c("Logistic"),
		col = c("blue"), lwd = 2)


# 랜덤포레스트 ROC 커브
plot(rf_auc, col = "red", main = "ROC Curves")
legend("bottomright", legend = c("Random Forest"),
		col = c("red"), lwd = 2)

# xgboost roc 커브
plot(xgb_auc, col = "green", main = "ROC Curves")
legend("bottomright", legend = c("XGBoost"),
		col = c("green"), lwd = 2)

# 3개 같이 그리기
plot(logit_auc, col = "blue", main = "ROC Curves")
plot(rf_auc, col = "red", add = TRUE)
plot(xgb_auc, col = "green", add = TRUE)

legend("bottomright",
       legend = c("Logistic", "Random Forest", "XGBoost"),
       col = c("blue", "red", "green"), lwd = 2)







