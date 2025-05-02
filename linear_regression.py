# 학습 데이터 생성
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([4, 6, 8, 10, 12])

# Linear Regression 모델 학습
linear_regression = LinearRegression()
linear_regression.fit(X, y)
pred = linear_regression.predict([[6]])

print("기울기:", linear_regression.coef_)  # 기울기:[2.]
print("절편:", linear_regression.intercept_)  # 절편: 2.0
print("6에 대한 예측:", pred)  # 6에 대한 예측 : [14.]

plt.plot(X, y, 'r')
plt.show


## 혈당 예측 모델
# 당뇨병 데이터셋을 사용하여 혈당예측 회귀모델을 다음과 같이 구축할 수 있음
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
print("feature names:", diabetes.feature_names)
# feature names : ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print("data_size = ", x.shape)    # data: (442, 10)
print("target size =", y.shape)  # target: (442,)

## 데이터 셋을 학습과 테스트 셋 (8:2)으로 나누고 다음과 같이 회귀모델을 구성할 수 있음
from sklearn.model_selection import train_test_split
# trainset과 testset을 8:2 비율로 나눕니다.
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
linear_regression = LinearRegression()
linear_regression.fit(train_x, train_y)

# 결과 예측
from sklearn.metrics import mean_absolute_error

pred = linear_regression.predict(test_x)
mae = mean_absolute_error(test_y, pred)
print("MAE:", mae)

# 결과
# 테스트셋 내에서 모델의 결과와 실제 혈당간 차이를 MSE로 계산했을 때 44.62정도로 차이가 있는 것을 확인할 수 있음
#  Xgboost 등 다른 모델을 활용하면 linear regression 보다 더 실제 혈당에 더욱 가깝게 출력이 될 수 있음