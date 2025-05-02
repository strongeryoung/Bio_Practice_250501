## Polynomial Linear Regression
import sys
assert sys.version_info >= (3, 7)

# numpy, Linear Regression과 Polynomial Features 불러오기
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 학습 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([4, 6, 8, 10, 12])

# 데이터를 다항식으로 변환, include_bias는 절편을 추가하는 파라미터입니다.
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(X, y) # 출력 feature의 수를 계산합니다.
X_transformed = poly.transform(X)  # X를 2차식으로 변환하는 함수입니다.

# Linear Regression 모델 학습
linear_regression = LinearRegression()
linear_regression.fit(X_transformed, y)
x_test = [[6]]
x_test_transformed = poly.transform(x_test)
pred = linear_regression.predict(x_test_transformed)

print("기울기:", linear_regression.coef_)
print("절편:", linear_regression.intercept_)
print("6에 대한 예측:", pred)

######################################################
# 당뇨병 데이터셋(Diabetes Dataset)을 사용하여,
# 혈당 예측 다항 회귀모델 다음과 같이 구출해 볼 수 있음
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

# feature names: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print("feature names:", diabetes.feature_names)
print("data:", x.shape)
# data : (442, 10)
print("target:", y.shape)  # target : (442,)

### 실습) 당뇨병 환자의 혈당 예측 Linear Regression
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
# feature names : ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print("feature names:", diabetes.feature_names)
print("data:", x.shape)
# data : (442, 10)
print("target:", y.shape)  # target : (442, )

## 추가 programming
# 데이터를 다항식으로 변환, include_bias는 절편을 추가하는 파라미터
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(x, y)  # 출력 feature의 수를 계산
x_transformed = poly.transform(x)  # X를 2차식으로 변환하는 함수

# Linear Regression 모델 학습
linear_regression = LinearRegression()
linear_regression.fit(x_transformed, y)
print('x_test[0] =', x_transformed[[0]])
x_test = x_transformed[[0]]
pred = linear_regression.predict(x_test)
print('Polynomical Features 당뇨병 환자 혈당 예측 =', pred)

# 당뇨병 데이터셋(Diabetes Dataset)을 사용하여 dataset split을 진행함
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
print("feature names:", diabetes.feature_names)
# feature names : ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print("data:", x.shape)  # data: (442, 10)
print("target:", y.shape)  # target: (442, )

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(x, y)  # 출력 feature의 수를 계산
x_transformed = poly.transform(x)  # x를 2차식으로 변환하는 함수
# trainset과 testset를 8:2 비율로 나눕니다.
train_x, test_x, train_y, test_y = train_test_split(x_transformed, y, test_size=0.2)

## Linear Regressionn Fit
linear_regression = LinearRegression()
linear_regression.fit(train_x, train_y)

# 결과 예측
from sklearn.metrics import mean_squared_error
# Predict
pred = linear_regression.predict(test_x)
# MAE 측정
mae = mean_squared_error(test_y, pred)
print("MAE:", mae)