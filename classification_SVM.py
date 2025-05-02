## Support Vector Machine 실습
# numpy와 SVC 불러오기
import numpy as np
from sklearn.svm import SVC

# 학습 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

# Linear Regression 모델 학습
svc = SVC()
svc.fit(X, y)
pred = svc.predict([[6]])
print("6에 대한 예측:", pred)  # [1]

## 예시 코드에서는 numpy와 scikit learn에서 svc를 import하고, numpy의 array를 사용하여 측정하고자 하는
## 독립변수를 X, 종속변수를 y로 설정함.
# SVC()함수를 통해 모델을 선언하고, svc라는 이름을 부여함
# fit()은 선언한 모델을 학습시키는 함수로 독립변수와 종속변수를 순차적으로 넣어주면 자동으로 데이터에 맞춰 fitting됨
# 학습이 완료된 모델에 6을 독립변수로 넣고, pred에 최종결과를 저장함

### 유방암 분류 모델 ###
# 유방암 데이터셋 load
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# feature names : ['mean radius' 'mean texture' 'mean perimeter' 'mean ...
print("feature names :", cancer.feature_names)
print("data :", x.shape)     # data : (569, 30)
print("target :", y.shape)   # target : (569,)

## dataset 8:2로 split을 하고, SVM 모델 데이터 학습
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

## trainset과 testset을 8:2 비율로 나눕니다.
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

# Scaler를 적용할 경우
from sklearn.preprocessing import MinMaxScaler
# MinMaxScaler로 데이터 전처리 (0~1 스케일)
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)  # 학습 데이터에 fit + transform
test_x_scaled = scaler.transform(test_x)        # 테스트 데이터는 transform만

svc_scale = SVC(max_iter=1000)  # 모델 선언
svc_scale.fit(train_x_scaled, train_y)  # 모델 학습

# accuracy score 뽑기
from sklearn.metrics import accuracy_score

pred = svc_scale.predict(test_x_scaled)          # test set 예측
acc = accuracy_score(test_y, pred)               # accuracy 수치 뽑기
print("Scaled Test Set Accuracy: ", acc)         # Accuracy 수치 확인
## 결과: 로지스틱회귀와 비교했을 때 성능이 더 좋은 것을 확인할 수 있음 - 0.98


#####################
# Scaler를 적용하지 않을 경우
svc = SVC(max_iter=1000)               # 모델 선언
svc.fit(train_x, train_y)              # 모델 학습

# accuracy score 뽑기
# from sklearn.metrics import accuracy_score
pred = svc.predict(test_x)             # test set 예측
acc = accuracy_score(test_y, pred)     # accuracy 수치 뽑기
print("Non Scaled Test Set Accuracy: ", acc)  # Accuracy 수치 확인

# 실제 acc를 측정했을 경우, 결과 값 차이를 알 수 있음 (실행할 때마다 초기값 등에 따라 수치가 상이할 수 있음)
# Scaled Test Set Accuracy:  0.9824561403508771
# Non Scaled Test Set Accuracy:  0.9298245614035088

####### Kaggle - Predicting Heart Failure: 심부전 임상 기록 데이터 세트 ######
import numpy as np
from sklearn.svm import SVC
import pandas as pd

# 사용자의 현재 디렉토리에 맞게 설정 필요
df = pd.read_csv(r'Heart Failure Clinical Records.csv')

# 칼럼명을 리스트로 저장
feature = df.columns.tolist()

# 입력 변수 X (DEATH_EVENT 제외)
X = df.drop('DEATH_EVENT', axis=1)

# 타겟 변수 Y (DEATH_EVENT 만)
Y = df['DEATH_EVENT']

print("Feature 목록:", feature)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# dataset 8:2로 split을 하고, SVM 모델 데이터 학습
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=11)

# Scaler를 적용할 경우
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# MinMaxScaler로 데이터 전처리 (0~1 스케일)
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)  # 학습 데이터에 fit + transform
test_x_scaled = scaler.transform(test_x)        # 테스트 데이터는 transform만

# SVM 모델 선언 (max_iter 지정)
svc_scale = SVC(max_iter=1000)  # 하이퍼파라미터 설정
svc_scale.fit(train_x_scaled, train_y)  # 모델 학습

# Accuracy score 뽑기
pred = svc_scale.predict(test_x_scaled)          # test set 예측
acc = accuracy_score(test_y, pred)               # accuracy 수치 계산
print("Scaled Test Set Accuracy: ", acc)         # 결과 출력