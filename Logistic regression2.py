# 실습) Breast Cancer 예측 모델
# 유방암 데이터셋 불러오기
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# feature names: ['mean radius', mean texture', 'mean perimeter', 'mean area'
print("feature names:", cancer.feature_names)
print("data :", x.shape)
print("target :", y.shape)

# 모델 구성 및 학습
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 학습용/테스트용 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# 정규화 # MinMaxScaler로 데이터 전처리 (0~1스케일)
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)  # 학습데이터에 fit + transform
test_x_scaled = scaler.transform(test_x)        # 테스트 데이터는 transform만

logistic_regression = LogisticRegression(solver = "lbfgs", max_iter = 100)
logistic_regression.fit(train_x_scaled, train_y)

# 결과 예측 (prediction)
from sklearn.metrics import  accuracy_score

pred = logistic_regression.predict(test_x_scaled)
acc = accuracy_score(test_y, pred)
print("Breast Cancer Accuracy:", acc)  # acc : 0.96

## 결과
# 테스트셋을 가지고 모델을 평가했을 때 정확도가 Scaler를 적용했을 때, Accuracy가 0.9xx이 나온 것을 확인
# 이후에 Random Forest등 다른 모델을 활용하면 Logistic Reression보다 더 정확한 모델을 만들 수 있음