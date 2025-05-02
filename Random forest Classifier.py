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