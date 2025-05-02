# scikit-learn 라이브러리에서 train_test_split 불러오기
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 데이터 로딩
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# train/validation 데이터셋 분할
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
#     test_size=0.2,            # 검증셋 비율 20%
#     shuffle=True,             # 데이터를 무작위로 섞기
#     stratify=y,               # 클래스 비율 유지
#     random_state=42           # 재현 가능성 보장
# )

# 데이터 크기 출력
print('split 전 X 유방암 전체 데이터셋 크기 :', x.shape)
print('split 전 Y 유방암 전체 데이터셋 크기 :', y.shape)

print('split 후 x_train 데이터셋 크기 :', x_train.shape)
print('split 후 x_valid 데이터셋 크기 :', x_valid.shape)
print('split 후 y_train :', y_train.shape)
print('split 후 y_valid :', y_valid.shape)

## 데이터 준비
import numpy as np
import pandas as pd

df = pd.DataFrame({'x1' : np.arange(11), 'x2' : np.arange(11) ** 2})
print(df)

# z-score 정규화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_std = scaler.fit_transform(df)

z_df = pd.DataFrame(df_std, columns=['x1_std', 'x2_std'])
print(z_df)

## Min-Max 스케일링 (Normalizaion)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = scaler.fit_transform(df)
pd.DataFrame(df_minmax, columns=['x1_min', 'x2_min'])
print(df_minmax)