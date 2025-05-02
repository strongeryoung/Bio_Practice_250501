from sklearn import datasets
import pandas as pd

# 유방암 데이터셋 불러오기  - 딕셔너리 형식
cancer = datasets.load_breast_cancer()

# 데이터셋 구조 확인  - 그래서 key값으로 확인
print(cancer.keys())

# 특성데이터와 타겟 데이터 확인
print("특성 이름:\n", cancer.feature_names)
print("타겟 이름:\n", cancer.target_names)

# 특성 데이터 (앞의 5개)와 타겟 데이터 확인
print("특성 데이터(Feature):\n", cancer.data[:5])
print("타겟데이터(Target):\n", cancer.target[:5])

# Pandas DataFrame 변환
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# 데이터 프레임 상위 5개 행 확인
print(df.head())