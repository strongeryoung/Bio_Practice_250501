# 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score, silhouette_samples

# 데이터 불러오기
datafile = "data_org.csv"
labels_file = "labels_org.csv"

# input 데이터 읽기
df = pd.read_csv(datafile)
df = df.drop('Unnamed: 0', axis=1)  # 불필요한 열 제거
df.head()

# Label 데이터 읽기
df_labels = pd.read_csv(labels_file)
df_labels = df_labels.drop('Unnamed: 0', axis=1)
df_labels.head()

# 1. 데이터 정규화 (모든 Feature를 0과 1 사이의 값으로 변환)
scaler = MinMaxScaler()
df_nomalized = scaler.fit_transform(df)

# 2. PCA를 통해 데이터를 2차원으로 축소
pca = PCA(n_components=2, random_state=42)
df_pca = pca.fit_transform(df_nomalized)

# 최적의 클러스터 수 결정을 위한 Elbow Method를 적용
cluster_range = range(1,11)
inertia_values = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_pca)
    inertia_values.append(kmeans.inertia_)
    
# Elbow Method 결과 그래프화 하기
plt.figure(figsize=(10,6))
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()  # 최적의 클러스터

# K-means(k=5)는, elbow에서 나타난 최적의 clustering 수를 사용하여 수행
# K-means clustering 수행 (k=5로 설정)
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=50, max_iter=500, random_state=42)
cluster = kmeans.fit_predict(df_pca)

# k-means 클러스터링 결과를 데이터 프레임에 통합하기
df_result = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
df_result['Cluster'] = cluster
print(df_result)

# df_result와 df_labels를 합치기
df_result = pd.concat([df_result, df_labels], axis=1)
print(df_result)

# 클러스터링 결과 시각화
plt.style.use("default")    # 시각화 스타일 설정
plt.figure(figsize=(12,8))  # 그래프 사이즈 설정

# scatterplot 생성
scat = sns.scatterplot(
    data=df_result,  # 사용할 데이터
    x='PC1',         # x축 변수
    y='PC2',         # y축 변수
    s = 50,          # 마커의 크기
    hue='Cluster',   # 클러스터 레이블에 따라 색상 구분
    style='Class',    # 'Class'정보에 따라 마커의 스타일을 다르게 표시 (옵션)
    palette = "Set2" # 색상 팔레트 설정
)

# 클러스터 중심점을 검은색 세모로 표시
predicted_centers = kmeans.cluster_centers_
plt.scatter(predicted_centers[:, 0], predicted_centers[:, 1], c='black', marker='^', s=200, label='Centers')

# 그래프 타이틀 및 범주 설정
scat.set_title("Clustering results from TCGA Cancer-nGene Expression Data")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderdaxespad=0.0)
plt.legend(loc='best', ncol=2, fontsize=10)
plt.show()

print(df_result.head())
## 실루엣 분석
# 실루엣 분석을 위한 실루엣 점수 계산 (2차원 PCA 결과와 클러스터 정보를 이용)
df_result['Silhouette'] = silhouette_samples(df_result[['PC1', 'PC2']], df_result['Cluster'])

# 클러스터 수 계산
n_clusters = df_result['Cluster'].nunique()

# 클러스터 라벨을 정렬
sorted_unique_labels = sorted(df_result['Cluster'].unique())

# 그래프 준비
fig, ax = plt.subplots()
ax.set_xlim([-0.1, 1])  # 실루엣 점수 범위 설정
ax.set_ylim([0, len(df_result) + (n_clusters + 1) * 10])  # Y축은 클러스터별 높이 공간 확보

y_lower = 10  # 첫 클러스터 시작 위치

# 각 클러스터별로 실루엣 분포 시각화
for i in sorted_unique_labels:
    # 해당 클러스터에 속한 실루엣 점수 추출 및 정렬
    ith_cluster_silhouette_values = df_result.loc[df_result['Cluster'] == i, 'Silhouette'].sort_values()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]  # 클러스터에 속한 샘플 수
    y_upper = y_lower + size_cluster_i  # 해당 클러스터의 상단 위치

    # 색상 설정
    color = cm.nipy_spectral(float(i) / n_clusters)

    # 실루엣 점수 영역 시각화 (수평 막대 형태)
    ax.fill_betweenx(y=range(y_lower, y_upper),
                     x1=0, x2=ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)

    # 클러스터 번호 텍스트로 표시
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # 다음 클러스터를 위한 y축 시작점 업데이트
    y_lower = y_upper + 10

# 전체 평균 실루엣 점수 계산 및 기준선 추가
avg_silhouette_score = df_result['Silhouette'].mean()
ax.axvline(x=avg_silhouette_score, color="red", linestyle="--")  # 평균 기준선 (수직선)

# 축 라벨 설정
ax.set_xlabel("Silhouette Coefficient Values")
ax.set_ylabel("Cluster label")

# 제목 설정 (평균 실루엣 점수 포함)
plt.title(f"Silhouette Analysis for KMeans Clustering\\n Average Silhouette Score: {avg_silhouette_score:.2f}")

# 그래프 출력
plt.show()
