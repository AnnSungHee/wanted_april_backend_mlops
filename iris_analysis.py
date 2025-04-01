from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로딩
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# 데이터 미리보기
print(df.head())

# 통계 정보 확인
print(df.describe())

# 시각화 (선택)
sns.pairplot(df, hue="target")
plt.show()