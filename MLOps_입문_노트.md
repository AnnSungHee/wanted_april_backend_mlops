
📝 MLOps 입문 수업용 – 코드 중심 해설 노트 (with 자세한 주석)

---

📌 1. scikit-learn으로 Decision Tree 모델 학습 & 저장

```python
from sklearn.datasets import load_iris                      # Iris 샘플 데이터 로드
from sklearn.model_selection import train_test_split        # 데이터 학습/테스트 분리 함수
from sklearn.tree import DecisionTreeClassifier             # 결정 트리 모델 클래스
from sklearn.metrics import accuracy_score                  # 정확도 평가 함수
import joblib                                               # 모델 저장 및 로딩을 위한 라이브러리

iris = load_iris()             # 붓꽃 데이터셋 (딕셔너리 형태)
X = iris.data                  # 특성 데이터
y = iris.target                # 레이블 데이터

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()     # 모델 생성
model.fit(X_train, y_train)          # 모델 학습

y_pred = model.predict(X_test)       # 예측
print("정확도:", accuracy_score(y_test, y_pred))  # 정확도 출력

joblib.dump(model, "iris_model.pkl") # 모델 저장
```

---

📌 2. pandas & scikit-learn을 활용한 데이터 탐색

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

df.head()                      # 상위 5개 행 출력
df.describe()                  # 통계 요약
df["target"].value_counts()    # 클래스별 개수 확인
```

---

📌 주요 객체 / 함수 정리

| 객체 / 함수 | 역할 |
|-------------|------|
| load_iris() | 붓꽃 데이터셋 로드 |
| DecisionTreeClassifier() | 결정 트리 모델 객체 생성 |
| .fit(X, y) | 학습 수행 |
| .predict(X) | 예측 수행 |
| accuracy_score() | 정확도 계산 |
| train_test_split() | 데이터 분할 |
| joblib.dump() | 모델 저장 |
| pd.DataFrame() | 데이터프레임 생성 |
| df.head() | 데이터 미리보기 |
| df.describe() | 통계 정보 출력 |
| df["col"].value_counts() | 값 개수 세기 |
