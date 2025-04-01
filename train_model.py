from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. train/test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 예측 및 성능 확인
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
print("리포트:\n", classification_report(y_test, y_pred))

# 5. 모델 저장
joblib.dump(model, 'iris_model.pkl')
print("모델이 'iris_model.pkl'로 저장되었습니다.")
