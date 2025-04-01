from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 모델 로드
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return "Hello, Iris Classifier!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # JSON에서 입력 받기
    features = data.get("features")
    
    # 예측 실행
    prediction = model.predict(np.array(features).reshape(1, -1))
    
    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
