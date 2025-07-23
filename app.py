from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "✅ API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[ 
            data['Sleep_Hours'],
            data['Start_Work_Hour'],
            data['Total_Work_Hours'],
            data['Meetings_Count'],
            data['Interruptions_Count'],
            data['Break_Minutes'],
            data['Task_Completion_Rate'],
            data['SocialMedia_Min'],
            data['Emails_Handled']
        ]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        return jsonify({'productivity_score': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)