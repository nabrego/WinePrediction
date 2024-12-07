from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load
import numpy as np

app = Flask(__name__)
CORS(app)

model = load('../../models/random_forest_model.joblib')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = request.json
        features = [
            float(data['fixedAcidity']),
            float(data['volatileAcidity']),
            float(data['citricAcid']),
            float(data['residualSugar']),
            float(data['chlorides']),
            float(data['freeSulfurDioxide']),
            float(data['totalSulfurDioxide']),
            float(data['pH']),
            float(data['sulphates']),
            float(data['alcohol']),
        ]
        
        prediction = model.predict(np.array(features).reshape(1, -1))

        return jsonify({"prediction": prediction[0]})
    else:
        return "soemthing went wrong, Please try again."
    

if __name__ == "__main__":
    app.run(debug=True)

