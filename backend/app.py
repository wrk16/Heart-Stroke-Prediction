from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS
# # CORS is a library that allows us to enable cross-origin resource sharing. it allows us to share resources between different origins.

# Load the trained model
model = load('stroke_prediction.joblib')

#initialize the flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
# # we have created a route called predict and the method is POST. If the method is POST, it means this API will accept the data from the user(frontend).

def predict():
    try:
        data = request.json

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        # predict the data using the model
        # model.predict() returns an array of predictions, so we are taking the first element of the array
        # and returning it as a response to the user

        print(f"prediction: {prediction}")

        return jsonify({"stroke": int(prediction)}), 200
        # jsonify() converts the response into a JSON response and this 200 is the status code.


    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def home():
    return "Welcome to the Stroke Prediction API"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)