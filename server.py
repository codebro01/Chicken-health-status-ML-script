from flask import Flask, request, jsonify
from waitress import serve
import joblib
import pandas as pd
import os





PORT = int(os.environ.get("PORT", 5000))  # default to 5000 for local dev
app = Flask(__name__)

# Load the model and encoders
model_bundle = joblib.load('./chicken_health_bundle.joblib')
model = model_bundle['model']
feather_encoder = model_bundle['feather_encoder']
comb_encoder = model_bundle['comb_encoder']
status_encoder = model_bundle['status_encoder']

print('prediction:', model_bundle)


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request (expecting feature values as input)
    data = request.json

    # Ensure you have the required fields in the data
    required_fields = ["temperature", "heartRate", "activityLevel", "appetiteLevel", 
                       "featherCondition", "combColor", "respiratoryRate", "ageInWeeks"]

    # Check if all required fields are present
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Preprocess data: Convert it into a pandas DataFrame
    input_data = {
        "temperature": [data["temperature"]],
        "heartRate": [data["heartRate"]],
        "activityLevel": [data["activityLevel"]],
        "appetiteLevel": [data["appetiteLevel"]],
        "featherCondition": [data["featherCondition"]],
        "combColor": [data["combColor"]],
        "respiratoryRate": [data["respiratoryRate"]],
        "ageInWeeks": [data["ageInWeeks"]]
    }

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame(input_data)

    # Preprocess categorical columns using the LabelEncoders
    input_df["featherCondition"] = feather_encoder.transform(input_df["featherCondition"])
    input_df["combColor"] = comb_encoder.transform(input_df["combColor"])

    # Make prediction using the model
    prediction = model.predict(input_df)
    print('prediction:', prediction)
    # Decode the prediction back to the original label
    prediction_label = status_encoder.inverse_transform(prediction)

    # Return the prediction as a response
    return jsonify({'prediction': prediction_label[0]})

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=PORT)

