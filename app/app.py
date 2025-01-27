from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_folder = 'E:/Git_repo/W6_Bati_Bank_Credit_Risk_Analysis'
model_path = os.path.join(model_folder, 'LR_model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Ensure input data is valid
    if not data or not 'features' in data:
        return jsonify({'error': 'No input data provided or invalid format.'}), 400

    # Convert input data to numpy array
    try:
        input_data = np.array(data['features']).reshape(1, -1)  # Reshape for a single sample
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make prediction
    try:
        prediction = model.predict(input_data)
        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()})  # Convert to list for JSON serialization
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
