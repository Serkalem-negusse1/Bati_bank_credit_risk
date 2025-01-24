# app.py (Flask API)
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()

    # Convert the data into a DataFrame (modify accordingly)
    df = pd.DataFrame([data])
    
    # Process data (you can re-use your preprocessing steps here)
    # Example: X = preprocess_data(df)
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return the prediction as a response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
