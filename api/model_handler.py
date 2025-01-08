from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create a Flask app
app = Flask(__name__)

# Load the pre-trained Naive Bayes model
model = joblib.load("api/naive_bayes_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input JSON data
        input_data = request.json.get("input")
        
        # Validate input
        if not input_data:
            return jsonify({"error": "Invalid input. Please provide data in the 'input' field."}), 400
        
        # Convert input data to numpy array
        input_array = np.array(input_data).astype("float")

        # Check if input is a single sample or a batch
        if len(input_array.shape) == 1:
            input_array = [input_array]

        # Predict using the loaded model
        predictions = model.predict(input_array)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask app entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
