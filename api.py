# Api.py (Regenerated based on model_training.py)
from flask import Flask, request, jsonify
import joblib
import numpy as np
from scipy.fft import fft
import os

# --- Configuration ---
SCALER_FILE = 'scaler.pkl'
EXPECTED_RAW_FEATURES = 3197

# --- Feature Engineering Function (Equivalent to model_training.py) ---
def extract_features(flux_data_list):
    """Convert raw flux time-series list to engineered features"""
    features = []
    for series in flux_data_list:
        mean = np.mean(series)
        std = np.std(series)
        slope = np.polyfit(range(len(series)), series, 1)[0]
        fft_vals = np.abs(fft(series))
        dominant_freq = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
        spectral_entropy = -np.sum(fft_vals * np.log(fft_vals + 1e-12))
        features.append([mean, std, slope, dominant_freq, spectral_entropy])
    return np.array(features)

# --- Load Scaler ---
print("Loading scaler...")
try:
    if not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}")
    scaler = joblib.load(SCALER_FILE)
    print(f"Scaler ({SCALER_FILE}) loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading scaler: {e}. Please ensure 'model_training.py' was run.")
    exit()
except Exception as e:
    print(f"An error occurred during scaler loading: {e}")
    exit()

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Define Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives raw flux data and model name, engineers features, scales, predicts, and returns result."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'flux_data' not in data:
        return jsonify({"error": "Missing 'flux_data' key in JSON payload"}), 400

    if 'model_name' not in data:
        return jsonify({"error": "Missing 'model_name' key in JSON payload."}), 400

    model_name = data['model_name']
    raw_flux_values = data['flux_data']
    model_path = model_name

    # --- Input Validation ---
    if not isinstance(raw_flux_values, list):
        return jsonify({"error": "'flux_data' must be a list of numbers"}), 400

    try:
        raw_flux_array = np.array(raw_flux_values, dtype=float)
    except ValueError:
        return jsonify({"error": "Invalid numeric data found in 'flux_data'"}), 400

    if raw_flux_array.shape[0] != EXPECTED_RAW_FEATURES:
        return jsonify({"error": f"Incorrect number of raw flux features. Expected {EXPECTED_RAW_FEATURES}, got {raw_flux_array.shape[0]}"}), 400

    if np.isnan(raw_flux_array).any() or np.isinf(raw_flux_array).any():
        return jsonify({"error": "Input raw flux data contains NaN or Infinite values."}), 400

    # --- Feature Engineering & Prediction ---
    try:
        engineered_features = extract_features([raw_flux_array])
        scaled_features = scaler.transform(engineered_features)

        print(f"Loading model: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        model_type = type(model).__name__
        print(f"Model ({model_type} from {model_path}) loaded successfully.")

        prediction = model.predict(scaled_features)
        try:
            probability = model.predict_proba(scaled_features)
            confidence = float(np.max(probability))
        except AttributeError:
            confidence = 1.0 if prediction[0] == 1 else 0.0
            print("Warning: predict_proba not available for this model. Confidence estimated.")

        predicted_class = int(prediction[0])
        class_label = "Planet" if predicted_class == 1 else "No Planet"

        response = {
            'prediction': predicted_class,
            'class_label': class_label,
            'confidence': round(confidence, 4),
            'model_used': model_name,
            'engineered_features_shape': engineered_features.shape
        }
        return jsonify(response), 200

    except FileNotFoundError as e:
        print(f"Error loading model file: {e}")
        return jsonify({"error": f"Error loading the selected model: {str(e)}"}), 500
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)