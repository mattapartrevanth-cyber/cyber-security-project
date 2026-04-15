from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_FILE = "rf_ids_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "encoders.pkl"

# Load the trained model and preprocessors
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(ENCODERS_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoders = joblib.load(ENCODERS_FILE)
    print("[+] Model and preprocessors loaded successfully.")
else:
    print("[!] Warning: Model files not found. Please run model_training.py first.")
    model, scaler, encoders = None, None, None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "message": "AI-Powered IDS Service is Active"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        # Expecting a single sample or list of samples
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
            
        # Expected features (minus 'label' and 'attack_class')
        expected_cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
        
        # Validate input
        for col in expected_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing required feature: {col}"}), 400
                
        # Preprocess input data
        for col in ['protocol_type', 'service', 'flag']:
            # Handle unknown labels in real-time gracefully
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                # If an unknown categorical value appears, map it to a default (like 0)
                df[col] = 0
                
        # Scale features
        X_scaled = scaler.transform(df[expected_cols])
        
        # Predict
        predictions = model.predict(X_scaled)
        
        results = []
        for i, pred in enumerate(predictions):
            status = "Intrusion Detected" if pred == 1 else "Normal Traffic"
            results.append({
                "sample_id": i,
                "prediction": int(pred),
                "status": status,
                "action": "Alert triggered" if pred == 1 else "Allowed"
            })
            
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[*] Starting IDS Real-time Detection Service on port 5001...")
    # Using 5001 to avoid conflicting with the vulnerability scanner mock app
    app.run(port=5001, debug=True)
