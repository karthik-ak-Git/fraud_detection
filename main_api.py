"""
Fraud Detection API

Flask-based REST API for fraud detection predictions and feedback collection.
"""

from data.dataloader import FraudDataLoader
from models.fraud_models import get_model
import os
import sys
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import logging
from datetime import datetime
import joblib
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and data processor
model = None
model_info = None
scaler = None
feature_columns = None
device = None


def load_fraud_model(model_path: str = "outputs/best_model_nn.pth"):
    """Load the trained fraud detection model"""
    global model, model_info, scaler, feature_columns, device

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        model_info = {
            'model_type': checkpoint['model_type'],
            'model_params': checkpoint['model_params'],
            'input_size': checkpoint['input_size'],
            'feature_columns': checkpoint['feature_columns'],
            'epoch': checkpoint['epoch'],
            'val_accuracy': checkpoint['val_accuracy']
        }

        # Recreate and load model
        model = get_model(
            model_info['model_type'],
            model_info['input_size'],
            **model_info['model_params']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        feature_columns = model_info['feature_columns']

        # Initialize data loader to get scaler
        loader = FraudDataLoader()
        # Small dataset just for scaler
        df = loader.create_synthetic_data(num_samples=1000)
        loader.preprocess_data(df)
        scaler = loader.scaler

        logger.info(f"Model loaded successfully: {model_info['model_type']}")
        logger.info(f"Feature columns: {feature_columns}")

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def preprocess_transaction(transaction_data: Dict) -> np.ndarray:
    """Preprocess a single transaction for prediction"""
    try:
        # Create a DataFrame with the expected features
        df = pd.DataFrame([transaction_data])

        # Ensure all required features are present
        for feature in feature_columns:
            if feature not in df.columns:
                df[feature] = 0.0  # Default value for missing features

        # Select only the required features in the correct order
        df = df[feature_columns]

        # Handle missing values
        df = df.fillna(0.0)

        # Scale the features
        X = scaler.transform(df.values)

        return X[0]  # Return single sample

    except Exception as e:
        logger.error(f"Error preprocessing transaction: {e}")
        raise


@app.route('/')
def home():
    """API home page with documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 5px; margin-bottom: 30px; }
            .endpoint { background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #007bff; border-radius: 5px; }
            .method { background-color: #28a745; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .method.post { background-color: #007bff; }
            code { background-color: #e9ecef; padding: 2px 5px; border-radius: 3px; }
            pre { background-color: #e9ecef; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ Fraud Detection API</h1>
                <p>AI-powered transaction fraud detection system</p>
            </div>
            
            <div class="status success">
                <strong>✅ Model Status:</strong> {{ model_status }}
            </div>
            
            <div class="status info">
                <strong>ℹ️ Model Info:</strong> {{ model_type }} | Features: {{ feature_count }} | Accuracy: {{ accuracy }}%
            </div>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /predict</h3>
                <p><strong>Description:</strong> Predict fraud probability for a transaction</p>
                <p><strong>Request Body:</strong></p>
                <pre>{
  "amount": 1500.50,
  "oldbalanceOrg": 5000.0,
  "newbalanceOrig": 3500.0,
  "oldbalanceDest": 1000.0,
  "newbalanceDest": 2500.0,
  "hour": 14,
  "day_of_week": 2,
  "transaction_count_1h": 3,
  "transaction_count_24h": 15,
  "type_CASH_IN": 0,
  "type_CASH_OUT": 1,
  "type_DEBIT": 0,
  "type_PAYMENT": 0,
  "type_TRANSFER": 0
}</pre>
                <p><strong>Response:</strong></p>
                <pre>{
  "prediction": "fraud" | "normal",
  "fraud_probability": 0.85,
  "confidence": "high" | "medium" | "low",
  "risk_score": 8.5,
  "timestamp": "2025-08-26T10:30:45",
  "model_version": "nn_v1"
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /feedback</h3>
                <p><strong>Description:</strong> Submit feedback for model improvement</p>
                <p><strong>Request Body:</strong></p>
                <pre>{
  "transaction_id": "txn_12345",
  "predicted_class": "fraud",
  "actual_class": "normal",
  "transaction_data": { ... },
  "user_comment": "This was a legitimate transaction"
}</pre>
                <p><strong>Response:</strong></p>
                <pre>{
  "status": "success",
  "message": "Feedback recorded successfully",
  "feedback_id": "fb_67890"
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p><strong>Description:</strong> API health check</p>
                <p><strong>Response:</strong></p>
                <pre>{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-08-26T10:30:45"
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /stats</h3>
                <p><strong>Description:</strong> Get API usage statistics</p>
                <p><strong>Response:</strong></p>
                <pre>{
  "total_predictions": 1250,
  "fraud_detected": 123,
  "fraud_rate": 9.84,
  "feedback_count": 45,
  "model_accuracy": 94.5
}</pre>
            </div>
            
            <h2>Example Usage</h2>
            <pre># Using curl
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "amount": 1500.50,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 3500.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 2500.0,
    "hour": 14,
    "day_of_week": 2,
    "transaction_count_1h": 3,
    "transaction_count_24h": 15,
    "type_CASH_IN": 0,
    "type_CASH_OUT": 1,
    "type_DEBIT": 0,
    "type_PAYMENT": 0,
    "type_TRANSFER": 0
  }'

# Using Python requests
import requests

data = {
    "amount": 1500.50,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 3500.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 2500.0,
    "hour": 14,
    "day_of_week": 2,
    "transaction_count_1h": 3,
    "transaction_count_24h": 15,
    "type_CASH_IN": 0,
    "type_CASH_OUT": 1,
    "type_DEBIT": 0,
    "type_PAYMENT": 0,
    "type_TRANSFER": 0
}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())</pre>
        </div>
    </body>
    </html>
    """

    # Get model status information
    model_status = "Model Loaded Successfully" if model is not None else "Model Not Loaded"
    model_type = model_info['model_type'].upper() if model_info else "Unknown"
    feature_count = len(feature_columns) if feature_columns else 0
    accuracy = f"{model_info['val_accuracy']:.1f}" if model_info else "Unknown"

    return render_template_string(html_template,
                                  model_status=model_status,
                                  model_type=model_type,
                                  feature_count=feature_count,
                                  accuracy=accuracy)


@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'device': str(device) if device else 'unknown'
    })


@app.route('/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud for a given transaction"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Fraud detection model is not available'
            }), 500

        # Get transaction data from request
        transaction_data = request.get_json()

        if not transaction_data:
            return jsonify({
                'error': 'Invalid input',
                'message': 'No transaction data provided'
            }), 400

        # Preprocess the transaction
        X = preprocess_transaction(transaction_data)

        # Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)
            output = model(X_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            fraud_probability = probabilities[0, 1].item()

        # Determine confidence level
        if fraud_probability > 0.8 or fraud_probability < 0.2:
            confidence = "high"
        elif fraud_probability > 0.6 or fraud_probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        # Calculate risk score (0-10 scale)
        risk_score = round(fraud_probability * 10, 1)

        result = {
            'prediction': 'fraud' if prediction == 1 else 'normal',
            'fraud_probability': round(fraud_probability, 4),
            'normal_probability': round(1 - fraud_probability, 4),
            'confidence': confidence,
            'risk_score': risk_score,
            'timestamp': datetime.now().isoformat(),
            'model_version': f"{model_info['model_type']}_v1"
        }

        # Log prediction
        logger.info(
            f"Prediction made: {result['prediction']} (prob: {fraud_probability:.4f})")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for model improvement"""
    try:
        feedback_data = request.get_json()

        if not feedback_data:
            return jsonify({
                'error': 'Invalid input',
                'message': 'No feedback data provided'
            }), 400

        # Validate required fields
        required_fields = ['predicted_class',
                           'actual_class', 'transaction_data']
        for field in required_fields:
            if field not in feedback_data:
                return jsonify({
                    'error': 'Missing field',
                    'message': f'Required field "{field}" is missing'
                }), 400

        # Generate feedback ID
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

        # Add metadata
        feedback_record = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'predicted_class': feedback_data['predicted_class'],
            'actual_class': feedback_data['actual_class'],
            'transaction_data': feedback_data['transaction_data'],
            'user_comment': feedback_data.get('user_comment', ''),
            'transaction_id': feedback_data.get('transaction_id', ''),
            'model_version': f"{model_info['model_type']}_v1"
        }

        # Save feedback to file
        feedback_file = 'outputs/feedback_log.jsonl'
        os.makedirs('outputs', exist_ok=True)

        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_record) + '\\n')

        logger.info(f"Feedback recorded: {feedback_id}")

        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded successfully',
            'feedback_id': feedback_id
        })

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({
            'error': 'Feedback submission failed',
            'message': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics"""
    try:
        # Read feedback log to calculate stats
        feedback_file = 'outputs/feedback_log.jsonl'
        feedback_count = 0
        correct_predictions = 0

        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        feedback = json.loads(line)
                        feedback_count += 1
                        if feedback['predicted_class'] == feedback['actual_class']:
                            correct_predictions += 1

        # Calculate accuracy from feedback
        feedback_accuracy = (correct_predictions /
                             feedback_count * 100) if feedback_count > 0 else 0

        stats = {
            'model_type': model_info['model_type'] if model_info else 'unknown',
            'model_accuracy': round(model_info['val_accuracy'], 2) if model_info else 0,
            'feedback_accuracy': round(feedback_accuracy, 2),
            'feedback_count': feedback_count,
            'correct_predictions': correct_predictions,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({
            'error': 'Failed to get statistics',
            'message': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict fraud for multiple transactions"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Fraud detection model is not available'
            }), 500

        data = request.get_json()
        transactions = data.get('transactions', [])

        if not transactions:
            return jsonify({
                'error': 'Invalid input',
                'message': 'No transactions provided'
            }), 400

        results = []

        for i, transaction in enumerate(transactions):
            try:
                # Preprocess the transaction
                X = preprocess_transaction(transaction)

                # Make prediction
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)
                    output = model(X_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = output.argmax(dim=1).item()
                    fraud_probability = probabilities[0, 1].item()

                result = {
                    'transaction_index': i,
                    'prediction': 'fraud' if prediction == 1 else 'normal',
                    'fraud_probability': round(fraud_probability, 4),
                    'risk_score': round(fraud_probability * 10, 1)
                }

                results.append(result)

            except Exception as e:
                results.append({
                    'transaction_index': i,
                    'error': str(e)
                })

        return jsonify({
            'results': results,
            'total_transactions': len(transactions),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # Load the model on startup
    model_loaded = load_fraud_model()

    if not model_loaded:
        logger.error("Failed to load model. API may not function properly.")

    # Start the Flask app
    logger.info("Starting Fraud Detection API...")
    app.run(host='0.0.0.0', port=5000, debug=True)
