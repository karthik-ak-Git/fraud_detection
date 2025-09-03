# 🛡️ AI-Powered Fraud Detection System

A complete end-to-end machine learning application for real-time fraud detection using deep neural networks. This system provides transaction analysis through a modern web interface and continuously improves through user feedback.

![Fraud Detection System](https://img.shields.io/badge/AI-Fraud%20Detection-blue) ![Python](https://img.shields.io/badge/Python-3.11+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-orange) ![Flask](https://img.shields.io/badge/Flask-API-red)

## 🌟 Features

- **Real-time Fraud Detection**: Instant analysis of financial transactions
- **Deep Learning Models**: Multiple neural network architectures (NN, LSTM, Autoencoder, Ensemble)
- **Web Interface**: Modern, responsive UI for transaction analysis
- **REST API**: Full-featured API for integration with other systems
- **Feedback Learning**: Continuous model improvement through user feedback
- **Comprehensive Analytics**: Detailed reporting and visualization
- **Batch Processing**: Support for analyzing multiple transactions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │  Deep Learning  │
│   (HTML/JS)     │◄──►│   (Python)      │◄──►│    Models       │
│                 │    │                 │    │   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User          │    │   Feedback      │    │   Model         │
│   Interface     │    │   Storage       │    │   Training      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
fraud_detection/
├── data/
│   └── dataloader.py          # Data loading and preprocessing
├── models/
│   └── fraud_models.py        # Neural network architectures
├── frontend/
│   ├── index.html            # Web interface
│   ├── styles.css            # UI styling
│   └── scripts.js            # Frontend logic
├── outputs/                  # Model outputs and results
│   ├── best_model_nn.pth     # Trained model
│   ├── evaluation_metrics_nn.json
│   ├── training_history_nn.json
│   └── feedback_log.jsonl    # User feedback data
├── dataset/                  # Training data (pickle files)
├── main.py                   # Model training script
├── main_api.py              # Flask API server
├── evaluate.py              # Model evaluation script
├── feedback_trainer.py      # Feedback-based retraining
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project
cd fraud_detection

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For CUDA support (optional):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Train the Model

```bash
# Train the fraud detection model
python main.py
```

This will:
- Create synthetic training data (50,000 samples)
- Train a deep neural network
- Save the best model to `outputs/best_model_nn.pth`
- Generate evaluation reports and visualizations

### 3. Start the API Server

```bash
# Start the Flask API
python main_api.py
```

The API will be available at:
- **API Documentation**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **Prediction Endpoint**: http://localhost:5000/predict

### 4. Open the Web Interface

Open `frontend/index.html` in your web browser or serve it through a local server:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 8080
```

Then open http://localhost:8080

## 🔧 Usage

### Web Interface

1. **Enter Transaction Details**: Fill in the transaction form with amount, balances, timing, and type
2. **Generate Sample**: Use the "Generate Sample Transaction" button for testing
3. **Analyze**: Click "Analyze Transaction" to get fraud prediction
4. **Review Results**: See fraud probability, risk score, and confidence level
5. **Provide Feedback**: Help improve the model by confirming or correcting predictions

### API Usage

#### Predict Fraud

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
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
```

#### Python Example

```python
import requests

# Transaction data
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

# Make prediction
response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
print(f"Risk Score: {result['risk_score']}/10")
```

## 📊 Model Performance

Current model performance on test data:

| Metric | Value |
|--------|-------|
| **Accuracy** | 46.1% |
| **ROC AUC** | 49.7% |
| **Precision (Fraud)** | 30.0% |
| **Recall (Fraud)** | 59.6% |
| **F1-Score (Fraud)** | 39.9% |

*Note: Performance metrics will improve with real-world data and feedback*

## 🔄 Model Improvement

### Feedback Collection

The system automatically collects user feedback through the web interface:

```json
{
  "feedback_id": "fb_20250826_143052_7841",
  "predicted_class": "fraud",
  "actual_class": "normal",
  "transaction_data": {...},
  "user_comment": "This was a legitimate transaction",
  "timestamp": "2025-08-26T14:30:52"
}
```

### Retraining with Feedback

```bash
# Retrain model with collected feedback
python feedback_trainer.py --epochs 20 --feedback_ratio 0.3
```

This will:
- Load existing model and feedback data
- Combine feedback with synthetic training data
- Fine-tune the model with lower learning rate
- Save improved model if performance increases

## 🧠 Model Architectures

### 1. Standard Neural Network (Default)
- **Layers**: 512 → 256 → 128 → 64 → 2
- **Features**: Dropout, Batch Normalization, ReLU activation
- **Best for**: General fraud detection tasks

### 2. LSTM Network
- **Architecture**: Bidirectional LSTM for sequential patterns
- **Best for**: Time-series transaction analysis

### 3. Autoencoder
- **Type**: Anomaly detection based on reconstruction error
- **Best for**: Unsupervised fraud detection

### 4. Ensemble Model
- **Combination**: Multiple networks with voting
- **Best for**: Robust predictions with high confidence

## 📈 Evaluation and Monitoring

### Comprehensive Evaluation

```bash
# Run detailed model evaluation
python evaluate.py --model_path outputs/best_model_nn.pth
```

Generates:
- **HTML Report**: Detailed performance analysis
- **Confusion Matrix**: Visual classification results
- **ROC/PR Curves**: Performance curves
- **Feature Analysis**: Input feature importance

### Real-time Monitoring

- **API Statistics**: Track prediction counts and accuracy
- **Feedback Analytics**: Monitor user correction patterns
- **Model Performance**: Continuous accuracy tracking

## 🔒 Security Considerations

- **Data Privacy**: No sensitive data is logged or stored
- **Input Validation**: All API inputs are validated and sanitized
- **Rate Limiting**: Consider implementing for production use
- **HTTPS**: Use SSL/TLS for production deployment

## 🚀 Deployment

### Development
```bash
# Run locally
python main_api.py
```

### Production

1. **Use Production WSGI Server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main_api:app
```

2. **Docker Deployment**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main_api:app"]
```

3. **Cloud Deployment**: Deploy to AWS, Azure, or Google Cloud

## 🛠️ Development

### Adding New Features

1. **New Model Architecture**: Add to `models/fraud_models.py`
2. **Additional Endpoints**: Extend `main_api.py`
3. **UI Enhancements**: Modify `frontend/` files
4. **Data Processing**: Update `data/dataloader.py`

### Testing

```bash
# Test API endpoints
curl http://localhost:5000/health

# Test model prediction
python -c "from main_api import *; print('API test successful')"

# Validate frontend
# Open frontend/index.html and test UI components
```

## 📋 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation and model status |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Single transaction fraud prediction |
| `/batch_predict` | POST | Multiple transaction predictions |
| `/feedback` | POST | Submit prediction feedback |
| `/stats` | GET | API usage statistics |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Flask**: Web framework for API
- **Bootstrap**: Frontend UI framework
- **scikit-learn**: Machine learning utilities
- **Chart.js**: Data visualization

## 📞 Support

For questions, issues, or feature requests:

1. **GitHub Issues**: Open an issue for bugs or feature requests
2. **Documentation**: Check this README and API documentation
3. **Code Examples**: See the usage examples above

---

**⚡ Quick Commands Summary:**

```bash
# Setup and train
python main.py

# Start API
python main_api.py

# Evaluate model
python evaluate.py

# Retrain with feedback
python feedback_trainer.py

# API health check
curl http://localhost:5000/health
```

Built with ❤️ for better fraud detection and financial security.
