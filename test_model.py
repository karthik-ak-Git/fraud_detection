"""
Test script to verify the trained model
"""

import torch
from models.simple_models import HighAccuracyFraudModel
import json


def test_model():
    # Load training results
    with open('simplified_training_results.json', 'r') as f:
        results = json.load(f)

    print('🎯 Training Results:')
    print(f'   Test Accuracy: {results["test_accuracy"]:.4f}')
    print(f'   Test F1: {results["test_f1"]:.4f}')
    print(f'   Test AUC: {results["test_auc"]:.4f}')

    # Test model loading
    model = HighAccuracyFraudModel(20)
    model.load_state_dict(torch.load(
        'models/high_accuracy_model.pth', map_location='cpu'))
    model.eval()

    print('\n✅ High-accuracy model loaded successfully!')
    print(
        f'   Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Test prediction
    import numpy as np
    test_input = torch.randn(1, 20)
    with torch.no_grad():
        output = model(test_input)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

    print(f'\n🧪 Test Prediction:')
    print(
        f'   Prediction: {prediction} ({"FRAUD" if prediction == 1 else "NORMAL"})')
    print(f'   Confidence: {confidence:.4f}')

    return True


if __name__ == "__main__":
    test_model()
