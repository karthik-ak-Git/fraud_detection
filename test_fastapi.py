"""
Test script for FastAPI Fraud Detection API

Tests all endpoints and functionality of the enhanced fraud detection system.
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://127.0.0.1:8000"


def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Models loaded: {list(data['models'].keys())}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    print("\n🔍 Testing prediction endpoint...")

    # Sample transaction data
    transaction_data = {
        "amount": 5000.0,
        "oldbalanceOrg": 15000.0,
        "newbalanceOrig": 10000.0,
        "oldbalanceDest": 2000.0,
        "newbalanceDest": 7000.0,
        "hour": 14,
        "day_of_week": 2,
        "transaction_count_1h": 3,
        "transaction_count_24h": 15,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 1,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 0,
        "merchant_risk_score": 6.5,
        "location_risk_score": 4.2,
        "mobile_transaction": 1,
        "new_device": 0
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=transaction_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful")
            print(
                f"   Prediction: {'FRAUD' if data['prediction'] == 1 else 'NORMAL'}")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Risk Score: {data['risk_score']:.1f}")
            print(f"   Model Used: {data['model_used']}")

            if data['risk_factors']:
                print(f"   Risk Factors: {list(data['risk_factors'].keys())}")

            return True
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔍 Testing batch prediction endpoint...")

    # Create multiple sample transactions
    transactions = []
    for i in range(3):
        transaction = {
            "amount": 1000.0 + i * 2000,
            "oldbalanceOrg": 10000.0 + i * 5000,
            "newbalanceOrig": 9000.0 + i * 3000,
            "oldbalanceDest": 1000.0 + i * 1000,
            "newbalanceDest": 2000.0 + i * 3000,
            "hour": 10 + i * 4,
            "day_of_week": i % 7,
            "transaction_count_1h": 1 + i,
            "transaction_count_24h": 5 + i * 5,
            "type_CASH_IN": 1 if i == 0 else 0,
            "type_CASH_OUT": 1 if i == 1 else 0,
            "type_DEBIT": 0,
            "type_PAYMENT": 1 if i == 2 else 0,
            "type_TRANSFER": 0,
            "merchant_risk_score": 3.0 + i * 2,
            "location_risk_score": 2.0 + i,
            "mobile_transaction": i % 2,
            "new_device": 0
        }
        transactions.append(transaction)

    batch_data = {"transactions": transactions}

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful")
            print(f"   Total transactions: {data['total_transactions']}")
            print(
                f"   Successful predictions: {data['successful_predictions']}")

            # Show first few results
            for i, result in enumerate(data['results'][:3]):
                if 'error' not in result:
                    pred_text = 'FRAUD' if result['prediction'] == 1 else 'NORMAL'
                    print(
                        f"   Transaction {i}: {pred_text} (confidence: {result['confidence']:.3f})")
                else:
                    print(f"   Transaction {i}: ERROR - {result['error']}")

            return True
        else:
            print(
                f"❌ Batch prediction failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False


def test_feedback_endpoint():
    """Test feedback submission endpoint"""
    print("\n🔍 Testing feedback endpoint...")

    feedback_data = {
        "transaction_id": "test_txn_001",
        "actual_label": 0,
        "prediction": 1,
        "confidence": 0.85,
        "user_notes": "False positive - legitimate transaction"
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Feedback submission successful")
            print(f"   Status: {data['status']}")
            print(f"   Message: {data['message']}")
            return True
        else:
            print(
                f"❌ Feedback submission failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Feedback error: {e}")
        return False


def test_stats_endpoint():
    """Test statistics endpoint"""
    print("\n🔍 Testing stats endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/stats")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats retrieval successful")
            print(f"   Total predictions: {data['total_predictions']}")
            print(f"   Fraud detected: {data['fraud_detected']}")
            print(f"   Fraud rate: {data['fraud_rate']:.3f}")
            print(f"   Accuracy: {data['accuracy']:.3f}")
            print(f"   Model version: {data['model_version']}")
            return True
        else:
            print(
                f"❌ Stats retrieval failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False


def test_models_endpoint():
    """Test models listing endpoint"""
    print("\n🔍 Testing models endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/models")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models listing successful")
            print(f"   Total models: {data['total_models']}")
            print(f"   Active models: {data['active_models']}")

            for model_name, model_info in data['models'].items():
                print(
                    f"   {model_name}: {model_info['type']} ({model_info['parameters']} params)")

            return True
        else:
            print(
                f"❌ Models listing failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models error: {e}")
        return False


def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting FastAPI Enhanced Fraud Detection System Tests")
    print("=" * 60)

    tests = [
        test_health_endpoint,
        test_prediction_endpoint,
        test_batch_prediction,
        test_feedback_endpoint,
        test_stats_endpoint,
        test_models_endpoint
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            results.append(False)

        time.sleep(0.5)  # Small delay between tests

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test_func.__name__}: {status}")

    print(
        f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! FastAPI system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("⏳ Waiting for FastAPI server to be ready...")
    time.sleep(2)

    success = run_all_tests()

    if success:
        print("\n✅ FastAPI Enhanced Fraud Detection System is fully operational!")
    else:
        print("\n❌ Some issues detected in the FastAPI system.")
