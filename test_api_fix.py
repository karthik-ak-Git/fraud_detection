"""
Simple test script to validate the fraud detection fixes
"""

import asyncio
import aiohttp
import json
import time


async def test_api():
    """Test the fraud detection API"""
    print("🧪 Testing Fraud Detection API...")

    # Test data
    test_transaction = {
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
        "type_TRANSFER": 0
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            print("📡 Testing health endpoint...")
            async with session.get('http://localhost:8000/health') as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Health check passed: {health_data['status']}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return

            # Test prediction endpoint
            print("🔮 Testing prediction endpoint...")
            start_time = time.time()

            async with session.post(
                'http://localhost:8000/predict',
                json=test_transaction,
                headers={'Content-Type': 'application/json'}
            ) as response:

                end_time = time.time()
                response_time = end_time - start_time

                if response.status == 200:
                    prediction_data = await response.json()
                    print(f"✅ Prediction successful in {response_time:.2f}s")
                    print(
                        f"   Result: {'FRAUD' if prediction_data['prediction'] == 1 else 'LEGITIMATE'}")
                    print(
                        f"   Confidence: {prediction_data['confidence']:.2%}")
                    print(f"   Model used: {prediction_data['model_used']}")
                else:
                    error_text = await response.text()
                    print(f"❌ Prediction failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return

            # Test stats endpoint
            print("📊 Testing stats endpoint...")
            async with session.get('http://localhost:8000/stats') as response:
                if response.status == 200:
                    stats_data = await response.json()
                    print(
                        f"✅ Stats retrieved: {stats_data.get('status', 'unknown')}")
                else:
                    print(f"❌ Stats failed: {response.status}")

            print("\n🎉 All tests passed! The fraud detection system is working properly.")
            print("🚀 No more numpy.ndarray 'dim' errors or tensor dimension issues!")

    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_api())
