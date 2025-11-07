"""
Tests básicos para NOESIS Prediction APIs
Uso: python test_api.py
"""

import requests
import json
import time
from datetime import datetime

# Configuración
API_BASE_URL = "http://localhost:8000"
TEST_USER = {"username": "admin", "password": "admin123"}

def test_health_check():
    """Test del health check"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_authentication():
    """Test de autenticación"""
    print("🔐 Testing authentication...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json=TEST_USER,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]
            print("✅ Authentication passed")
            return token
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return None

def test_single_prediction(token):
    """Test de predicción individual"""
    print("📊 Testing single prediction...")
    try:
        request_data = {
            "type": "forecasting",
            "horizon": "short",
            "data_source": "hybrid",
            "parameters": {
                "seasonal_period": 7,
                "trend_factor": 1.0
            },
            "historical_period_days": 30,
            "confidence_level": 0.95
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predictions/single",
            json=request_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Single prediction passed")
            print(f"   Prediction ID: {data['prediction_id']}")
            print(f"   Status: {data['status']}")
            print(f"   Cached: {data['cached']}")
            return data
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
        return None

def test_batch_prediction(token):
    """Test de predicción en lote"""
    print("📦 Testing batch prediction...")
    try:
        request_data = {
            "batch_id": f"test-batch-{int(time.time())}",
            "requests": [
                {
                    "type": "forecasting",
                    "horizon": "short"
                },
                {
                    "type": "demand",
                    "horizon": "medium"
                },
                {
                    "type": "trends",
                    "horizon": "long"
                }
            ]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predictions/batch",
            json=request_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch prediction started")
            print(f"   Batch ID: {data['batch_id']}")
            print(f"   Total requests: {data['total_requests']}")
            return data
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return None

def test_webhook_registration(token):
    """Test de registro de webhook"""
    print("🔔 Testing webhook registration...")
    try:
        webhook_config = {
            "url": "https://httpbin.org/post",
            "events": ["prediction_completed"],
            "secret": "test-secret-123",
            "active": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/webhooks/register",
            json=webhook_config,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Webhook registration passed")
            print(f"   Webhook ID: {data['webhook_id']}")
            return data
        else:
            print(f"❌ Webhook registration failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Webhook registration error: {e}")
        return None

def test_metrics(token):
    """Test de métricas"""
    print("📈 Testing metrics...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Metrics retrieved")
            print(f"   Cache stats: {data['cache']}")
            print(f"   Active clients: {data['rate_limiting']['active_clients']}")
            return data
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return None

def test_different_prediction_types(token):
    """Test de diferentes tipos de predicción"""
    print("🎯 Testing different prediction types...")
    
    test_cases = [
        {
            "name": "Demand Prediction",
            "data": {
                "type": "demand",
                "horizon": "short",
                "parameters": {"product_category": "electronics"}
            }
        },
        {
            "name": "Trends Analysis",
            "data": {
                "type": "trends",
                "horizon": "medium",
                "parameters": {"analysis_window": 60}
            }
        }
    ]
    
    results = []
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predictions/single",
                json=test_case["data"],
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                results.append({"name": test_case["name"], "status": "success"})
                print(f"✅ {test_case['name']} passed")
            else:
                results.append({"name": test_case["name"], "status": "failed"})
                print(f"❌ {test_case['name']} failed")
                
        except Exception as e:
            results.append({"name": test_case["name"], "status": "error", "error": str(e)})
            print(f"❌ {test_case['name']} error: {e}")
    
    return results

def run_all_tests():
    """Ejecutar todos los tests"""
    print("🚀 Starting NOESIS Prediction API Tests")
    print("=" * 50)
    
    # 1. Health check
    if not test_health_check():
        print("❌ Tests stopped: Health check failed")
        return False
    
    print()
    
    # 2. Authentication
    token = test_authentication()
    if not token:
        print("❌ Tests stopped: Authentication failed")
        return False
    
    print()
    
    # 3. Single prediction
    test_single_prediction(token)
    print()
    
    # 4. Different prediction types
    test_different_prediction_types(token)
    print()
    
    # 5. Batch prediction
    test_batch_prediction(token)
    print()
    
    # 6. Webhook registration
    test_webhook_registration(token)
    print()
    
    # 7. Metrics
    test_metrics(token)
    print()
    
    print("🎉 All tests completed!")
    return True

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")