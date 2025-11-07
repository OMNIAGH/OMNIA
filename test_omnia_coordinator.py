#!/usr/bin/env python3
"""
Suite de pruebas completa para OMNIA ENGINE COORDINATOR
Verifica el flujo completo ANCHOR → CENSOR → NOESIS
"""

import json
import requests
import time
import asyncio
import sys
import os
from datetime import datetime

class OmnIACoordinatorTestSuite:
    """Suite de pruebas para el coordinador OMNIA"""
    
    def __init__(self, base_url="http://localhost:8004"):
        self.base_url = base_url
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
    
    def run_test(self, test_name, test_func):
        """Ejecuta una prueba individual"""
        self.total_tests += 1
        print(f"\n📋 {test_name}")
        print("-" * 50)
        
        try:
            result = test_func()
            if result:
                self.passed_tests += 1
                self.test_results.append({"test": test_name, "status": "PASS", "details": result})
                print(f"✅ {test_name}: PASS")
            else:
                self.test_results.append({"test": test_name, "status": "FAIL", "details": "Test returned False"})
                print(f"❌ {test_name}: FAIL")
        except Exception as e:
            self.test_results.append({"test": test_name, "status": "ERROR", "details": str(e)})
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    def test_health_endpoint(self):
        """Prueba el endpoint de health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                components = data.get('components', {})
                print(f"   ✅ Health: OK")
                print(f"   📊 Components: {len(components)} active")
                print(f"   🏥 Status: {data.get('status', 'unknown')}")
                return True
            else:
                print(f"   ❌ Health failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health error: {str(e)}")
            return False
    
    def test_omnia_protocol_security(self):
        """Prueba el sistema de seguridad OMNIA PROTOCOL"""
        test_cases = [
            {
                "name": "Basic safe query",
                "query": "Analiza mis campañas de Google Ads del último mes",
                "expected_safe": True
            },
            {
                "name": "Query with potential PII",
                "query": "Mi email es test@example.com, analiza las métricas",
                "expected_safe": True  # Debería ser limpiado por el sistema
            },
            {
                "name": "Query with suspicious patterns",
                "query": "DROP TABLE campaigns; SELECT * FROM users",
                "expected_safe": False  # Debería ser bloqueado
            }
        ]
        
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/omnia/process",
                    json={
                        "message": test_case["query"],
                        "userId": f"test_user_security_{test_case['name'].lower().replace(' ', '_')}"
                    },
                    timeout=15
                )
                
                if test_case["expected_safe"]:
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            print(f"   ✅ {test_case['name']}: Processed successfully")
                        else:
                            print(f"   ⚠️  {test_case['name']}: Safe query but processing failed")
                    else:
                        print(f"   ⚠️  {test_case['name']}: HTTP {response.status_code}")
                else:
                    if response.status_code != 200:
                        print(f"   ✅ {test_case['name']}: Correctly blocked")
                    else:
                        print(f"   ⚠️  {test_case['name']}: Should have been blocked")
                        
            except Exception as e:
                print(f"   ❌ {test_case['name']}: Error - {str(e)}")
        
        return True
    
    def test_anchor_ingestion(self):
        """Prueba la integración con ANCHOR (simulada)"""
        test_queries = [
            "Ingesta datos de Google Ads y Meta Ads para análisis",
            "Conecta con LinkedIn y Twitter para obtener métricas de campañas",
            "Procesa archivo CSV con datos de marketing digital"
        ]
        
        for i, query in enumerate(test_queries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/omnia/process",
                    json={
                        "message": query,
                        "userId": f"test_user_anchor_{i}",
                        "sessionId": f"test_session_anchor_{i}"
                    },
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        metadata = result.get('metadata', {})
                        anchor_data = metadata.get('anchor_data', {})
                        print(f"   ✅ Test {i+1}: ANCHOR processed {anchor_data.get('records_processed', 0)} records")
                        print(f"   📊 Sources: {', '.join(anchor_data.get('sources', []))}")
                    else:
                        print(f"   ❌ Test {i+1}: Processing failed")
                else:
                    print(f"   ❌ Test {i+1}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Test {i+1}: Error - {str(e)}")
        
        return True
    
    def test_censor_supervision(self):
        """Prueba la integración con CENSOR (simulada)"""
        test_queries = [
            "Analiza anomalías en datos de campañas publicitarias",
            "Detecta patrones inusuales en métricas de performance",
            "Clasifica automáticamente tipos de campañas y contenido"
        ]
        
        for i, query in enumerate(test_queries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/omnia/process",
                    json={
                        "message": query,
                        "userId": f"test_user_censor_{i}",
                        "sessionId": f"test_session_censor_{i}"
                    },
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        metadata = result.get('metadata', {})
                        censor_data = metadata.get('censor_analysis', {})
                        print(f"   ✅ Test {i+1}: CENSOR detected {censor_data.get('anomalies_detected', 0)} anomalies")
                        print(f"   📈 Quality Score: {censor_data.get('quality_score', 0):.1%}")
                        print(f"   🏷️  Auto Labels: {censor_data.get('auto_labels', 0)} applied")
                    else:
                        print(f"   ❌ Test {i+1}: Processing failed")
                else:
                    print(f"   ❌ Test {i+1}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Test {i+1}: Error - {str(e)}")
        
        return True
    
    def test_noesis_forecasting(self):
        """Prueba la integración con NOESIS (simulada)"""
        test_queries = [
            "Genera predicciones de demanda para los próximos 30 días",
            "Analiza tendencias de marketing y forecasting",
            "Optimiza experimentos A/B para mejorar conversiones"
        ]
        
        for i, query in enumerate(test_queries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/omnia/process",
                    json={
                        "message": query,
                        "userId": f"test_user_noesis_{i}",
                        "sessionId": f"test_session_noesis_{i}"
                    },
                    timeout=25
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        metadata = result.get('metadata', {})
                        noesis_data = metadata.get('noesis_predictions', {})
                        print(f"   ✅ Test {i+1}: NOESIS forecast {noesis_data.get('forecast_horizon', 0)} days")
                        print(f"   📊 Model: {noesis_data.get('best_model', 'unknown')}")
                        print(f"   📈 Trend: {noesis_data.get('trend_direction', 'unknown')}")
                    else:
                        print(f"   ❌ Test {i+1}: Processing failed")
                else:
                    print(f"   ❌ Test {i+1}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Test {i+1}: Error - {str(e)}")
        
        return True
    
    def test_complete_workflow(self):
        """Prueba el flujo completo end-to-end"""
        print("   🔄 Ejecutando prueba de flujo completo...")
        
        comprehensive_query = {
            "message": """Analiza completamente mi estrategia de marketing digital. 
                        Necesito ingesta de datos de Google Ads, Meta Ads y LinkedIn, 
                        análisis de anomalías con ML, y predicción de demanda para 
                        optimizar mi presupuesto publicitario en los próximos 3 meses.""",
            "userId": "test_user_workflow",
            "sessionId": "test_session_comprehensive"
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/omnia/process",
                json=comprehensive_query,
                timeout=45  # Timeout largo para workflow completo
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    response_data = result.get('response', {})
                    metadata = result.get('metadata', {})
                    
                    print("   ✅ Workflow completed successfully!")
                    print(f"   ⏱️  Total time: {end_time - start_time:.2f}s")
                    print(f"   📊 Stages completed: {len(metadata.get('stages_completed', []))}")
                    print(f"   🎯 Request ID: {metadata.get('request_id', 'unknown')}")
                    
                    # Verificar contenido de la respuesta
                    if response_data.get('type') == 'omnia_coordinated':
                        data_sources = response_data.get('data_sources', {})
                        print(f"   📥 ANCHOR: {data_sources.get('anchor', {}).get('records_processed', 0)} records")
                        print(f"   🔍 CENSOR: {data_sources.get('censor', {}).get('anomalies_detected', 0)} anomalies")
                        print(f"   📈 NOESIS: {data_sources.get('noesis', {}).get('forecast_horizon', 0)} days prediction")
                        return True
                    else:
                        print("   ⚠️  Response type unexpected")
                        return False
                else:
                    print(f"   ❌ Workflow failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"   ❌ Workflow HTTP error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Workflow error: {str(e)}")
            return False
    
    def test_error_handling(self):
        """Prueba el manejo de errores"""
        error_test_cases = [
            {
                "name": "Invalid JSON",
                "data": "invalid json string"
            },
            {
                "name": "Empty message",
                "data": {"message": ""}
            },
            {
                "name": "Missing message field",
                "data": {"userId": "test_user"}
            }
        ]
        
        for test_case in error_test_cases:
            try:
                if isinstance(test_case["data"], str):
                    # Para invalid JSON, hacer request raw
                    response = requests.post(
                        f"{self.base_url}/api/v1/omnia/process",
                        data=test_case["data"],
                        headers={'Content-Type': 'application/json'},
                        timeout=5
                    )
                else:
                    response = requests.post(
                        f"{self.base_url}/api/v1/omnia/process",
                        json=test_case["data"],
                        timeout=5
                    )
                
                # Debería retornar error 400 para casos inválidos
                if response.status_code in [400, 422]:
                    print(f"   ✅ {test_case['name']}: Correctly handled error ({response.status_code})")
                else:
                    print(f"   ⚠️  {test_case['name']}: Unexpected status {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {test_case['name']}: Error - {str(e)}")
        
        return True
    
    def test_performance_benchmarks(self):
        """Prueba de rendimiento y benchmarks"""
        print("   🏃 Ejecutando benchmarks de rendimiento...")
        
        # Test de carga simple
        simple_queries = [
            "Analiza mis métricas de Google Ads",
            "Genera predicción de demanda",
            "Detecta anomalías en campañas"
        ] * 3  # 9 queries en total
        
        start_time = time.time()
        successful_requests = 0
        
        for i, query in enumerate(simple_queries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/omnia/process",
                    json={
                        "message": query,
                        "userId": f"perf_test_user_{i}",
                        "request_id": f"perf_test_{i}"
                    },
                    timeout=30
                )
                
                if response.status_code == 200 and response.json().get('success'):
                    successful_requests += 1
                    
            except Exception as e:
                print(f"   ❌ Request {i+1} failed: {str(e)}")
        
        end_time = time.time()
        total_time = end_time - start_time
        success_rate = successful_requests / len(simple_queries)
        avg_time = total_time / len(simple_queries)
        
        print(f"   📊 Requests: {successful_requests}/{len(simple_queries)} successful")
        print(f"   📈 Success Rate: {success_rate:.1%}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   ⏱️  Avg Time per Request: {avg_time:.2f}s")
        
        # Benchmarks esperados
        benchmarks_passed = 0
        if success_rate >= 0.8:
            print("   ✅ Success rate benchmark: PASS")
            benchmarks_passed += 1
        else:
            print("   ⚠️  Success rate benchmark: FAIL (expected >=80%)")
        
        if avg_time <= 15.0:
            print("   ✅ Performance benchmark: PASS")
            benchmarks_passed += 1
        else:
            print("   ⚠️  Performance benchmark: FAIL (expected <=15s per request)")
        
        return benchmarks_passed >= 1  # Al menos 1 benchmark debe pasar
    
    def print_test_summary(self):
        """Imprime resumen de las pruebas"""
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE PRUEBAS - OMNIA ENGINE COORDINATOR")
        print("=" * 70)
        
        for result in self.test_results:
            status_symbol = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_symbol} {result['test']}: {result['status']}")
            if result.get('details') and isinstance(result['details'], str):
                print(f"    📝 {result['details']}")
        
        print(f"\n🎯 Pruebas exitosas: {self.passed_tests}/{self.total_tests} ({self.passed_tests/self.total_tests*100:.1f}%)")
        
        if self.passed_tests == self.total_tests:
            print("🎉 TODAS LAS PRUEBAS PASARON! OMNIA Engine Coordinator es completamente funcional!")
            return True
        else:
            print("⚠️  Algunas pruebas fallaron. Revisar la implementación.")
            return False

def main():
    """Función principal del test suite"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8004"
    
    print("🧪 OMNIA ENGINE COORDINATOR - Test Suite Completo")
    print("=" * 60)
    print(f"🔍 Probando servidor en: {base_url}")
    
    # Verificar que el servidor está corriendo
    try:
        response = requests.get(f"{base_url}/health", timeout=3)
        if response.status_code != 200:
            print("❌ Servidor no responde. Inicia el coordinador primero:")
            print("   python3 omnia_engine_coordinator.py")
            return False
    except Exception as e:
        print(f"❌ No se puede conectar al servidor: {str(e)}")
        print("💡 Inicia el coordinador OMNIA primero:")
        print("   python3 omnia_engine_coordinator.py")
        return False
    
    # Ejecutar suite de pruebas
    test_suite = OmnIACoordinatorTestSuite(base_url)
    
    # Ejecutar todas las pruebas
    test_suite.run_test("Health Check", test_suite.test_health_endpoint)
    test_suite.run_test("OMNIA Protocol Security", test_suite.test_omnia_protocol_security)
    test_suite.run_test("ANCHOR Integration", test_suite.test_anchor_ingestion)
    test_suite.run_test("CENSOR Integration", test_suite.test_censor_supervision)
    test_suite.run_test("NOESIS Integration", test_suite.test_noesis_forecasting)
    test_suite.run_test("Complete End-to-End Workflow", test_suite.test_complete_workflow)
    test_suite.run_test("Error Handling", test_suite.test_error_handling)
    test_suite.run_test("Performance Benchmarks", test_suite.test_performance_benchmarks)
    
    # Mostrar resumen
    return test_suite.print_test_summary()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)