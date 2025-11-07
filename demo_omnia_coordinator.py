#!/usr/bin/env python3
"""
Demostración del OMNIA ENGINE COORDINATOR
Ejemplos prácticos de uso del orquestador central
"""

import json
import requests
import time
import asyncio
import sys
from datetime import datetime

class OmnIACoordinatorDemo:
    """Demostración práctica del coordinador OMNIA"""
    
    def __init__(self, base_url="http://localhost:8004"):
        self.base_url = base_url
        self.session_id = f"demo_session_{int(time.time())}"
    
    def print_header(self, title):
        """Imprime header decorativo"""
        print("\n" + "=" * 60)
        print(f"🎯 {title}")
        print("=" * 60)
    
    def print_step(self, step, description):
        """Imprime paso del proceso"""
        print(f"\n🔄 PASO {step}: {description}")
        print("-" * 40)
    
    def print_result(self, title, data):
        """Imprime resultado de forma estructurada"""
        print(f"\n✅ {title}")
        print("-" * 30)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"📊 {key.upper()}:")
                    for sub_key, sub_value in value.items():
                        print(f"   • {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    print(f"📋 {key.upper()}: {len(value)} items")
                    for item in value[:3]:  # Mostrar solo primeros 3
                        if isinstance(item, dict):
                            print(f"   • {json.dumps(item, ensure_ascii=False)[:100]}...")
                        else:
                            print(f"   • {item}")
                else:
                    print(f"• {key}: {value}")
        else:
            print(data)
    
    def demo_basic_analysis(self):
        """Demostración 1: Análisis básico de marketing"""
        self.print_header("DEMO 1: ANÁLISIS BÁSICO DE MARKETING")
        
        query = {
            "message": "Analiza mis campañas de Google Ads y Meta Ads del último mes. Quiero ver el rendimiento y detectar cualquier anomalía en las métricas.",
            "userId": "demo_user_basic",
            "sessionId": self.session_id
        }
        
        return self._process_and_display("Análisis Básico", query, show_full_response=True)
    
    def demo_forecasting_scenario(self):
        """Demostración 2: Escenario de predicción"""
        self.print_header("DEMO 2: PREDICCIÓN Y FORECASTING")
        
        query = {
            "message": "Necesito predecir la demanda de mis productos para los próximos 30 días. Considera las tendencias estacionales, campañas activas y patrones históricos para optimizar mi inventario.",
            "userId": "demo_user_forecast",
            "sessionId": self.session_id
        }
        
        return self._process_and_display("Forecasting", query, show_full_response=True)
    
    def demo_ab_testing_scenario(self):
        """Demostración 3: Escenario A/B testing"""
        self.print_header("DEMO 3: OPTIMIZACIÓN A/B TESTING")
        
        query = {
            "message": "Estoy lanzando una nueva landing page y quiero optimizar la tasa de conversión. Diseña un experimento A/B testing con recomendaciones específicas para mejorar el performance.",
            "userId": "demo_user_abtest",
            "sessionId": self.session_id
        }
        
        return self._process_and_display("A/B Testing", query, show_full_response=True)
    
    def demo_comprehensive_analysis(self):
        """Demostración 4: Análisis integral"""
        self.print_header("DEMO 4: ANÁLISIS INTEGRAL COMPLETO")
        
        query = {
            "message": """Realiza un análisis completo de mi estrategia de marketing digital. Necesito:
                        1) Ingesta de datos de todas mis fuentes: Google Ads, Meta Ads, LinkedIn, Twitter, TikTok y Pinterest
                        2) Análisis ML profundo para detectar anomalías, clasificar campañas y etiquetar contenido automáticamente
                        3) Predicción de demanda con forecasting avanzado para los próximos 3 meses
                        4) Recomendaciones A/B testing para optimizar conversiones
                        5) Análisis de ROI y optimización de presupuesto
                        Proporciona un reporte ejecutivo completo con insights accionables.""",
            "userId": "demo_user_comprehensive",
            "sessionId": self.session_id
        }
        
        return self._process_and_display("Análisis Integral", query, show_full_response=True)
    
    def demo_security_scenarios(self):
        """Demostración 5: Escenarios de seguridad"""
        self.print_header("DEMO 5: ESCENARIOS DE SEGURIDAD OMNIA PROTOCOL")
        
        security_tests = [
            {
                "name": "Query Segura Normal",
                "query": "Analiza el rendimiento de mis campañas de Facebook Ads este trimestre",
                "expected": "Procesamiento normal"
            },
            {
                "name": "Query con Posible PII",
                "query": "Mi email es marketing@empresa.com y mi teléfono +34 666 777 888, analiza las métricas de mis campañas",
                "expected": "Limpieza automática de PII"
            },
            {
                "name": "Query con Patrones Sospechosos",
                "query": "DROP TABLE campaigns; SELECT * FROM users WHERE admin=1; -- Estas son mis métricas",
                "expected": "Bloqueo por seguridad"
            }
        ]
        
        results = []
        for test in security_tests:
            print(f"\n🔐 Probando: {test['name']}")
            print(f"📝 Query: {test['query'][:50]}...")
            print(f"🎯 Esperado: {test['expected']}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/omnia/process",
                    json={
                        "message": test['query'],
                        "userId": f"security_test_{test['name'].lower().replace(' ', '_')}",
                        "sessionId": f"security_session_{int(time.time())}"
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print("✅ Procesado exitosamente")
                        results.append("SUCCESS")
                    else:
                        print("⚠️  Request procesada pero con errores")
                        results.append("WARNING")
                else:
                    print("❌ Bloqueado por seguridad" if response.status_code == 400 else f"❌ Error HTTP {response.status_code}")
                    results.append("BLOCKED" if response.status_code == 400 else "ERROR")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results.append("ERROR")
            
            time.sleep(1)  # Pausa entre tests
        
        self.print_result("Resultados de Seguridad", {
            "tests_performed": len(security_tests),
            "security_tests": dict(zip([t['name'] for t in security_tests], results)),
            "protocol_levels": ["SHIELD", "GUARDIAN", "SENTINEL", "WATCHER"]
        })
        
        return results
    
    def demo_performance_comparison(self):
        """Demostración 6: Comparación de rendimiento"""
        self.print_header("DEMO 6: COMPARACIÓN DE RENDIMIENTO")
        
        query_types = [
            ("Simple", "Analiza mis métricas básicas"),
            ("Medium", "Genera predicción de demanda para el próximo mes"),
            ("Complex", "Realiza análisis integral de toda mi estrategia de marketing digital con predicciones A/B testing y optimización de presupuesto")
        ]
        
        performance_results = []
        
        for query_type, query_text in query_types:
            print(f"\n⚡ Probando: {query_type}")
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/omnia/process",
                    json={
                        "message": query_text,
                        "userId": f"perf_test_{query_type.lower()}",
                        "sessionId": f"perf_session_{int(time.time())}"
                    },
                    timeout=30
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    processing_time = end_time - start_time
                    
                    if result.get('success'):
                        performance_results.append({
                            "query_type": query_type,
                            "processing_time": f"{processing_time:.2f}s",
                            "status": "SUCCESS",
                            "stages_completed": len(result.get('metadata', {}).get('stages_completed', []))
                        })
                        print(f"✅ {query_type}: {processing_time:.2f}s")
                    else:
                        performance_results.append({
                            "query_type": query_type,
                            "processing_time": f"{processing_time:.2f}s",
                            "status": "FAILED",
                            "error": result.get('error', 'Unknown')
                        })
                        print(f"❌ {query_type}: Failed")
                else:
                    print(f"❌ {query_type}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {query_type}: Error - {str(e)}")
            
            time.sleep(2)  # Pausa entre queries
        
        self.print_result("Métricas de Rendimiento", {
            "total_queries": len(query_types),
            "performance_results": performance_results,
            "avg_processing_time": sum([float(r['processing_time'].replace('s', '')) for r in performance_results if r['status'] == 'SUCCESS']) / max(1, len([r for r in performance_results if r['status'] == 'SUCCESS'])),
            "system_load": "Optimizado para respuestas <15s por query"
        })
        
        return performance_results
    
    def _process_and_display(self, demo_name, query, show_full_response=False):
        """Procesa query y muestra resultado"""
        print(f"\n🚀 Iniciando: {demo_name}")
        print(f"📝 Query: {query['message'][:100]}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/omnia/process",
                json=query,
                timeout=45
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    processing_time = end_time - start_time
                    
                    print(f"⏱️  Tiempo de procesamiento: {processing_time:.2f}s")
                    
                    # Mostrar metadata resumida
                    metadata = result.get('metadata', {})
                    stages = metadata.get('stages_completed', [])
                    
                    print(f"✅ Procesamiento exitoso!")
                    print(f"📊 Etapas completadas: {len(stages)}")
                    for stage in stages:
                        print(f"   • {stage}")
                    
                    # Mostrar datos de cada módulo
                    anchor_data = metadata.get('anchor_data', {})
                    censor_data = metadata.get('censor_analysis', {})
                    noesis_data = metadata.get('noesis_predictions', {})
                    
                    print(f"\n📥 ANCHOR: {anchor_data.get('records_processed', 0)} registros de {len(anchor_data.get('sources', []))} fuentes")
                    print(f"🔍 CENSOR: {censor_data.get('anomalies_detected', 0)} anomalías, calidad {censor_data.get('quality_score', 0):.1%}")
                    print(f"📈 NOESIS: {noesis_data.get('forecast_horizon', 0)} días, tendencia {noesis_data.get('trend_direction', 'unknown')}")
                    
                    # Mostrar respuesta completa si se solicita
                    if show_full_response:
                        response_data = result.get('response', {})
                        print(f"\n📄 RESPUESTA COMPLETA:")
                        print(f"🎯 Tipo: {response_data.get('type', 'unknown')}")
                        print(f"📊 Contenido: {response_data.get('content', 'No content')[:200]}...")
                        
                        if 'insights' in response_data:
                            print(f"\n💡 INSIGHTS:")
                            for insight in response_data['insights']:
                                print(f"   • {insight}")
                        
                        if 'recommendations' in response_data:
                            print(f"\n🎯 RECOMENDACIONES:")
                            for rec in response_data['recommendations'][:2]:  # Mostrar solo primeras 2
                                if isinstance(rec, dict):
                                    print(f"   • {rec.get('test_name', rec.get('description', 'Recommendation'))}")
                    
                    return True
                else:
                    print(f"❌ Error en procesamiento: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Error HTTP: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error en demo: {str(e)}")
            return False
    
    def run_complete_demo(self):
        """Ejecuta todas las demostraciones"""
        print("🎪 DEMOSTRACIÓN COMPLETA - OMNIA ENGINE COORDINATOR")
        print("=" * 60)
        print("Esta demo muestra el ecosistema completo en acción:")
        print("📥 ANCHOR → 🔍 CENSOR → 📈 NOESIS → 🎯 ORCHESTRATION")
        print("=" * 60)
        
        demos = [
            ("Análisis Básico", self.demo_basic_analysis),
            ("Predicción y Forecasting", self.demo_forecasting_scenario),
            ("A/B Testing", self.demo_ab_testing_scenario),
            ("Análisis Integral", self.demo_comprehensive_analysis),
            ("Escenarios de Seguridad", self.demo_security_scenarios),
            ("Comparación de Rendimiento", self.demo_performance_comparison)
        ]
        
        successful_demos = 0
        total_time = 0
        
        for demo_name, demo_func in demos:
            try:
                start_demo = time.time()
                if demo_func():
                    successful_demos += 1
                    print(f"✅ {demo_name}: Completado exitosamente")
                else:
                    print(f"❌ {demo_name}: Falló")
                end_demo = time.time()
                total_time += (end_demo - start_demo)
                
                # Pausa entre demos
                print(f"\n⏸️  Pausa de 3 segundos antes del siguiente demo...")
                time.sleep(3)
                
            except Exception as e:
                print(f"❌ Error en {demo_name}: {str(e)}")
        
        # Resumen final
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE DEMOSTRACIÓN")
        print("=" * 60)
        print(f"🎯 Demos completados: {successful_demos}/{len(demos)}")
        print(f"⏱️  Tiempo total: {total_time:.1f} segundos")
        print(f"📈 Tasa de éxito: {successful_demos/len(demos)*100:.1f}%")
        
        if successful_demos == len(demos):
            print("\n🎉 TODAS LAS DEMOSTRACIONES EXITOSAS!")
            print("🚀 OMNIA Engine Coordinator está completamente operativo")
        else:
            print(f"\n⚠️  {len(demos) - successful_demos} demostraciones fallaron")
            print("🔧 Revisar configuración y dependencias")
        
        return successful_demos == len(demos)

def main():
    """Función principal de la demo"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8004"
    
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
    
    # Ejecutar demo
    demo = OmnIACoordinatorDemo(base_url)
    return demo.run_complete_demo()

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 DEMO EXITOSA' if success else '⚠️  DEMO CON ERRORES'}")
    sys.exit(0 if success else 1)