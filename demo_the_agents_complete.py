"""
Demo Completo de THE AGENTS Integrado con OMNIA
Demostración exhaustiva de todas las funcionalidades

Autor: OMNIA Development Team
Fecha: 2025-11-07
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import uuid

# Importar módulos
from the_agents_system import (
    AgentType, RequestStatus, SecurityLevel, UserRequest, AgentResponse,
    AgentOrchestrator, AgentsAPIServer
)
from omnia_engine_agents_client import (
    OMNIAIntegrationConfig, OMNIAEngineClient, OMNIAAgentsIntegration
)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("THE_AGENTS_DEMO")

# =============================================================================
# DEMO DATA GENERATORS
# =============================================================================

class DemoDataGenerator:
    """Generador de datos de prueba para el demo"""
    
    @staticmethod
    def generate_campaign_data() -> List[Dict[str, Any]]:
        """Genera datos de campaña de ejemplo"""
        campaigns = [
            {
                "id": "camp_001",
                "name": "Summer Sale 2025",
                "type": "search",
                "status": "active",
                "budget": 5000.0,
                "spent": 3200.5,
                "impressions": 125000,
                "clicks": 3500,
                "conversions": 125,
                "ctr": 0.028,
                "cpc": 0.91,
                "cpa": 25.60,
                "roas": 4.2,
                "quality_score": 8.5
            },
            {
                "id": "camp_002", 
                "name": "Black Friday Campaign",
                "type": "display",
                "status": "paused",
                "budget": 8000.0,
                "spent": 2100.0,
                "impressions": 85000,
                "clicks": 1200,
                "conversions": 45,
                "ctr": 0.014,
                "cpc": 1.75,
                "cpa": 46.67,
                "roas": 2.1,
                "quality_score": 7.2
            },
            {
                "id": "camp_003",
                "name": "Product Launch Video",
                "type": "video",
                "status": "active",
                "budget": 3000.0,
                "spent": 2800.0,
                "impressions": 45000,
                "clicks": 2200,
                "conversions": 88,
                "ctr": 0.049,
                "cpc": 1.27,
                "cpa": 31.82,
                "roas": 3.8,
                "quality_score": 9.1
            }
        ]
        return campaigns
    
    @staticmethod
    def generate_customer_data() -> List[Dict[str, Any]]:
        """Genera datos de clientes de ejemplo"""
        customers = [
            {
                "customer_id": "cust_001",
                "name": "Juan Pérez González",
                "email": "juan.perez@email.com",
                "phone": "+34666123456",
                "company": "TechSolutions S.L.",
                "segment": "enterprise",
                "lifetime_value": 15000.0,
                "total_campaigns": 8,
                "avg_campaign_budget": 2000.0,
                "last_interaction": "2025-11-05T14:30:00Z",
                "satisfaction_score": 8.5,
                "status": "active"
            },
            {
                "customer_id": "cust_002",
                "name": "María García López", 
                "email": "maria.garcia@email.com",
                "phone": "+34666789012",
                "company": "Digital Marketing Pro",
                "segment": "medium",
                "lifetime_value": 8500.0,
                "total_campaigns": 5,
                "avg_campaign_budget": 1200.0,
                "last_interaction": "2025-11-06T09:15:00Z",
                "satisfaction_score": 9.2,
                "status": "active"
            },
            {
                "customer_id": "cust_003",
                "name": "Carlos Rodríguez Martín",
                "email": "carlos.rodriguez@email.com", 
                "phone": "+34666345678",
                "company": "StartupXYZ",
                "segment": "small",
                "lifetime_value": 3200.0,
                "total_campaigns": 3,
                "avg_campaign_budget": 800.0,
                "last_interaction": "2025-10-28T16:45:00Z",
                "satisfaction_score": 7.8,
                "status": "churn_risk"
            }
        ]
        return customers
    
    @staticmethod
    def generate_search_queries() -> List[Dict[str, Any]]:
        """Genera consultas de búsqueda de ejemplo"""
        queries = [
            {
                "query": "campaign performance last 30 days",
                "intent": "performance_review",
                "filters": {"date_range": "last_30_days", "type": "performance"}
            },
            {
                "query": "buscar campañas de búsqueda activas",
                "intent": "campaign_search",
                "filters": {"status": "active", "type": "search"}
            },
            {
                "query": "análisis de ROI de campañas",
                "intent": "roi_analysis", 
                "filters": {"metric": "roas", "period": "last_quarter"}
            },
            {
                "query": "customer engagement campaigns",
                "intent": "customer_campaigns",
                "filters": {"segment": "engagement"}
            }
        ]
        return queries

# =============================================================================
# MOCK OMNIA ENGINE PARA DEMO
# =============================================================================

class MockOMNIAEngine:
    """Mock del motor OMNIA para demostración"""
    
    def __init__(self):
        self.campaigns = DemoDataGenerator.generate_campaign_data()
        self.customers = DemoDataGenerator.generate_customer_data()
        self.logger = logging.getLogger("MockOMNIAEngine")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del motor"""
        await asyncio.sleep(0.1)  # Simular latencia
        return {
            "status": "healthy",
            "version": "1.0.0",
            "modules": {
                "anchor": "active",
                "censor": "active", 
                "noesis": "active",
                "midas": "active",
                "protocol": "active"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def handle_agents_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitudes de agentes"""
        await asyncio.sleep(0.2)  # Simular procesamiento
        
        request_type = request_data.get("request_type")
        data = request_data.get("data", {})
        
        if request_type == "data_query":
            return self._handle_data_query(data)
        elif request_type == "campaign_status":
            return self._handle_campaign_status(data)
        elif request_type == "performance_analytics":
            return self._handle_performance_analytics(data)
        elif request_type == "system_status":
            return self._handle_system_status()
        else:
            return {
                "result_type": "error",
                "error_code": "UNKNOWN_REQUEST_TYPE",
                "error_message": f"Request type {request_type} not supported"
            }
    
    def _handle_data_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja consultas de datos"""
        query_type = data.get("query_type")
        
        if query_type == "campaign_data":
            return {
                "result_type": "success",
                "data": {
                    "campaigns": self.campaigns,
                    "total_count": len(self.campaigns),
                    "active_campaigns": len([c for c in self.campaigns if c["status"] == "active"]),
                    "total_budget": sum(c["budget"] for c in self.campaigns),
                    "total_spent": sum(c["spent"] for c in self.campaigns)
                },
                "execution_time": 0.15
            }
        
        return {"result_type": "error", "error_code": "UNKNOWN_QUERY_TYPE"}
    
    def _handle_campaign_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitudes de estado de campañas"""
        campaign_ids = data.get("campaign_ids", [])
        
        if not campaign_ids:
            # Retornar estado de todas las campañas
            campaign_status = {}
            for campaign in self.campaigns:
                campaign_status[campaign["id"]] = {
                    "status": campaign["status"],
                    "performance": {
                        "ctr": campaign["ctr"],
                        "cpc": campaign["cpc"],
                        "cpa": campaign["cpa"],
                        "roas": campaign["roas"],
                        "quality_score": campaign["quality_score"]
                    },
                    "budget": {
                        "allocated": campaign["budget"],
                        "spent": campaign["spent"],
                        "remaining": campaign["budget"] - campaign["spent"]
                    }
                }
            
            return {
                "result_type": "success",
                "data": {"campaign_status": campaign_status},
                "execution_time": 0.12
            }
        
        return {"result_type": "error", "error_code": "NO_CAMPAIGN_IDS"}
    
    def _handle_performance_analytics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitudes de análisis de performance"""
        # Generar analytics simulados
        analytics = {
            "performance_metrics": {
                "avg_ctr": 0.030,
                "avg_cpc": 1.31,
                "avg_cpa": 34.69,
                "avg_roas": 3.37,
                "total_impressions": 255000,
                "total_clicks": 6900,
                "total_conversions": 258
            },
            "trends": {
                "ctr_trend": "+5.2%",
                "cpc_trend": "-2.1%", 
                "cpa_trend": "+3.8%",
                "roas_trend": "+7.5%"
            },
            "recommendations": [
                "Aumentar presupuesto en campañas con ROAS > 4.0",
                "Pausar campaña con CTR < 1.5%",
                "Optimizar keywords con alto CPC"
            ]
        }
        
        return {
            "result_type": "success", 
            "data": analytics,
            "execution_time": 0.25
        }
    
    def _handle_system_status(self) -> Dict[str, Any]:
        """Maneja solicitudes de estado del sistema"""
        return {
            "result_type": "success",
            "data": {
                "system_health": "healthy",
                "active_campaigns": len([c for c in self.campaigns if c["status"] == "active"]),
                "total_customers": len(self.customers),
                "system_load": 0.35,
                "uptime": "99.9%",
                "modules_status": {
                    "anchor": "active",
                    "censor": "active",
                    "noesis": "active", 
                    "midas": "active",
                    "protocol": "active"
                }
            },
            "execution_time": 0.08
        }

# =============================================================================
# DEMO PRINCIPAL
# =============================================================================

class THEAGENTSDemo:
    """Demo principal de THE AGENTS"""
    
    def __init__(self):
        self.config = {
            "default_agents": {
                "voice": {"enabled": True, "timeout": 30},
                "text": {"enabled": True, "max_message_length": 4000},
                "search": {"enabled": True, "max_results": 50},
                "crm": {"enabled": True, "auto_actions": True}
            }
        }
        
        self.omnia_config = OMNIAIntegrationConfig(
            engine_url="http://localhost:8004",
            timeout=30,
            enable_caching=True
        )
        
        self.mock_engine = MockOMNIAEngine()
        self.api_server: Optional[AgentsAPIServer] = None
        self.omnia_integration: Optional[OMNIAAgentsIntegration] = None
    
    async def start(self):
        """Inicia el demo"""
        print("🤖 THE AGENTS - Demo Completo Integrado")
        print("=" * 60)
        print(f"⏰ Iniciado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        # Inicializar servicios
        await self._initialize_services()
        
        try:
            # Ejecutar demos
            await self._demo_conversational_agents()
            await self._demo_search_agent()
            await self._demo_crm_agent()
            await self._demo_omnia_integration()
            await self._demo_performance_metrics()
            await self._demo_security_validation()
            
        except Exception as e:
            print(f"❌ Error en demo: {e}")
            raise
        
        finally:
            await self._cleanup()
    
    async def _initialize_services(self):
        """Inicializa servicios"""
        print("🔧 Inicializando servicios...")
        
        # Inicializar API Server
        self.api_server = AgentsAPIServer(self.config)
        await self.api_server.orchestrator.start()
        
        # Inicializar integración con OMNIA
        self.omnia_integration = OMNIAAgentsIntegration(self.omnia_config)
        await self.omnia_integration.start()
        
        print("✅ Servicios inicializados")
    
    async def _demo_conversational_agents(self):
        """Demo de agentes conversacionales"""
        print("\n💬 DEMO: Agentes Conversacionales")
        print("-" * 40)
        
        # Demo TextAgent
        print("📝 TextAgent - Consulta de Status")
        text_requests = [
            {
                'message': '¿Cuál es el status de mis campañas?',
                'expected_intent': 'status_inquiry'
            },
            {
                'message': 'Necesito ayuda para crear una nueva campaña',
                'expected_intent': 'help_request'
            },
            {
                'message': 'Buscar información sobre performance',
                'expected_intent': 'search_request'
            }
        ]
        
        for i, request_data in enumerate(text_requests, 1):
            print(f"  {i}. Usuario: \"{request_data['message']}\"")
            
            api_request = {
                'user_id': f'demo_user_{i}',
                'agent_type': 'text',
                'content': {'message': request_data['message']},
                'context': {'language': 'es', 'channel': 'web'}
            }
            
            result = await self.api_server.handle_conversational_text(api_request)
            
            if result['success']:
                response_data = result['data']['content']
                print(f"     🤖 Agente: \"{response_data.get('response', 'N/A')[:80]}...\"")
                print(f"     🎯 Confianza: {result['data']['confidence']:.2f}")
                print(f"     🏷️  Intención: {response_data.get('intent_analysis', {}).get('intent', 'N/A')}")
            else:
                print(f"     ❌ Error: {result.get('error', 'Unknown')}")
            print()
        
        # Demo VoiceAgent
        print("🎤 VoiceAgent - Procesamiento de Voz")
        voice_request = {
            'user_id': 'voice_demo_user',
            'agent_type': 'voice',
            'content': {
                'audio_data': 'mock_audio_data_hola_como_estas',
                'transcript': 'Hola, ¿cómo está mi campaña de búsqueda?'
            },
            'context': {'language': 'es', 'format': 'voice'}
        }
        
        result = await self.api_server.handle_conversational_voice(voice_request)
        
        if result['success']:
            response_data = result['data']['content']
            print(f"     🎙️  Transcripción: \"{response_data.get('original_text', 'N/A')}\"")
            print(f"     🤖 Respuesta: \"{response_data.get('response_text', 'N/A')[:80]}...\"")
            print(f"     🔊 Audio generado: {response_data.get('response_audio', 'N/A')}")
        else:
            print(f"     ❌ Error: {result.get('error', 'Unknown')}")
        print()
    
    async def _demo_search_agent(self):
        """Demo de SearchAgent"""
        print("🔍 DEMO: SearchAgent - Búsqueda Inteligente")
        print("-" * 40)
        
        search_queries = DemoDataGenerator.generate_search_queries()
        
        for i, query_data in enumerate(search_queries, 1):
            print(f"  {i}. Búsqueda: \"{query_data['query']}\"")
            print(f"     🎯 Intención: {query_data['intent']}")
            
            search_request = {
                'user_id': f'search_user_{i}',
                'agent_type': 'search',
                'content': {
                    'query': query_data['query'],
                    'filters': query_data['filters']
                }
            }
            
            # Procesar con SearchAgent
            result = await self.api_server.handle_search_request(search_request)
            
            if result['success']:
                response_data = result['data']['content']
                print(f"     📊 Resultados: {response_data.get('total_count', 0)}")
                print(f"     ⏱️  Tiempo: {response_data.get('execution_time', 0):.3f}s")
                
                # Mostrar algunos resultados
                results = response_data.get('results', [])[:2]
                for j, result_item in enumerate(results, 1):
                    print(f"       {j}. {result_item.get('title', 'N/A')}")
                    print(f"          📈 Relevancia: {result_item.get('relevance', 0):.2f}")
            else:
                print(f"     ❌ Error: {result.get('error', 'Unknown')}")
            print()
    
    async def _demo_crm_agent(self):
        """Demo de CRMManagementAgent"""
        print("👤 DEMO: CRMManagementAgent - Gestión de Clientes")
        print("-" * 40)
        
        customers = DemoDataGenerator.generate_customer_data()
        
        for i, customer in enumerate(customers, 1):
            print(f"  {i}. Cliente: {customer['name']} ({customer['company']})")
            print(f"     💰 LTV: €{customer['lifetime_value']:,.2f}")
            print(f"     📊 Campañas: {customer['total_campaigns']}")
            print(f"     ⭐ Satisfacción: {customer['satisfaction_score']}/10")
            
            crm_request = {
                'user_id': f'crm_user_{i}',
                'agent_type': 'crm',
                'content': {
                    'customer_id': customer['customer_id'],
                    'interaction_type': 'follow_up',
                    'interaction_data': {
                        'call_duration': 300,
                        'satisfaction_score': customer['satisfaction_score'],
                        'notes': f'Llamada de seguimiento para {customer["company"]}'
                    }
                }
            }
            
            # Procesar con CRM Agent
            result = await self.api_server.handle_conversational_text(crm_request)
            
            if result['success']:
                response_data = result['data']['content']
                profile = response_data.get('customer_profile', {})
                behavior = response_data.get('behavior_analysis', {})
                actions = response_data.get('actions_suggested', [])
                
                print(f"     📈 Score de Comportamiento: {behavior.get('behavior_score', 'N/A')}/10")
                print(f"     🎯 Nivel de Engagement: {behavior.get('engagement_level', 'N/A')}")
                print(f"     💡 Acciones Sugeridas: {len(actions)}")
                
                for action in actions[:2]:  # Mostrar máximo 2 acciones
                    print(f"       • {action.get('description', 'N/A')}")
            else:
                print(f"     ❌ Error: {result.get('error', 'Unknown')}")
            print()
    
    async def _demo_omnia_integration(self):
        """Demo de integración con OMNIA"""
        print("🔗 DEMO: Integración con OMNIA ENGINE")
        print("-" * 40)
        
        # Mock health check
        print("  🏥 Health Check con OMNIA ENGINE")
        health_result = await self.mock_engine.health_check()
        print(f"     💚 Estado: {health_result['status']}")
        print(f"     🔧 Módulos: {len(health_result['modules'])} activos")
        
        # Demo integración SearchAgent + OMNIA
        print("\n  🔍 Integración: SearchAgent + OMNIA")
        search_response = AgentResponse(
            request_id="demo_search_001",
            agent_type=AgentType.SEARCH,
            content={
                "results": [
                    {
                        "id": "camp_001",
                        "title": "Summer Sale 2025",
                        "relevance": 0.95,
                        "source": "search_agent"
                    }
                ],
                "total_count": 1
            },
            confidence=0.92,
            execution_time=1.2,
            status=RequestStatus.COMPLETED
        )
        
        # Procesar integración
        integration_result = await self.omnia_integration.process_agent_response(search_response)
        print(f"     ✅ Integración: {integration_result['status']}")
        
        if integration_result['status'] == 'success':
            data = integration_result['data']
            if 'enriched_results' in data:
                enriched = data['enriched_results'][0]
                print(f"     📊 Resultado enriquecido: {enriched.get('title', 'N/A')}")
                if 'omnia_metrics' in enriched:
                    metrics = enriched['omnia_metrics']
                    print(f"     💰 ROAS: {metrics.get('roas', 'N/A')}")
                    print(f"     ⭐ Quality Score: {metrics.get('quality_score', 'N/A')}")
        
        # Demo integración CRM + OMNIA
        print("\n  👤 Integración: CRM Agent + OMNIA")
        crm_response = AgentResponse(
            request_id="demo_crm_001",
            agent_type=AgentType.CRM,
            content={
                "customer_profile": {
                    "customer_id": "cust_001",
                    "name": "Juan Pérez González",
                    "lifetime_value": 15000.0
                },
                "actions_suggested": [
                    {
                        "type": "follow_up",
                        "description": "Llamada de seguimiento",
                        "priority": "medium"
                    }
                ]
            },
            confidence=0.90,
            execution_time=2.1,
            status=RequestStatus.COMPLETED
        )
        
        integration_result = await self.omnia_integration.process_agent_response(crm_response)
        print(f"     ✅ Integración: {integration_result['status']}")
        
        if integration_result['status'] == 'success':
            data = integration_result['data']
            if 'updated_actions' in data:
                print(f"     💡 Acciones actualizadas: {len(data['updated_actions'])}")
                for action in data['updated_actions']:
                    print(f"       • {action.get('description', 'N/A')}")
        print()
    
    async def _demo_performance_metrics(self):
        """Demo de métricas de performance"""
        print("📊 DEMO: Métricas de Performance")
        print("-" * 40)
        
        # Obtener estadísticas de agentes
        stats = self.api_server.orchestrator.get_agent_stats()
        print(f"  🤖 Total de agentes: {stats['total_agents']}")
        print(f"  📈 Agentes activos:")
        
        for agent_type, agent_stats in stats['agents'].items():
            if agent_stats['is_active']:
                success_rate = agent_stats['success_rate']
                avg_time = agent_stats['stats']['avg_response_time']
                print(f"     • {agent_type}: {success_rate:.1f}% éxito, {avg_time:.2f}s promedio")
        
        # Simular métricas de integración
        print(f"\n  🔗 Métricas de Integración:")
        if self.omnia_integration and self.omnia_integration.client:
            health_status = self.omnia_integration.client.get_health_status()
            print(f"     • Requests enviados: {health_status['request_count']}")
            print(f"     • Cache hits: {health_status['cache_size']}")
            print(f"     • Estado OMNIA: {health_status['status']}")
        
        # Métricas simuladas de negocio
        print(f"\n  💼 Métricas de Negocio (Simuladas):")
        business_metrics = {
            "conversaciones_por_hora": 245,
            "tasa_resolucion_automatica": 78.5,
            "tiempo_promedio_resolucion": "2.3 min",
            "satisfaccion_usuario": 4.6,
            "leads_generados_dia": 12,
            "campanas_optimizadas": 8
        }
        
        for metric, value in business_metrics.items():
            print(f"     • {metric.replace('_', ' ').title()}: {value}")
        print()
    
    async def _demo_security_validation(self):
        """Demo de validación de seguridad"""
        print("🛡️  DEMO: Validación de Seguridad")
        print("-" * 40)
        
        # Test solicitud normal
        print("  ✅ Solicitud Normal:")
        normal_request = {
            'user_id': 'secure_user_1',
            'agent_type': 'text',
            'content': {'message': '¿Cómo está mi campaña de performance?'},
            'context': {'language': 'es'},
            'security_level': 'medium'
        }
        
        result = await self.api_server.handle_conversational_text(normal_request)
        print(f"     Resultado: {'✅ Aprobado' if result['success'] else '❌ Rechazado'}")
        
        # Test solicitud con contenido sospechoso
        print("\n  ⚠️  Solicitud con Contenido Sospechoso:")
        suspicious_request = {
            'user_id': 'secure_user_2',
            'agent_type': 'text',
            'content': {'message': 'Mi email es test@malicious.com y mi teléfono 123-456-7890'},
            'context': {'language': 'es'},
            'security_level': 'high'
        }
        
        result = await self.api_server.handle_conversational_text(suspicious_request)
        print(f"     Resultado: {'✅ Aprobado' if result['success'] else '❌ Bloqueado por PII'}")
        print(f"     Razón: Detección de información personal")
        print()
    
    async def _cleanup(self):
        """Limpia recursos"""
        print("🧹 Limpiando recursos...")
        
        if self.api_server:
            await self.api_server.orchestrator.stop()
        
        if self.omnia_integration:
            await self.omnia_integration.stop()
        
        print("✅ Demo completado")
    
    async def run_interactive_demo(self):
        """Ejecuta demo interactivo"""
        print("\n🎮 DEMO INTERACTIVO")
        print("=" * 60)
        print("Ingresa comandos para probar diferentes agentes:")
        print("  • 'texto: [mensaje]' - Probar TextAgent")
        print("  • 'voz: [mensaje]' - Probar VoiceAgent")
        print("  • 'buscar: [query]' - Probar SearchAgent")
        print("  • 'crm: [customer_id]' - Probar CRM Agent")
        print("  • 'status' - Ver estado del sistema")
        print("  • 'quit' - Salir")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n🤖> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'status':
                    await self._show_system_status()
                    continue
                
                # Parsear comando
                if ':' in user_input:
                    command, message = user_input.split(':', 1)
                    command = command.strip().lower()
                    message = message.strip()
                    
                    await self._process_interactive_command(command, message)
                else:
                    print("❌ Formato incorrecto. Usa 'comando: mensaje'")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 ¡Hasta luego!")
    
    async def _process_interactive_command(self, command: str, message: str):
        """Procesa comando interactivo"""
        try:
            if command == 'texto':
                request = {
                    'user_id': 'interactive_user',
                    'agent_type': 'text',
                    'content': {'message': message},
                    'context': {'language': 'es', 'channel': 'interactive'}
                }
                result = await self.api_server.handle_conversational_text(request)
                
            elif command == 'voz':
                request = {
                    'user_id': 'interactive_user',
                    'agent_type': 'voice',
                    'content': {
                        'audio_data': f'voice_data_for_{message}',
                        'transcript': message
                    },
                    'context': {'language': 'es', 'format': 'voice'}
                }
                result = await self.api_server.handle_conversational_voice(request)
                
            elif command == 'buscar':
                request = {
                    'user_id': 'interactive_user',
                    'agent_type': 'search',
                    'content': {'query': message, 'filters': {}},
                }
                result = await self.api_server.handle_search_request(request)
                
            elif command == 'crm':
                request = {
                    'user_id': 'interactive_user',
                    'agent_type': 'crm',
                    'content': {
                        'customer_id': message,
                        'interaction_type': 'interactive_query',
                        'interaction_data': {}
                    }
                }
                result = await self.api_server.handle_conversational_text(request)
                
            else:
                print(f"❌ Comando desconocido: {command}")
                return
            
            if result['success']:
                print("✅ Respuesta:")
                data = result['data']['content']
                
                if command == 'texto' or command == 'voz':
                    print(f"   {data.get('response', 'N/A')}")
                elif command == 'buscar':
                    print(f"   📊 {data.get('total_count', 0)} resultados encontrados")
                elif command == 'crm':
                    profile = data.get('customer_profile', {})
                    print(f"   👤 {profile.get('name', 'Cliente no encontrado')}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error procesando comando: {e}")
    
    async def _show_system_status(self):
        """Muestra estado del sistema"""
        try:
            status = await self.api_server.get_system_status()
            print("📊 Estado del Sistema:")
            print(f"   💚 Estado: {status['status']}")
            print(f"   🤖 Agentes: {status['agents']['total_agents']}")
            print(f"   ⏰ Timestamp: {status['timestamp']}")
        except Exception as e:
            print(f"❌ Error obteniendo status: {e}")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def main():
    """Función principal del demo"""
    demo = THEAGENTSDemo()
    
    try:
        # Ejecutar demo completo
        await demo.start()
        
        # Preguntar si quiere demo interactivo
        print("\n" + "="*60)
        response = input("¿Deseas probar el modo interactivo? (s/n): ").strip().lower()
        
        if response in ['s', 'si', 'y', 'yes']:
            await demo.run_interactive_demo()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en demo: {e}")
        raise
    finally:
        await demo._cleanup()

if __name__ == "__main__":
    print("🚀 Iniciando Demo Completo de THE AGENTS...")
    print("Este demo mostrará todas las funcionalidades integradas con OMNIA")
    print("-" * 60)
    
    # Ejecutar demo
    asyncio.run(main())
