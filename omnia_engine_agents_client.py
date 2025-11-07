"""
Cliente de Integración de THE AGENTS con OMNIA ENGINE
Maneja la comunicación entre THE AGENTS y el sistema principal OMNIA

Autor: OMNIA Development Team
Fecha: 2025-11-07
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import uuid
import hashlib
from enum import Enum

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar tipos de THE AGENTS
from the_agents_system import (
    AgentType, RequestStatus, SecurityLevel, UserRequest, AgentResponse
)

# =============================================================================
# ENUMS Y ESTRUCTURAS DE INTEGRACIÓN
# =============================================================================

class OMNIARequestType(Enum):
    """Tipos de solicitudes a OMNIA ENGINE"""
    DATA_QUERY = "data_query"
    CAMPAIGN_STATUS = "campaign_status"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    CAMPAIGN_CREATION = "campaign_creation"
    SYSTEM_STATUS = "system_status"
    USER_MANAGEMENT = "user_management"

class OMNIAResultType(Enum):
    """Tipos de resultados de OMNIA ENGINE"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"

@dataclass
class OMNIARequest:
    """Solicitud a OMNIA ENGINE"""
    request_id: str
    request_type: OMNIARequestType
    source_agent: str
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 5
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class OMNIAResponse:
    """Respuesta de OMNIA ENGINE"""
    request_id: str
    result_type: OMNIAResultType
    timestamp: datetime
    data: Dict[str, Any]
    execution_time: float
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = None

@dataclass
class OMNIAIntegrationConfig:
    """Configuración de integración con OMNIA"""
    engine_url: str = "http://localhost:8004"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutos
    health_check_interval: int = 60  # 1 minuto
    rate_limit: int = 100  # requests per minute

# =============================================================================
# CLIENTE DE INTEGRACIÓN
# =============================================================================

class OMNIAEngineClient:
    """Cliente para comunicación con OMNIA ENGINE"""
    
    def __init__(self, config: OMNIAIntegrationConfig):
        self.config = config
        self.logger = logging.getLogger("OMNIAEngineClient")
        self.session: Optional[aiohttp.ClientSession] = None
        self.health_status = "unknown"
        self.last_health_check = None
        self.request_count = 0
        self.cache = {}
        self.rate_limiter = asyncio.Semaphore(config.rate_limit)
        
    async def __aenter__(self):
        """Context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.stop()
    
    async def start(self):
        """Inicia el cliente"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self.logger.info("OMNIA Engine Client started")
        
        # Realizar health check inicial
        await self.health_check()
    
    async def stop(self):
        """Detiene el cliente"""
        if self.session:
            await self.session.close()
        self.logger.info("OMNIA Engine Client stopped")
    
    # =============================================================================
    # HEALTH CHECK Y MONITOREO
    # =============================================================================
    
    async def health_check(self) -> bool:
        """Verifica la salud del motor OMNIA"""
        try:
            async with self.session.get(f"{self.config.engine_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    self.health_status = health_data.get("status", "unknown")
                    self.last_health_check = datetime.utcnow()
                    self.logger.info(f"OMNIA Engine health: {self.health_status}")
                    return True
                else:
                    self.health_status = "unhealthy"
                    return False
        except Exception as e:
            self.health_status = "error"
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene el estado de salud del motor"""
        return {
            "status": self.health_status,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "request_count": self.request_count,
            "cache_size": len(self.cache)
        }
    
    # =============================================================================
    # COMUNICACIÓN CON OMNIA ENGINE
    # =============================================================================
    
    async def send_request(self, request: OMNIARequest) -> OMNIAResponse:
        """Envía solicitud a OMNIA ENGINE"""
        start_time = datetime.utcnow()
        
        try:
            # Verificar rate limiting
            async with self.rate_limiter:
                # Verificar cache
                if self.config.enable_caching:
                    cache_key = self._generate_cache_key(request)
                    if cache_key in self.cache:
                        cached_response = self.cache[cache_key]
                        if self._is_cache_valid(cached_response):
                            self.logger.info(f"Returning cached response for {request.request_id}")
                            return cached_response
                
                # Realizar solicitud
                response = await self._make_request(request)
                
                # Actualizar métricas
                self.request_count += 1
                
                # Cachear respuesta si es exitoso
                if (self.config.enable_caching and 
                    response.result_type == OMNIAResultType.SUCCESS):
                    self._cache_response(cache_key, response)
                
                return response
                
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return OMNIAResponse(
                request_id=request.request_id,
                result_type=OMNIAResultType.ERROR,
                timestamp=datetime.utcnow(),
                data={},
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                error_code="REQUEST_FAILED",
                error_message=str(e)
            )
    
    async def _make_request(self, request: OMNIARequest) -> OMNIAResponse:
        """Realiza la solicitud HTTP a OMNIA ENGINE"""
        url = f"{self.config.engine_url}/agents/integrate"
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request.request_id,
            "X-Source-Agent": request.source_agent,
            "X-Request-Type": request.request_type.value
        }
        
        payload = {
            "request_id": request.request_id,
            "request_type": request.request_type.value,
            "source_agent": request.source_agent,
            "user_id": request.user_id,
            "timestamp": request.timestamp.isoformat(),
            "data": request.data,
            "priority": request.priority
        }
        
        # Intentar con reintentos
        for attempt in range(request.max_retries + 1):
            try:
                async with self.session.post(url, json=payload, headers=headers) as response:
                    response_data = await response.json()
                    
                    return OMNIAResponse(
                        request_id=response_data.get("request_id", request.request_id),
                        result_type=OMNIAResultType(response_data.get("result_type", "success")),
                        timestamp=datetime.utcnow(),
                        data=response_data.get("data", {}),
                        execution_time=response_data.get("execution_time", 0.0),
                        error_code=response_data.get("error_code"),
                        error_message=response_data.get("error_message"),
                        warnings=response_data.get("warnings", [])
                    )
                    
            except Exception as e:
                if attempt < request.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise e
    
    # =============================================================================
    # MÉTODOS DE INTEGRACIÓN ESPECÍFICOS
    # =============================================================================
    
    async def query_campaign_data(self, agent_response: AgentResponse, 
                                 filters: Dict[str, Any] = None) -> OMNIAResponse:
        """Consulta datos de campañas desde OMNIA"""
        request = OMNIARequest(
            request_id=str(uuid.uuid4()),
            request_type=OMNIARequestType.DATA_QUERY,
            source_agent=agent_response.agent_type.value,
            user_id="system",  # Podría extraer del contexto
            timestamp=datetime.utcnow(),
            data={
                "query_type": "campaign_data",
                "agent_request_id": agent_response.request_id,
                "filters": filters or {},
                "confidence": agent_response.confidence
            }
        )
        
        return await self.send_request(request)
    
    async def get_campaign_status(self, campaign_ids: List[str]) -> OMNIAResponse:
        """Obtiene estado de campañas específicas"""
        request = OMNIARequest(
            request_id=str(uuid.uuid4()),
            request_type=OMNIARequestType.CAMPAIGN_STATUS,
            source_agent="the_agents",
            user_id="system",
            timestamp=datetime.utcnow(),
            data={
                "campaign_ids": campaign_ids,
                "include_metrics": True,
                "include_performance": True
            }
        )
        
        return await self.send_request(request)
    
    async def request_performance_analytics(self, query_params: Dict[str, Any]) -> OMNIAResponse:
        """Solicita análisis de performance"""
        request = OMNIARequest(
            request_id=str(uuid.uuid4()),
            request_type=OMNIARequestType.PERFORMANCE_ANALYTICS,
            source_agent="the_agents",
            user_id="system",
            timestamp=datetime.utcnow(),
            data={
                "query_params": query_params,
                "analysis_type": "comprehensive",
                "include_forecasts": True
            }
        )
        
        return await self.send_request(request)
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> OMNIAResponse:
        """Crea nueva campaña en OMNIA"""
        request = OMNIARequest(
            request_id=str(uuid.uuid4()),
            request_type=OMNIARequestType.CAMPAIGN_CREATION,
            source_agent="the_agents",
            user_id="system",
            timestamp=datetime.utcnow(),
            data={
                "campaign_data": campaign_data,
                "creation_source": "the_agents",
                "auto_optimize": True
            }
        )
        
        return await self.send_request(request)
    
    async def get_system_status(self) -> OMNIAResponse:
        """Obtiene estado completo del sistema OMNIA"""
        request = OMNIARequest(
            request_id=str(uuid.uuid4()),
            request_type=OMNIARequestType.SYSTEM_STATUS,
            source_agent="the_agents",
            user_id="system",
            timestamp=datetime.utcnow(),
            data={
                "include_modules": True,
                "include_metrics": True,
                "include_performance": True
            }
        )
        
        return await self.send_request(request)
    
    # =============================================================================
    # CACHING Y OPTIMIZACIÓN
    # =============================================================================
    
    def _generate_cache_key(self, request: OMNIARequest) -> str:
        """Genera clave de cache para la solicitud"""
        content = f"{request.request_type.value}:{json.dumps(request.data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, response: OMNIAResponse) -> bool:
        """Verifica si la respuesta en cache es válida"""
        age = (datetime.utcnow() - response.timestamp).total_seconds()
        return age < self.config.cache_ttl
    
    def _cache_response(self, cache_key: str, response: OMNIAResponse):
        """Guarda respuesta en cache"""
        self.cache[cache_key] = response
        
        # Limpiar cache si es muy grande
        if len(self.cache) > 1000:
            # Eliminar las entradas más antiguas
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
            
            # Mantener solo las últimas 500
            self.cache = dict(sorted_items[-500:])

# =============================================================================
# INTEGRACIÓN CON THE AGENTS
# =============================================================================

class OMNIAAgentsIntegration:
    """Integración de THE AGENTS con OMNIA ENGINE"""
    
    def __init__(self, config: OMNIAIntegrationConfig):
        self.config = config
        self.client: Optional[OMNIAEngineClient] = None
        self.logger = logging.getLogger("OMNIAAgentsIntegration")
        
        # Configuración de integraciones específicas
        self.agent_integrations = {
            AgentType.SEARCH: self._integrate_search_agent,
            AgentType.CRM: self._integrate_crm_agent,
            AgentType.TEXT: self._integrate_text_agent,
            AgentType.VOICE: self._integrate_voice_agent
        }
    
    async def start(self):
        """Inicia la integración"""
        self.client = OMNIAEngineClient(self.config)
        await self.client.start()
        self.logger.info("OMNIA Agents Integration started")
    
    async def stop(self):
        """Detiene la integración"""
        if self.client:
            await self.client.stop()
        self.logger.info("OMNIA Agents Integration stopped")
    
    async def process_agent_response(self, agent_response: AgentResponse) -> Dict[str, Any]:
        """Procesa respuesta de agente y la integra con OMNIA"""
        try:
            # Determinar el tipo de integración basado en el agente
            integration_handler = self.agent_integrations.get(agent_response.agent_type)
            
            if not integration_handler:
                self.logger.warning(f"No integration handler for agent type: {agent_response.agent_type}")
                return {"status": "no_integration", "data": {}}
            
            # Ejecutar integración específica
            integration_result = await integration_handler(agent_response)
            
            # Registrar resultado
            self.logger.info(
                f"Agent {agent_response.agent_type.value} integration completed: "
                f"{integration_result.get('status', 'unknown')}"
            )
            
            return integration_result
            
        except Exception as e:
            self.logger.error(f"Integration failed for agent {agent_response.agent_type.value}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "data": {}
            }
    
    async def _integrate_search_agent(self, agent_response: AgentResponse) -> Dict[str, Any]:
        """Integración específica para SearchAgent"""
        if "results" not in agent_response.content:
            return {"status": "no_data", "data": {}}
        
        # Enriquecer resultados con datos de OMNIA
        search_data = agent_response.content["results"]
        
        # Si hay resultados de campañas, enriquecer con datos de OMNIA
        campaign_results = [r for r in search_data if "campaign" in r.get("title", "").lower()]
        
        if campaign_results:
            # Obtener IDs de campañas
            campaign_ids = [r.get("id", "") for r in campaign_results if r.get("id")]
            
            if campaign_ids:
                # Consultar estado en OMNIA
                omnia_response = await self.client.get_campaign_status(campaign_ids)
                
                if omnia_response.result_type == OMNIAResultType.SUCCESS:
                    # Enriquecer resultados
                    enriched_results = self._enrich_search_results_with_omnia_data(
                        search_data, omnia_response.data
                    )
                    
                    return {
                        "status": "success",
                        "data": {
                            "original_results": search_data,
                            "enriched_results": enriched_results,
                            "omnia_data": omnia_response.data
                        }
                    }
        
        return {
            "status": "no_campaigns_found",
            "data": {"results": search_data}
        }
    
    async def _integrate_crm_agent(self, agent_response: AgentResponse) -> Dict[str, Any]:
        """Integración específica para CRMManagementAgent"""
        customer_profile = agent_response.content.get("customer_profile", {})
        
        if not customer_profile:
            return {"status": "no_customer_data", "data": {}}
        
        # Consultar datos adicionales del cliente en OMNIA
        customer_id = customer_profile.get("customer_id")
        
        if customer_id:
            # Obtener historial de campañas del cliente
            analytics_request = {
                "customer_id": customer_id,
                "include_campaigns": True,
                "include_performance": True,
                "date_range": "last_90_days"
            }
            
            omnia_response = await self.client.request_performance_analytics(analytics_request)
            
            if omnia_response.result_type == OMNIAResultType.SUCCESS:
                # Enriquecer perfil del cliente
                enriched_profile = self._enrich_customer_profile_with_omnia_data(
                    customer_profile, omnia_response.data
                )
                
                # Actualizar recomendaciones basadas en datos de OMNIA
                updated_actions = self._update_crm_recommendations(
                    agent_response.content.get("actions_suggested", []),
                    omnia_response.data
                )
                
                return {
                    "status": "success",
                    "data": {
                        "customer_profile": enriched_profile,
                        "updated_actions": updated_actions,
                        "omnia_analytics": omnia_response.data
                    }
                }
        
        return {
            "status": "no_omnia_data",
            "data": agent_response.content
        }
    
    async def _integrate_text_agent(self, agent_response: AgentResponse) -> Dict[str, Any]:
        """Integración específica para TextAgent"""
        # Para agentes de texto, verificar si necesitan datos de OMNIA
        response_text = agent_response.content.get("response", "")
        
        # Si la respuesta menciona campañas o performance, enriquecer
        if any(keyword in response_text.lower() for keyword in 
               ["campaña", "campaign", "performance", "status", "métricas"]):
            
            # Obtener estado general del sistema
            system_response = await self.client.get_system_status()
            
            if system_response.result_type == OMNIAResultType.SUCCESS:
                # Enriquecer respuesta con datos del sistema
                enriched_response = self._enrich_text_response_with_system_data(
                    response_text, system_response.data
                )
                
                return {
                    "status": "success",
                    "data": {
                        "original_response": response_text,
                        "enriched_response": enriched_response,
                        "system_data": system_response.data
                    }
                }
        
        return {
            "status": "no_enrichment_needed",
            "data": {"response": response_text}
        }
    
    async def _integrate_voice_agent(self, agent_response: AgentResponse) -> Dict[str, Any]:
        """Integración específica para VoiceAgent"""
        # Similar a TextAgent pero para respuestas de voz
        return await self._integrate_text_agent(agent_response)
    
    # =============================================================================
    # MÉTODOS AUXILIARES
    # =============================================================================
    
    def _enrich_search_results_with_omnia_data(self, search_results: List[Dict[str, Any]], 
                                             omnia_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enriquece resultados de búsqueda con datos de OMNIA"""
        enriched_results = []
        campaign_status = omnia_data.get("campaign_status", {})
        
        for result in search_results:
            result_id = result.get("id")
            enriched_result = result.copy()
            
            # Añadir datos de OMNIA si están disponibles
            if result_id and result_id in campaign_status:
                omnia_campaign_data = campaign_status[result_id]
                enriched_result["omnia_metrics"] = omnia_campaign_data
                enriched_result["last_updated"] = datetime.utcnow().isoformat()
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    def _enrich_customer_profile_with_omnia_data(self, customer_profile: Dict[str, Any], 
                                               omnia_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enriquece perfil de cliente con datos de OMNIA"""
        enriched_profile = customer_profile.copy()
        
        # Añadir métricas de performance
        performance_data = omnia_data.get("performance_metrics", {})
        if performance_data:
            enriched_profile["omnia_performance"] = performance_data
        
        # Añadir historial de campañas
        campaign_history = omnia_data.get("campaign_history", [])
        if campaign_history:
            enriched_profile["campaign_history"] = campaign_history[-10:]  # Últimas 10
        
        # Calcular nuevas métricas
        if "lifetime_value" in enriched_profile and performance_data:
            total_campaigns = len(campaign_history)
            if total_campaigns > 0:
                avg_campaign_value = enriched_profile["lifetime_value"] / total_campaigns
                enriched_profile["avg_campaign_value"] = avg_campaign_value
        
        return enriched_profile
    
    def _update_crm_recommendations(self, current_actions: List[Dict[str, Any]], 
                                  omnia_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Actualiza recomendaciones CRM con datos de OMNIA"""
        updated_actions = current_actions.copy()
        
        # Obtener performance data
        performance_data = omnia_data.get("performance_metrics", {})
        
        if performance_data:
            # Añadir acciones basadas en performance
            if performance_data.get("low_performance_campaigns", 0) > 0:
                updated_actions.append({
                    "type": "optimization",
                    "description": f"Optimizar {performance_data['low_performance_campaigns']} campañas con bajo rendimiento",
                    "priority": "high",
                    "automated": True
                })
            
            # Añadir acciones de expansión
            if performance_data.get("high_performance_campaigns", 0) > 0:
                updated_actions.append({
                    "type": "expansion",
                    "description": "Expandir campañas de alto rendimiento",
                    "priority": "medium",
                    "automated": True
                })
        
        return updated_actions
    
    def _enrich_text_response_with_system_data(self, original_response: str, 
                                              system_data: Dict[str, Any]) -> str:
        """Enriquece respuesta de texto con datos del sistema"""
        # Añadir información contextual basada en datos del sistema
        if "active_campaigns" in system_data:
            active_count = system_data["active_campaigns"]
            enriched_response = f"{original_response} Actualmente tienes {active_count} campañas activas."
        else:
            enriched_response = original_response
        
        return enriched_response

# =============================================================================
# DEMO DE INTEGRACIÓN
# =============================================================================

async def demo_omnia_agents_integration():
    """Demostración de la integración OMNIA-Agents"""
    print("🔗 OMNIA-Agents Integration - Demostración")
    print("=" * 50)
    
    # Configuración
    config = OMNIAIntegrationConfig(
        engine_url="http://localhost:8004",
        timeout=30,
        enable_caching=True
    )
    
    # Inicializar integración
    integration = OMNIAAgentsIntegration(config)
    await integration.start()
    
    try:
        print("✅ Integración inicializada")
        
        # Simular respuesta de SearchAgent
        print("\n🔍 Simulando integración de SearchAgent")
        search_response = AgentResponse(
            request_id="search_demo_123",
            agent_type=AgentType.SEARCH,
            content={
                "results": [
                    {
                        "id": "campaign_1",
                        "title": "Summer Campaign 2025",
                        "relevance": 0.95
                    },
                    {
                        "id": "campaign_2", 
                        "title": "Black Friday Campaign",
                        "relevance": 0.88
                    }
                ],
                "total_count": 2
            },
            confidence=0.92,
            execution_time=1.5,
            status=RequestStatus.COMPLETED
        )
        
        search_integration = await integration.process_agent_response(search_response)
        print(f"✅ Integración SearchAgent: {search_integration['status']}")
        
        # Simular respuesta de CRM Agent
        print("\n👤 Simulando integración de CRM Agent")
        crm_response = AgentResponse(
            request_id="crm_demo_123",
            agent_type=AgentType.CRM,
            content={
                "customer_profile": {
                    "customer_id": "customer_123",
                    "name": "Juan Pérez",
                    "lifetime_value": 5000.0
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
            execution_time=2.0,
            status=RequestStatus.COMPLETED
        )
        
        crm_integration = await integration.process_agent_response(crm_response)
        print(f"✅ Integración CRM Agent: {crm_integration['status']}")
        
        # Estado del cliente
        print("\n📊 Estado del cliente OMNIA")
        if integration.client:
            health_status = integration.client.get_health_status()
            print(f"💚 Health Status: {health_status['status']}")
            print(f"📈 Request Count: {health_status['request_count']}")
            print(f"💾 Cache Size: {health_status['cache_size']}")
        
    except Exception as e:
        print(f"❌ Error en demo: {e}")
    
    finally:
        await integration.stop()
        print("\n🎉 Demo completada")

if __name__ == "__main__":
    # Ejecutar demo
    asyncio.run(demo_omnia_agents_integration())
