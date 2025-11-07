"""
Test Suite para THE AGENTS
Pruebas unitarias, integración y end-to-end

Autor: OMNIA Development Team
Fecha: 2025-11-07
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Importar módulos a testear
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from the_agents_system import (
    AgentType, RequestStatus, SecurityLevel, UserRequest, AgentResponse,
    BaseAgent, BaseConversationalAgent, VoiceAgent, TextAgent,
    BaseOperationalAgent, SearchAgent, CRMManagementAgent,
    AgentOrchestrator, SecurityProtocol, AgentsAPIServer
)

# =============================================================================
# FIXTURES Y HELPERS
# =============================================================================

@pytest.fixture
def sample_user_request():
    """Fixture para crear solicitud de usuario de prueba"""
    return UserRequest(
        request_id="test_123",
        user_id="user_456",
        timestamp=datetime.utcnow(),
        agent_type=AgentType.TEXT,
        content={"message": "Hola, ¿cómo estás?"},
        context={"language": "es"},
        security_level=SecurityLevel.LOW
    )

@pytest.fixture
def sample_search_request():
    """Fixture para solicitud de búsqueda de prueba"""
    return UserRequest(
        request_id="search_123",
        user_id="user_456",
        timestamp=datetime.utcnow(),
        agent_type=AgentType.SEARCH,
        content={
            "query": "campaign performance",
            "filters": {"date_range": "last_7_days"}
        },
        context={},
        security_level=SecurityLevel.LOW
    )

@pytest.fixture
def sample_crm_request():
    """Fixture para solicitud CRM de prueba"""
    return UserRequest(
        request_id="crm_123",
        user_id="user_456",
        timestamp=datetime.utcnow(),
        agent_type=AgentType.CRM,
        content={
            "customer_id": "customer_789",
            "interaction_type": "follow_up",
            "interaction_data": {"call_duration": 300}
        },
        context={},
        security_level=SecurityLevel.LOW
    )

@pytest.fixture
def agent_config():
    """Configuración de agentes para testing"""
    return {
        "voice": {"enabled": True, "timeout": 30},
        "text": {"enabled": True, "max_message_length": 4000},
        "search": {"enabled": True, "max_results": 50},
        "crm": {"enabled": True, "auto_actions": True}
    }

# =============================================================================
# TESTS PARA ESTRUCTURAS DE DATOS
# =============================================================================

class TestDataStructures:
    """Tests para estructuras de datos"""
    
    def test_user_request_creation(self):
        """Test creación de UserRequest"""
        request = UserRequest(
            request_id="123",
            user_id="user_456",
            timestamp=datetime.utcnow(),
            agent_type=AgentType.TEXT,
            content={"message": "test"},
            context={},
            security_level=SecurityLevel.MEDIUM
        )
        
        assert request.request_id == "123"
        assert request.user_id == "user_456"
        assert request.agent_type == AgentType.TEXT
        assert request.security_level == SecurityLevel.MEDIUM
        assert request.content["message"] == "test"
    
    def test_agent_response_creation(self):
        """Test creación de AgentResponse"""
        response = AgentResponse(
            request_id="123",
            agent_type=AgentType.TEXT,
            content={"result": "success"},
            confidence=0.95,
            execution_time=1.5,
            status=RequestStatus.COMPLETED
        )
        
        assert response.request_id == "123"
        assert response.agent_type == AgentType.TEXT
        assert response.confidence == 0.95
        assert response.execution_time == 1.5
        assert response.status == RequestStatus.COMPLETED
        assert response.error_message is None

# =============================================================================
# TESTS PARA AGENTES CONVERSACIONALES
# =============================================================================

class TestTextAgent:
    """Tests para TextAgent"""
    
    @pytest.fixture
    def text_agent(self):
        """Fixture para TextAgent"""
        config = {"enabled": True, "max_message_length": 4000}
        return TextAgent(config)
    
    @pytest.mark.asyncio
    async def test_process_text_message(self, text_agent, sample_user_request):
        """Test procesamiento de mensaje de texto"""
        # Cambiar contenido para text
        sample_user_request.content = {"message": "¿Cuál es el status de mis campañas?"}
        sample_user_request.agent_type = AgentType.TEXT
        
        response = await text_agent.process(sample_user_request)
        
        assert response.status == RequestStatus.COMPLETED
        assert response.confidence > 0.7
        assert "response_text" in response.content
        assert "sentiment" in response.content
        assert "entities" in response.content
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, text_agent):
        """Test análisis de sentimiento"""
        positive_message = "¡Excelente! Me encanta el rendimiento de la campaña"
        negative_message = "Esto es terrible, el rendimiento está muy malo"
        neutral_message = "La campaña está funcionando normalmente"
        
        positive_result = await text_agent._analyze_sentiment(positive_message)
        negative_result = await text_agent._analyze_sentiment(negative_message)
        neutral_result = await text_agent._analyze_sentiment(neutral_message)
        
        assert positive_result["sentiment"] == "positive"
        assert negative_result["sentiment"] == "negative"
        assert neutral_result["sentiment"] in ["neutral", "positive"]  # neutral puede ser clasificado como positive
    
    @pytest.mark.asyncio
    async def test_intent_analysis(self, text_agent):
        """Test análisis de intención"""
        status_message = "¿Cuál es el status de mis campañas?"
        search_message = "Buscar información sobre performance"
        help_message = "Necesito ayuda para configurar una campaña"
        
        status_intent = await text_agent.analyze_intent(status_message)
        search_intent = await text_agent.analyze_intent(search_message)
        help_intent = await text_agent.analyze_intent(help_message)
        
        assert status_intent["intent"] in ["status_inquiry", "general_query"]
        assert search_intent["intent"] in ["search_request", "general_query"]
        assert help_intent["intent"] in ["help_request", "general_query"]
    
    @pytest.mark.asyncio
    async def test_suggestion_generation(self, text_agent):
        """Test generación de sugerencias"""
        campaign_message = "¿Cómo está mi campaña de búsqueda?"
        suggestions = text_agent._generate_suggestions(campaign_message)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        assert any("campaña" in s.lower() for s in suggestions)

class TestVoiceAgent:
    """Tests para VoiceAgent"""
    
    @pytest.fixture
    def voice_agent(self):
        """Fixture para VoiceAgent"""
        config = {"enabled": True, "stt_provider": "google", "tts_provider": "google"}
        return VoiceAgent(config)
    
    @pytest.mark.asyncio
    async def test_process_voice_input(self, voice_agent):
        """Test procesamiento de entrada de voz"""
        voice_request = UserRequest(
            request_id="voice_123",
            user_id="user_456",
            timestamp=datetime.utcnow(),
            agent_type=AgentType.VOICE,
            content={"audio_data": "base64_audio_data"},
            context={"language": "es"},
            security_level=SecurityLevel.LOW
        )
        
        response = await voice_agent.process(voice_request)
        
        assert response.status == RequestStatus.COMPLETED
        assert response.confidence > 0.8
        assert "response_text" in response.content
        assert "response_audio" in response.content
        assert response.content["language"] == "es"

# =============================================================================
# TESTS PARA AGENTES OPERATIVOS
# =============================================================================

class TestSearchAgent:
    """Tests para SearchAgent"""
    
    @pytest.fixture
    def search_agent(self):
        """Fixture para SearchAgent"""
        config = {
            "enabled": True,
            "sources": ["internal", "external", "omnia"],
            "max_results": 50
        }
        return SearchAgent(config)
    
    @pytest.mark.asyncio
    async def test_process_search_request(self, search_agent, sample_search_request):
        """Test procesamiento de solicitud de búsqueda"""
        response = await search_agent.process(sample_search_request)
        
        assert response.status == RequestStatus.COMPLETED
        assert response.confidence > 0.7
        assert "results" in response.content
        assert "total_count" in response.content
        assert "original_query" in response.content
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, search_agent):
        """Test expansión de consultas"""
        query = "campaña status"
        expanded = await search_agent._expand_query(query)
        
        assert "campaign" in expanded.lower() or "estado" in expanded.lower()
        assert len(expanded) >= len(query)
    
    @pytest.mark.asyncio
    async def test_ranking_and_aggregation(self, search_agent):
        """Test ranking y agregación de resultados"""
        test_results = [
            {"title": "Result 1", "relevance": 0.8},
            {"title": "Result 2", "relevance": 0.95},
            {"title": "Result 3", "relevance": 0.7}
        ]
        
        ranked = await search_agent._rank_and_aggregate(test_results)
        
        # Verificar que están ordenados por relevancia
        for i in range(len(ranked) - 1):
            assert ranked[i]["relevance"] >= ranked[i + 1]["relevance"]

class TestCRMManagementAgent:
    """Tests para CRMManagementAgent"""
    
    @pytest.fixture
    def crm_agent(self):
        """Fixture para CRMManagementAgent"""
        config = {"enabled": True, "auto_actions": True, "crm_system": "internal"}
        return CRMManagementAgent(config)
    
    @pytest.mark.asyncio
    async def test_process_crm_request(self, crm_agent, sample_crm_request):
        """Test procesamiento de solicitud CRM"""
        response = await crm_agent.process(sample_crm_request)
        
        assert response.status == RequestStatus.COMPLETED
        assert "customer_profile" in response.content
        assert "behavior_analysis" in response.content
        assert "actions_suggested" in response.content
        assert "next_actions" in response.content
    
    @pytest.mark.asyncio
    async def test_customer_profile_retrieval(self, crm_agent):
        """Test recuperación de perfil de cliente"""
        profile = await crm_agent._get_customer_profile("customer_123")
        
        assert "customer_id" in profile
        assert "name" in profile
        assert "email" in profile
        assert "lifetime_value" in profile
    
    @pytest.mark.asyncio
    async def test_behavior_analysis(self, crm_agent):
        """Test análisis de comportamiento"""
        profile = {
            "last_interaction": "2025-11-05T14:30:00Z",
            "total_campaigns": 10,
            "avg_campaign_budget": 1000
        }
        interaction_data = {"type": "call", "duration": 300}
        
        analysis = await crm_agent._analyze_customer_behavior(profile, interaction_data)
        
        assert "engagement_level" in analysis
        assert "behavior_score" in analysis
        assert "recommendation_priority" in analysis

# =============================================================================
# TESTS PARA ORQUESTADOR
# =============================================================================

class TestAgentOrchestrator:
    """Tests para AgentOrchestrator"""
    
    @pytest.fixture
    def orchestrator(self, agent_config):
        """Fixture para AgentOrchestrator"""
        return AgentOrchestrator(agent_config)
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestrator):
        """Test registro de agentes"""
        assert len(orchestrator.agents) > 0
        assert AgentType.TEXT in orchestrator.agents
        assert AgentType.SEARCH in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_process_user_request_text(self, orchestrator, sample_user_request):
        """Test procesamiento de solicitud de usuario (texto)"""
        response = await orchestrator.process_user_request(sample_user_request)
        
        assert response.request_id == sample_user_request.request_id
        assert response.agent_type == sample_user_request.agent_type
        assert response.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_process_user_request_search(self, orchestrator, sample_search_request):
        """Test procesamiento de solicitud de usuario (búsqueda)"""
        response = await orchestrator.process_user_request(sample_search_request)
        
        assert response.request_id == sample_search_request.request_id
        assert response.agent_type == sample_search_request.agent_type
        assert response.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_intent_analysis(self, orchestrator):
        """Test análisis de intención"""
        text_request = UserRequest(
            request_id="123",
            user_id="user_456",
            timestamp=datetime.utcnow(),
            agent_type=AgentType.TEXT,
            content={"message": "¿Cuál es el status de mis campañas?"},
            context={},
            security_level=SecurityLevel.LOW
        )
        
        intent = await orchestrator.analyze_intent(text_request)
        
        assert "intent" in intent
        assert "confidence" in intent
        assert 0 <= intent["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_agent_selection(self, orchestrator):
        """Test selección de agentes"""
        text_intent = {"intent": "help_request", "confidence": 0.8}
        search_intent = {"intent": "search_request", "confidence": 0.9}
        
        text_agent = orchestrator.select_agent(text_intent, AgentType.TEXT)
        search_agent = orchestrator.select_agent(search_intent, AgentType.SEARCH)
        
        assert text_agent is not None
        assert search_agent is not None
        assert text_agent.agent_type == AgentType.TEXT
        assert search_agent.agent_type == AgentType.SEARCH
    
    @pytest.mark.asyncio
    async def test_session_context_update(self, orchestrator, sample_user_request):
        """Test actualización de contexto de sesión"""
        # Simular que no hay sesión existente
        session_id = sample_user_request.session_id or sample_user_request.user_id
        assert session_id not in orchestrator.active_sessions
        
        # Procesar solicitud
        response = await orchestrator.process_user_request(sample_user_request)
        
        # Verificar que se creó la sesión
        assert session_id in orchestrator.active_sessions
        session = orchestrator.active_sessions[session_id]
        assert session.user_id == sample_user_request.user_id
        assert len(session.conversation_history) > 0
    
    def test_agent_stats(self, orchestrator):
        """Test estadísticas de agentes"""
        stats = orchestrator.get_agent_stats()
        
        assert "total_agents" in stats
        assert "agents" in stats
        assert "queue_status" in stats
        assert stats["total_agents"] > 0

# =============================================================================
# TESTS PARA PROTOCOLO DE SEGURIDAD
# =============================================================================

class TestSecurityProtocol:
    """Tests para SecurityProtocol"""
    
    @pytest.fixture
    def security_protocol(self):
        """Fixture para SecurityProtocol"""
        return SecurityProtocol({})
    
    @pytest.mark.asyncio
    async def test_validate_request_clean(self, security_protocol, sample_user_request):
        """Test validación de solicitud limpia"""
        result = await security_protocol.validate_request(sample_user_request)
        
        assert isinstance(result, SecurityValidation)
        assert isinstance(result.approved, bool)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.restrictions, list)
        assert isinstance(result.audit_id, str)
    
    @pytest.mark.asyncio
    async def test_validate_request_with_pii(self, security_protocol):
        """Test validación con PII"""
        pii_request = UserRequest(
            request_id="pii_test",
            user_id="user_456",
            timestamp=datetime.utcnow(),
            agent_type=AgentType.TEXT,
            content={"message": "Mi email es test@example.com"},
            context={},
            security_level=SecurityLevel.MEDIUM
        )
        
        result = await security_protocol.validate_request(pii_request)
        
        # Debería detectar PII
        assert "potential_pii_detected" in result.restrictions
        assert not result.approved
    
    @pytest.mark.asyncio
    async def test_validate_request_malicious(self, security_protocol):
        """Test validación con contenido malicioso"""
        malicious_request = UserRequest(
            request_id="malicious_test",
            user_id="user_456",
            timestamp=datetime.utcnow(),
            agent_type=AgentType.TEXT,
            content={"message": "<script>alert('xss')</script>"},
            context={},
            security_level=SecurityLevel.HIGH
        )
        
        result = await security_protocol.validate_request(malicious_request)
        
        # Debería detectar contenido malicioso
        assert "potential_malicious_content" in result.restrictions
        assert not result.approved

# =============================================================================
# TESTS PARA API SERVER
# =============================================================================

class TestAgentsAPIServer:
    """Tests para AgentsAPIServer"""
    
    @pytest.fixture
    def api_server(self, agent_config):
        """Fixture para AgentsAPIServer"""
        return AgentsAPIServer(agent_config)
    
    @pytest.mark.asyncio
    async def test_create_user_request(self, api_server):
        """Test creación de UserRequest desde API"""
        request_data = {
            'user_id': 'user_123',
            'agent_type': 'text',
            'content': {'message': 'Hello world'},
            'context': {'language': 'es'}
        }
        
        user_request = await api_server.create_user_request(request_data)
        
        assert user_request.user_id == 'user_123'
        assert user_request.agent_type == AgentType.TEXT
        assert user_request.content['message'] == 'Hello world'
        assert user_request.context['language'] == 'es'
    
    @pytest.mark.asyncio
    async def test_handle_conversational_text(self, api_server):
        """Test manejo de solicitud de texto"""
        request_data = {
            'user_id': 'user_123',
            'agent_type': 'text',
            'content': {'message': '¿Cómo estás?'},
            'context': {'language': 'es'}
        }
        
        result = await api_server.handle_conversational_text(request_data)
        
        assert 'success' in result
        assert 'data' in result
        assert 'request_id' in result
    
    @pytest.mark.asyncio
    async def test_handle_search_request(self, api_server):
        """Test manejo de solicitud de búsqueda"""
        request_data = {
            'user_id': 'user_123',
            'agent_type': 'search',
            'content': {
                'query': 'campaign performance',
                'filters': {}
            }
        }
        
        result = await api_server.handle_search_request(request_data)
        
        assert 'success' in result
        assert 'data' in result
        assert 'request_id' in result
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, api_server):
        """Test obtención de estado del sistema"""
        status = await api_server.get_system_status()
        
        assert 'status' in status
        assert 'timestamp' in status
        assert 'agents' in status
        assert 'version' in status
        assert status['status'] == 'healthy'

# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

class TestIntegration:
    """Tests de integración end-to-end"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, agent_config):
        """Test flujo completo de conversación"""
        # Setup
        api_server = AgentsAPIServer(agent_config)
        await api_server.orchestrator.start()
        
        try:
            # Paso 1: Consulta inicial
            initial_request = {
                'user_id': 'integration_user',
                'agent_type': 'text',
                'content': {'message': 'Hola, necesito información sobre mis campañas'},
                'context': {'language': 'es', 'channel': 'web'}
            }
            
            initial_result = await api_server.handle_conversational_text(initial_request)
            assert initial_result['success']
            
            # Paso 2: Búsqueda específica
            search_request = {
                'user_id': 'integration_user',
                'agent_type': 'search',
                'content': {
                    'query': 'campaign status',
                    'filters': {'date_range': 'last_7_days'}
                },
                'session_id': initial_request.get('session_id')
            }
            
            search_result = await api_server.handle_search_request(search_request)
            assert search_result['success']
            
            # Paso 3: Consulta de seguimiento
            followup_request = {
                'user_id': 'integration_user',
                'agent_type': 'text',
                'content': {'message': '¿Puedes explicarme los resultados?'},
                'context': {'language': 'es', 'channel': 'web'},
                'session_id': initial_request.get('session_id')
            }
            
            followup_result = await api_server.handle_conversational_text(followup_request)
            assert followup_result['success']
            
        finally:
            await api_server.orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_crm_workflow(self, agent_config):
        """Test workflow de CRM"""
        # Setup
        api_server = AgentsAPIServer(agent_config)
        await api_server.orchestrator.start()
        
        try:
            crm_request = {
                'user_id': 'crm_user',
                'agent_type': 'crm',
                'content': {
                    'customer_id': 'customer_123',
                    'interaction_type': 'follow_up',
                    'interaction_data': {
                        'call_duration': 600,
                        'notes': 'Customer interested in premium features',
                        'satisfaction_score': 8
                    }
                }
            }
            
            # Procesar solicitud CRM
            result = await api_server.handle_conversational_text(crm_request)
            
            assert result['success']
            response_data = result['data']['content']
            assert 'customer_profile' in response_data
            assert 'behavior_analysis' in response_data
            assert 'actions_suggested' in response_data
            
            # Verificar que se sugirieron acciones
            actions = response_data['actions_suggested']
            assert len(actions) > 0
            
        finally:
            await api_server.orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, agent_config):
        """Test manejo de solicitudes concurrentes"""
        # Setup
        api_server = AgentsAPIServer(agent_config)
        await api_server.orchestrator.start()
        
        try:
            # Crear múltiples solicitudes concurrentes
            requests = []
            for i in range(10):
                request = {
                    'user_id': f'user_{i}',
                    'agent_type': 'text',
                    'content': {'message': f'Mensaje de prueba {i}'},
                    'context': {'language': 'es'}
                }
                requests.append(request)
            
            # Procesar concurrentemente
            tasks = [
                api_server.handle_conversational_text(req) 
                for req in requests
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verificar que todas fueron procesadas
            assert len(results) == 10
            for result in results:
                assert 'success' in result
                assert 'request_id' in result
            
        finally:
            await api_server.orchestrator.stop()

# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

class TestPerformance:
    """Tests de performance y carga"""
    
    @pytest.mark.asyncio
    async def test_response_time(self, agent_config):
        """Test tiempo de respuesta"""
        api_server = AgentsAPIServer(agent_config)
        await api_server.orchestrator.start()
        
        try:
            import time
            
            request_data = {
                'user_id': 'perf_user',
                'agent_type': 'text',
                'content': {'message': 'Performance test message'},
                'context': {'language': 'es'}
            }
            
            start_time = time.time()
            result = await api_server.handle_conversational_text(request_data)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Debería completarse en menos de 2 segundos
            assert execution_time < 2.0
            assert result['success']
            
        finally:
            await api_server.orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_throughput(self, agent_config):
        """Test throughput del sistema"""
        api_server = AgentsAPIServer(agent_config)
        await api_server.orchestrator.start()
        
        try:
            import time
            
            # Medir throughput (requests por segundo)
            start_time = time.time()
            request_count = 50
            
            requests = [
                {
                    'user_id': f'throughput_user_{i}',
                    'agent_type': 'text',
                    'content': {'message': f'Throughput test {i}'},
                    'context': {'language': 'es'}
                }
                for i in range(request_count)
            ]
            
            tasks = [
                api_server.handle_conversational_text(req) 
                for req in requests
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = request_count / total_time
            
            # Debería procesar al menos 10 requests por segundo
            assert throughput > 10.0
            assert len(results) == request_count
            
        finally:
            await api_server.orchestrator.stop()

# =============================================================================
# EJECUCIÓN DE TESTS
# =============================================================================

def run_all_tests():
    """Ejecuta todos los tests"""
    print("🧪 THE AGENTS - Ejecutando Test Suite")
    print("=" * 50)
    
    # Ejecutar tests usando pytest
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, 
        "-v", "--tb=short", "--asyncio-mode=auto"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Todos los tests pasaron exitosamente")
    else:
        print("❌ Algunos tests fallaron:")
        print(result.stdout)
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    # Ejecutar tests cuando se llame directamente
    run_all_tests()
