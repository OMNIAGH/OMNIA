"""
THE AGENTS - Módulo de Operación y Ejecución
Sistema de agentes conversacionales y operativos para OMNIA

Autor: OMNIA Development Team
Fecha: 2025-11-07
Versión: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import re
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

class AgentType(Enum):
    VOICE = "voice"
    TEXT = "text"
    SEARCH = "search"
    CLASSIFICATION = "classification"
    VERIFICATION = "verification"
    CRM = "crm"
    EMAIL = "email"
    SMS = "sms"
    CHAT = "chat"

class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UserRequest:
    """Solicitud de usuario"""
    request_id: str
    user_id: str
    timestamp: datetime
    agent_type: AgentType
    content: Dict[str, Any]
    context: Dict[str, Any]
    security_level: SecurityLevel
    priority: int = 5  # 1-10, 10 es máxima prioridad
    session_id: Optional[str] = None

@dataclass
class AgentResponse:
    """Respuesta de agente"""
    request_id: str
    agent_type: AgentType
    content: Dict[str, Any]
    confidence: float
    execution_time: float
    status: RequestStatus
    error_message: Optional[str] = None
    escalation_required: bool = False
    human_handoff_needed: bool = False

@dataclass
class SessionContext:
    """Contexto de sesión"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    active_agents: List[AgentType]

@dataclass
class SecurityValidation:
    """Validación de seguridad"""
    approved: bool
    confidence: float
    restrictions: List[str]
    audit_id: str
    timestamp: datetime

# =============================================================================
# CLASE BASE DE AGENTES
# =============================================================================

class BaseAgent(ABC):
    """Clase base para todos los agentes"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.config = config
        self.logger = logging.getLogger(f"Agent.{agent_type.value}")
        self.is_active = True
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
    
    @abstractmethod
    async def process(self, request: UserRequest) -> AgentResponse:
        """Procesa una solicitud del usuario"""
        pass
    
    async def validate_request(self, request: UserRequest) -> bool:
        """Valida una solicitud antes del procesamiento"""
        try:
            # Validaciones básicas
            if not request.user_id:
                return False
            
            if not request.content:
                return False
            
            if request.agent_type != self.agent_type:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating request: {e}")
            return False
    
    def update_stats(self, execution_time: float, success: bool):
        """Actualiza estadísticas del agente"""
        self.stats['total_requests'] += 1
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # Calcular promedio móvil de tiempo de respuesta
        current_avg = self.stats['avg_response_time']
        total_requests = self.stats['total_requests']
        self.stats['avg_response_time'] = (current_avg * (total_requests - 1) + execution_time) / total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del agente"""
        return {
            'agent_type': self.agent_type.value,
            'is_active': self.is_active,
            'stats': self.stats.copy(),
            'success_rate': (
                self.stats['successful_requests'] / max(self.stats['total_requests'], 1)
            ) * 100
        }

# =============================================================================
# AGENTES CONVERSACIONALES
# =============================================================================

class BaseConversationalAgent(BaseAgent):
    """Clase base para agentes conversacionales"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        super().__init__(agent_type, config)
        self.supported_languages = config.get('supported_languages', ['es', 'en'])
        self.max_context_length = config.get('max_context_length', 1000)
    
    async def analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analiza la intención del mensaje"""
        # Análisis básico de intenciones
        intent_patterns = {
            'status_inquiry': [r'status', r'cómo.*(?:está|va)', r'estado'],
            'campaign_info': [r'campaña', r'campaign', r'crear.*campaña'],
            'search_request': [r'buscar', r'search', r'encontrar'],
            'help_request': [r'ayuda', r'help', r'cómo.*hacer'],
            'configuration': [r'configurar', r'config', r'ajustar']
        }
        
        message_lower = message.lower()
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return {
                        'intent': intent,
                        'confidence': 0.8,
                        'entities': self._extract_entities(message)
                    }
        
        return {
            'intent': 'general_query',
            'confidence': 0.5,
            'entities': self._extract_entities(message)
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extrae entidades del texto"""
        entities = []
        
        # Patrones para fechas
        date_patterns = [
            r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b',
            r'\b(hoy|ayer|mañana|esta semana|este mes)\b'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'date',
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Patrones para números/montos
        number_patterns = [
            r'\$[\d,]+',
            r'\b\d+(?:\.\d{2})?\b',
            r'\b\d+%?\b'
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'type': 'number',
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities

class VoiceAgent(BaseConversationalAgent):
    """Agente para procesamiento de voz"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.VOICE, config)
        self.stt_provider = config.get('stt_provider', 'google')
        self.tts_provider = config.get('tts_provider', 'google')
        self.supported_formats = config.get('supported_formats', ['wav', 'mp3', 'flac'])
    
    async def process(self, request: UserRequest) -> AgentResponse:
        """Procesa entrada de voz"""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            # Validar solicitud
            if not await self.validate_request(request):
                return AgentResponse(
                    request_id=request_id,
                    agent_type=self.agent_type,
                    content={'error': 'Invalid request'},
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    status=RequestStatus.FAILED
                )
            
            # Extraer datos de audio
            audio_data = request.content.get('audio_data')
            if not audio_data:
                raise ValueError("No audio data provided")
            
            # 1. Speech-to-Text
            text_input = await self._speech_to_text(audio_data)
            
            # 2. Procesamiento de texto
            response_text = await self._process_text_input(text_input, request.context)
            
            # 3. Text-to-Speech
            response_audio = await self._text_to_speech(response_text)
            
            # 4. Construir respuesta
            response_content = {
                'original_text': text_input,
                'response_text': response_text,
                'response_audio': response_audio,
                'language': request.context.get('language', 'es'),
                'format': 'audio_response'
            }
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.95,
                execution_time=execution_time,
                status=RequestStatus.COMPLETED
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            self.logger.error(f"Voice processing failed: {e}")
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content={'error': str(e)},
                confidence=0.0,
                execution_time=execution_time,
                status=RequestStatus.FAILED
            )
    
    async def _speech_to_text(self, audio_data: str) -> str:
        """Convierte audio a texto"""
        # Simulación de Speech-to-Text
        await asyncio.sleep(0.1)  # Simular procesamiento
        
        # En implementación real, usaría APIs como Google Speech-to-Text
        return "Hola, ¿cómo puedo ayudarte hoy?"
    
    async def _process_text_input(self, text: str, context: Dict[str, Any]) -> str:
        """Procesa entrada de texto"""
        # Análisis de intención
        intent_analysis = await self.analyze_intent(text)
        
        # Generar respuesta contextual
        if intent_analysis['intent'] == 'status_inquiry':
            return "Tu última campaña tiene un performance del 85%. ¿Te gustaría ver los detalles?"
        elif intent_analysis['intent'] == 'help_request':
            return "Estoy aquí para ayudarte. Puedes preguntarme sobre el status de tus campañas, crear nuevas campañas, o buscar información específica."
        else:
            return "He recibido tu mensaje. ¿Podrías ser más específico sobre lo que necesitas?"
    
    async def _text_to_speech(self, text: str) -> str:
        """Convierte texto a audio"""
        # Simulación de Text-to-Speech
        await asyncio.sleep(0.05)  # Simular procesamiento
        
        # En implementación real, generaría audio real
        return f"audio_data_{hashlib.md5(text.encode()).hexdigest()}"

class TextAgent(BaseConversationalAgent):
    """Agente para procesamiento de texto"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.TEXT, config)
        self.max_message_length = config.get('max_message_length', 4000)
        self.enable_suggestions = config.get('enable_suggestions', True)
    
    async def process(self, request: UserRequest) -> AgentResponse:
        """Procesa mensaje de texto"""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            # Validar solicitud
            if not await self.validate_request(request):
                return AgentResponse(
                    request_id=request_id,
                    agent_type=self.agent_type,
                    content={'error': 'Invalid request'},
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    status=RequestStatus.FAILED
                )
            
            # Extraer mensaje
            message = request.content.get('message', '')
            if not message:
                raise ValueError("No message provided")
            
            # Validar longitud del mensaje
            if len(message) > self.max_message_length:
                raise ValueError(f"Message too long: {len(message)} > {self.max_message_length}")
            
            # 1. Análisis de sentimiento
            sentiment = await self._analyze_sentiment(message)
            
            # 2. Extracción de entidades
            entities = await self._extract_entities(message)
            
            # 3. Generación de respuesta contextual
            response = await self._generate_contextual_response(
                message=message,
                sentiment=sentiment,
                entities=entities,
                context=request.context
            )
            
            # 4. Generar sugerencias si está habilitado
            suggestions = []
            if self.enable_suggestions:
                suggestions = self._generate_suggestions(message)
            
            response_content = {
                'message': message,
                'response': response,
                'sentiment': sentiment,
                'entities': entities,
                'suggestions': suggestions,
                'language': request.context.get('language', 'es'),
                'channel': request.context.get('channel', 'web')
            }
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.92,
                execution_time=execution_time,
                status=RequestStatus.COMPLETED
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            self.logger.error(f"Text processing failed: {e}")
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content={'error': str(e)},
                confidence=0.0,
                execution_time=execution_time,
                status=RequestStatus.FAILED
            )
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analiza el sentimiento del texto"""
        # Análisis básico de sentimiento
        positive_words = ['bueno', 'excelente', 'genial', 'perfecto', 'happy', 'good', 'great']
        negative_words = ['malo', 'terrible', 'odio', 'problema', 'error', 'bad', 'hate', 'problem']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = 0.7 + (positive_count * 0.1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = 0.7 + (negative_count * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': min(confidence, 0.95),
            'positive_score': positive_count,
            'negative_score': negative_count
        }
    
    async def _generate_contextual_response(self, message: str, sentiment: Dict[str, Any], 
                                          entities: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Genera respuesta contextual"""
        # Análisis de intención
        intent_analysis = await self.analyze_intent(message)
        intent = intent_analysis['intent']
        
        # Respuestas basadas en intención y sentimiento
        if intent == 'status_inquiry':
            if sentiment['sentiment'] == 'negative':
                return "Veo que estás preocupado por el status. Permíteme verificar el estado actual de tus campañas y te daré un informe completo."
            else:
                return "¡Perfecto! Te muestro el status actual de tus campañas. ¿Hay alguna en particular que te interese?"
        
        elif intent == 'campaign_info':
            return "Puedo ayudarte a crear y gestionar tus campañas. ¿Qué tipo de campaña tienes en mente? ¿Búsqueda, display, o video?"
        
        elif intent == 'search_request':
            return "Entendido, voy a buscar la información que necesitas. ¿Podrías darme más detalles sobre lo que buscas?"
        
        elif intent == 'help_request':
            return "¡Por supuesto! Estoy aquí para ayudarte. Puedes preguntarme sobre campañas, estadísticas, creación de contenido, o cualquier otra cosa que necesites."
        
        else:
            return "He recibido tu mensaje. ¿Podrías ser más específico sobre lo que necesitas? Puedo ayudarte con campañas, estadísticas, análisis y mucho más."
    
    def _generate_suggestions(self, message: str) -> List[str]:
        """Genera sugerencias basadas en el mensaje"""
        suggestions = []
        message_lower = message.lower()
        
        if 'campaña' in message_lower or 'campaign' in message_lower:
            suggestions.extend([
                "¿Cuál es el status de mis campañas?",
                "Crear nueva campaña",
                "Ver estadísticas de campaña"
            ])
        
        if 'status' in message_lower or 'estado' in message_lower:
            suggestions.extend([
                "Ver campañas activas",
                "Ver campañas pausadas",
                "Ver rendimiento del mes"
            ])
        
        if 'ayuda' in message_lower or 'help' in message_lower:
            suggestions.extend([
                "¿Cómo crear una campaña?",
                "¿Cómo ver estadísticas?",
                "Configurar alertas"
            ])
        
        return suggestions[:3]  # Máximo 3 sugerencias

# =============================================================================
# AGENTES OPERATIVOS
# =============================================================================

class BaseOperationalAgent(BaseAgent):
    """Clase base para agentes operativos"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        super().__init__(agent_type, config)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.rate_limit = config.get('rate_limit', 100)  # requests per minute

class SearchAgent(BaseOperationalAgent):
    """Agente de búsqueda inteligente"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.SEARCH, config)
        self.sources = config.get('sources', ['internal', 'external', 'omnia'])
        self.max_results = config.get('max_results', 50)
    
    async def process(self, request: UserRequest) -> AgentResponse:
        """Procesa solicitud de búsqueda"""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            # Validar solicitud
            if not await self.validate_request(request):
                return AgentResponse(
                    request_id=request_id,
                    agent_type=self.agent_type,
                    content={'error': 'Invalid request'},
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    status=RequestStatus.FAILED
                )
            
            # Extraer parámetros de búsqueda
            query = request.content.get('query', '')
            filters = request.content.get('filters', {})
            sources = request.content.get('sources', self.sources)
            
            if not query:
                raise ValueError("No search query provided")
            
            # 1. Expansión de consulta
            expanded_query = await self._expand_query(query)
            
            # 2. Búsqueda en múltiples fuentes
            results = await self._search_multiple_sources(expanded_query, filters, sources)
            
            # 3. Ranking y agregación
            ranked_results = await self._rank_and_aggregate(results)
            
            response_content = {
                'original_query': query,
                'expanded_query': expanded_query,
                'results': ranked_results,
                'total_count': len(ranked_results),
                'sources_searched': sources,
                'filters_applied': filters,
                'execution_time': time.time() - start_time
            }
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.88,
                execution_time=execution_time,
                status=RequestStatus.COMPLETED
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            self.logger.error(f"Search processing failed: {e}")
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content={'error': str(e)},
                confidence=0.0,
                execution_time=execution_time,
                status=RequestStatus.FAILED
            )
    
    async def _expand_query(self, query: str) -> str:
        """Expande la consulta de búsqueda"""
        # Expansión básica de términos
        expansions = {
            'campaña': ['campaign', 'campaña publicitaria', 'anuncio'],
            'status': ['estado', 'situación', 'rendimiento'],
            'analytics': ['estadísticas', 'métricas', 'datos']
        }
        
        expanded = query
        for term, expansions_list in expansions.items():
            if term in query.lower():
                for expansion in expansions_list:
                    if expansion not in expanded.lower():
                        expanded += f" OR {expansion}"
        
        return expanded
    
    async def _search_multiple_sources(self, query: str, filters: Dict[str, Any], 
                                     sources: List[str]) -> List[Dict[str, Any]]:
        """Busca en múltiples fuentes"""
        results = []
        
        # Búsqueda en fuente interna
        if 'internal' in sources:
            internal_results = await self._search_internal_db(query, filters)
            results.extend(internal_results)
        
        # Búsqueda en fuentes externas
        if 'external' in sources:
            external_results = await self._search_external_apis(query, filters)
            results.extend(external_results)
        
        # Búsqueda en módulos OMNIA
        if 'omnia' in sources:
            omnia_results = await self._search_omnia_modules(query, filters)
            results.extend(omnia_results)
        
        return results
    
    async def _search_internal_db(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca en base de datos interna"""
        # Simulación de búsqueda en BD
        await asyncio.sleep(0.1)
        
        return [
            {
                'id': 'result_1',
                'title': 'Campaign Performance Report',
                'content': 'Análisis detallado del performance de campañas...',
                'source': 'internal_db',
                'relevance': 0.95,
                'metadata': {
                    'type': 'report',
                    'date': '2025-11-07',
                    'tags': ['campaign', 'performance', 'analytics']
                }
            },
            {
                'id': 'result_2',
                'title': 'MIDAS Campaign Optimization',
                'content': 'Resultados de optimización automática...',
                'source': 'internal_db',
                'relevance': 0.88,
                'metadata': {
                    'type': 'optimization',
                    'date': '2025-11-06',
                    'tags': ['midas', 'optimization']
                }
            }
        ]
    
    async def _search_external_apis(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca en APIs externas"""
        # Simulación de búsqueda en APIs externas
        await asyncio.sleep(0.2)
        
        return [
            {
                'id': 'ext_result_1',
                'title': 'Google Ads Best Practices',
                'content': 'Guía completa de mejores prácticas...',
                'source': 'external_api',
                'relevance': 0.75,
                'metadata': {
                    'type': 'documentation',
                    'source_url': 'https://support.google.com/ads',
                    'tags': ['google', 'best practices']
                }
            }
        ]
    
    async def _search_omnia_modules(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca en módulos OMNIA"""
        # Simulación de búsqueda en módulos OMNIA
        await asyncio.sleep(0.1)
        
        return [
            {
                'id': 'omnia_result_1',
                'title': 'ANCHOR Data Ingestion Status',
                'content': 'Estado actual de ingesta de datos...',
                'source': 'anchor_module',
                'relevance': 0.90,
                'metadata': {
                    'type': 'status',
                    'module': 'anchor',
                    'tags': ['anchor', 'data', 'status']
                }
            }
        ]
    
    async def _rank_and_aggregate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ranking y agregación de resultados"""
        # Ordenar por relevancia
        sorted_results = sorted(results, key=lambda x: x.get('relevance', 0), reverse=True)
        
        # Aplicar límite de resultados
        return sorted_results[:self.max_results]

class CRMManagementAgent(BaseOperationalAgent):
    """Agente de gestión CRM"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.CRM, config)
        self.crm_system = config.get('crm_system', 'internal')
        self.auto_actions = config.get('auto_actions', True)
    
    async def process(self, request: UserRequest) -> AgentResponse:
        """Procesa gestión CRM"""
        start_time = time.time()
        request_id = request.request_id
        
        try:
            # Validar solicitud
            if not await self.validate_request(request):
                return AgentResponse(
                    request_id=request_id,
                    agent_type=self.agent_type,
                    content={'error': 'Invalid request'},
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    status=RequestStatus.FAILED
                )
            
            # Extraer parámetros CRM
            customer_id = request.content.get('customer_id')
            interaction_type = request.content.get('interaction_type', 'general')
            interaction_data = request.content.get('interaction_data', {})
            
            if not customer_id:
                raise ValueError("No customer_id provided")
            
            # 1. Recuperar perfil del cliente
            customer_profile = await self._get_customer_profile(customer_id)
            
            # 2. Análisis de comportamiento
            behavior_analysis = await self._analyze_customer_behavior(customer_profile, interaction_data)
            
            # 3. Automatización de acciones
            actions = await self._suggest_automated_actions(behavior_analysis, interaction_type)
            
            # 4. Ejecución de acciones
            execution_results = await self._execute_crm_actions(actions, customer_id)
            
            # 5. Sugerir próximas acciones
            next_actions = await self._suggest_next_actions(customer_id, behavior_analysis)
            
            response_content = {
                'customer_id': customer_id,
                'customer_profile': customer_profile,
                'behavior_analysis': behavior_analysis,
                'actions_suggested': actions,
                'actions_executed': execution_results,
                'next_actions': next_actions,
                'interaction_summary': {
                    'type': interaction_type,
                    'data': interaction_data,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.90,
                execution_time=execution_time,
                status=RequestStatus.COMPLETED
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            self.logger.error(f"CRM processing failed: {e}")
            
            return AgentResponse(
                request_id=request_id,
                agent_type=self.agent_type,
                content={'error': str(e)},
                confidence=0.0,
                execution_time=execution_time,
                status=RequestStatus.FAILED
            )
    
    async def _get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """Obtiene perfil del cliente"""
        # Simulación de recuperación de perfil
        await asyncio.sleep(0.1)
        
        return {
            'customer_id': customer_id,
            'name': 'Juan Pérez',
            'email': 'juan.perez@email.com',
            'phone': '+1234567890',
            'company': 'Tech Solutions Inc.',
            'segment': 'enterprise',
            'lifetime_value': 5000.0,
            'last_interaction': '2025-11-05T14:30:00Z',
            'total_campaigns': 12,
            'avg_campaign_budget': 1500.0,
            'performance_score': 8.5
        }
    
    async def _analyze_customer_behavior(self, profile: Dict[str, Any], 
                                       interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza comportamiento del cliente"""
        # Análisis básico de comportamiento
        time_since_last = (datetime.now(timezone.utc) - datetime.fromisoformat(profile['last_interaction'])).days
        
        behavior_score = 0
        if time_since_last < 7:
            behavior_score = 9
        elif time_since_last < 30:
            behavior_score = 7
        elif time_since_last < 90:
            behavior_score = 5
        else:
            behavior_score = 3
        
        return {
            'engagement_level': 'high' if behavior_score > 7 else 'medium' if behavior_score > 5 else 'low',
            'time_since_last_interaction': time_since_last,
            'behavior_score': behavior_score,
            'recommendation_priority': 'high' if time_since_last > 30 else 'normal',
            'risk_factors': ['long_time_no_interaction'] if time_since_last > 60 else []
        }
    
    async def _suggest_automated_actions(self, behavior_analysis: Dict[str, Any], 
                                       interaction_type: str) -> List[Dict[str, Any]]:
        """Sugiere acciones automatizadas"""
        actions = []
        
        if behavior_analysis['time_since_last_interaction'] > 30:
            actions.append({
                'type': 'outreach',
                'description': 'Enviar email de reactivación',
                'priority': 'high',
                'automated': True
            })
        
        if behavior_analysis['behavior_score'] > 8:
            actions.append({
                'type': 'upsell',
                'description': 'Sugerir campaña premium',
                'priority': 'medium',
                'automated': True
            })
        
        return actions
    
    async def _execute_crm_actions(self, actions: List[Dict[str, Any]], 
                                 customer_id: str) -> List[Dict[str, Any]]:
        """Ejecuta acciones CRM"""
        executed_actions = []
        
        for action in actions:
            if action.get('automated', False):
                # Simular ejecución de acción automatizada
                await asyncio.sleep(0.05)
                
                executed_actions.append({
                    'action': action,
                    'status': 'completed',
                    'timestamp': datetime.utcnow().isoformat(),
                    'result': f"Action {action['type']} executed successfully"
                })
            else:
                executed_actions.append({
                    'action': action,
                    'status': 'pending_human',
                    'timestamp': datetime.utcnow().isoformat(),
                    'result': "Action requires human intervention"
                })
        
        return executed_actions
    
    async def _suggest_next_actions(self, customer_id: str, 
                                  behavior_analysis: Dict[str, Any]) -> List[str]:
        """Sugiere próximas acciones"""
        suggestions = []
        
        if behavior_analysis['time_since_last_interaction'] > 30:
            suggestions.append("Programar llamada de seguimiento")
            suggestions.append("Enviar propuesta personalizada")
        
        if behavior_analysis['behavior_score'] > 8:
            suggestions.append("Invitar a webinar exclusivo")
            suggestions.append("Ofrecer consultoría gratuita")
        
        suggestions.append("Actualizar perfil de preferencias")
        
        return suggestions

# =============================================================================
# SISTEMA DE ORQUESTACIÓN
# =============================================================================

class AgentOrchestrator:
    """Orquestador principal de agentes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AgentOrchestrator")
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.active_sessions: Dict[str, SessionContext] = {}
        self.task_queue = asyncio.Queue()
        self.human_handoff_queue = asyncio.Queue()
        self.running = False
        self.security_protocol = SecurityProtocol(config)
        
        # Inicializar agentes por defecto
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Inicializa agentes por defecto"""
        default_config = self.config.get('default_agents', {})
        
        # Agentes conversacionales
        voice_config = default_config.get('voice', {})
        if voice_config.get('enabled', True):
            self.agents[AgentType.VOICE] = VoiceAgent(voice_config)
        
        text_config = default_config.get('text', {})
        if text_config.get('enabled', True):
            self.agents[AgentType.TEXT] = TextAgent(text_config)
        
        # Agentes operativos
        search_config = default_config.get('search', {})
        if search_config.get('enabled', True):
            self.agents[AgentType.SEARCH] = SearchAgent(search_config)
        
        crm_config = default_config.get('crm', {})
        if crm_config.get('enabled', True):
            self.agents[AgentType.CRM] = CRMManagementAgent(crm_config)
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def register_agent(self, agent_type: AgentType, agent_instance: BaseAgent):
        """Registra un nuevo agente"""
        self.agents[agent_type] = agent_instance
        self.logger.info(f"Agent {agent_type.value} registered successfully")
    
    async def process_user_request(self, request: UserRequest) -> AgentResponse:
        """Procesa solicitud de usuario"""
        start_time = time.time()
        
        try:
            # 1. Análisis de intención
            intent_analysis = await self.analyze_intent(request)
            
            # 2. Selección de agente
            selected_agent = self.select_agent(intent_analysis, request.agent_type)
            
            if not selected_agent:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_type=request.agent_type,
                    content={'error': 'No suitable agent found'},
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    status=RequestStatus.FAILED
                )
            
            # 3. Verificación de seguridad (placeholder)
            security_check = await self.security_protocol.validate_request(request)
            if not security_check.approved:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_type=request.agent_type,
                    content={'error': 'Security validation failed', 'restrictions': security_check.restrictions},
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    status=RequestStatus.FAILED
                )
            
            # 4. Procesamiento por agente
            response = await selected_agent.process(request)
            
            # 5. Actualización de contexto de sesión
            await self.update_session_context(request, response)
            
            # 6. Verificar si necesita escalación
            if response.escalation_required or response.human_handoff_needed:
                await self.initiate_handoff(request, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            return AgentResponse(
                request_id=request.request_id,
                agent_type=request.agent_type,
                content={'error': str(e)},
                confidence=0.0,
                execution_time=time.time() - start_time,
                status=RequestStatus.FAILED
            )
    
    async def analyze_intent(self, request: UserRequest) -> Dict[str, Any]:
        """Analiza la intención de la solicitud"""
        # Análisis básico de intención basado en contenido
        content = request.content
        
        if 'message' in content:
            message = content['message'].lower()
            if any(word in message for word in ['status', 'estado', 'cómo va']):
                return {'intent': 'status_inquiry', 'confidence': 0.9}
            elif any(word in message for word in ['buscar', 'search', 'encontrar']):
                return {'intent': 'search_request', 'confidence': 0.85}
            elif any(word in message for word in ['ayuda', 'help', 'cómo']):
                return {'intent': 'help_request', 'confidence': 0.8}
        
        return {'intent': 'general_query', 'confidence': 0.5}
    
    def select_agent(self, intent_analysis: Dict[str, Any], requested_type: AgentType) -> Optional[BaseAgent]:
        """Selecciona el agente apropiado"""
        # Si se especifica un tipo de agente, usarlo
        if requested_type in self.agents:
            return self.agents[requested_type]
        
        # Selección basada en intención
        intent = intent_analysis['intent']
        
        agent_mapping = {
            'status_inquiry': AgentType.SEARCH,
            'search_request': AgentType.SEARCH,
            'help_request': AgentType.TEXT,
            'general_query': AgentType.TEXT
        }
        
        selected_type = agent_mapping.get(intent, AgentType.TEXT)
        return self.agents.get(selected_type)
    
    async def update_session_context(self, request: UserRequest, response: AgentResponse):
        """Actualiza contexto de sesión"""
        session_id = request.session_id or request.user_id
        
        if session_id not in self.active_sessions:
            # Crear nueva sesión
            self.active_sessions[session_id] = SessionContext(
                session_id=session_id,
                user_id=request.user_id,
                start_time=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                conversation_history=[],
                user_preferences={},
                active_agents=[]
            )
        
        session = self.active_sessions[session_id]
        session.last_activity = datetime.utcnow()
        
        # Agregar al historial de conversación
        session.conversation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'request': asdict(request),
            'response': asdict(response)
        })
        
        # Mantener solo los últimos 100 elementos
        if len(session.conversation_history) > 100:
            session.conversation_history = session.conversation_history[-100:]
        
        # Actualizar agentes activos
        if request.agent_type not in session.active_agents:
            session.active_agents.append(request.agent_type)
    
    async def initiate_handoff(self, request: UserRequest, response: AgentResponse):
        """Inicia transferencia a operador humano"""
        handoff_request = {
            'request_id': request.request_id,
            'user_id': request.user_id,
            'session_id': request.session_id,
            'reason': 'human_intervention_required',
            'context': {
                'original_request': asdict(request),
                'agent_response': asdict(response),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        await self.human_handoff_queue.put(handoff_request)
        self.logger.info(f"Human handoff initiated for request {request.request_id}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de todos los agentes"""
        stats = {
            'total_agents': len(self.agents),
            'agents': {},
            'queue_status': {
                'task_queue_size': self.task_queue.qsize(),
                'handoff_queue_size': self.human_handoff_queue.qsize()
            }
        }
        
        for agent_type, agent in self.agents.items():
            stats['agents'][agent_type.value] = agent.get_stats()
        
        return stats
    
    async def start(self):
        """Inicia el orquestador"""
        self.running = True
        self.logger.info("Agent Orchestrator started")
    
    async def stop(self):
        """Detiene el orquestador"""
        self.running = False
        self.logger.info("Agent Orchestrator stopped")

# =============================================================================
# PROTOCOLO DE SEGURIDAD
# =============================================================================

class SecurityProtocol:
    """Protocolo de seguridad para validación de solicitudes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("SecurityProtocol")
    
    async def validate_request(self, request: UserRequest) -> SecurityValidation:
        """Valida solicitud según protocolo de seguridad"""
        try:
            # Validaciones básicas
            audit_id = str(uuid.uuid4())
            
            # Verificar restricciones de contenido
            restrictions = []
            content = str(request.content)
            
            # Verificar PII (simplificado)
            if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content):
                restrictions.append("potential_pii_detected")
            
            # Verificar contenido malicioso básico
            malicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
            for pattern in malicious_patterns:
                if pattern.lower() in content.lower():
                    restrictions.append("potential_malicious_content")
                    break
            
            # Determinar si se aprueba
            approved = len(restrictions) == 0 or request.security_level == SecurityLevel.LOW
            
            return SecurityValidation(
                approved=approved,
                confidence=0.9 if approved else 0.7,
                restrictions=restrictions,
                audit_id=audit_id,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            return SecurityValidation(
                approved=False,
                confidence=0.0,
                restrictions=['validation_error'],
                audit_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow()
            )

# =============================================================================
# API SERVER
# =============================================================================

class AgentsAPIServer:
    """Servidor API para THE AGENTS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = AgentOrchestrator(config)
        self.security_protocol = SecurityProtocol(config)
        self.logger = logging.getLogger("AgentsAPIServer")
    
    async def create_user_request(self, request_data: Dict[str, Any]) -> UserRequest:
        """Crea objeto UserRequest desde datos de API"""
        return UserRequest(
            request_id=str(uuid.uuid4()),
            user_id=request_data['user_id'],
            timestamp=datetime.utcnow(),
            agent_type=AgentType(request_data.get('agent_type', 'text')),
            content=request_data['content'],
            context=request_data.get('context', {}),
            security_level=SecurityLevel(request_data.get('security_level', 'medium')),
            priority=request_data.get('priority', 5),
            session_id=request_data.get('session_id')
        )
    
    async def handle_conversational_voice(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitud de voz"""
        try:
            user_request = await self.create_user_request(request_data)
            response = await self.orchestrator.process_user_request(user_request)
            
            return {
                'success': response.status == RequestStatus.COMPLETED,
                'data': asdict(response),
                'request_id': user_request.request_id
            }
            
        except Exception as e:
            self.logger.error(f"Voice request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'request_id': request_data.get('request_id', str(uuid.uuid4()))
            }
    
    async def handle_conversational_text(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitud de texto"""
        try:
            user_request = await self.create_user_request(request_data)
            response = await self.orchestrator.process_user_request(user_request)
            
            return {
                'success': response.status == RequestStatus.COMPLETED,
                'data': asdict(response),
                'request_id': user_request.request_id
            }
            
        except Exception as e:
            self.logger.error(f"Text request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'request_id': request_data.get('request_id', str(uuid.uuid4()))
            }
    
    async def handle_search_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitud de búsqueda"""
        try:
            user_request = await self.create_user_request(request_data)
            response = await self.orchestrator.process_user_request(user_request)
            
            return {
                'success': response.status == RequestStatus.COMPLETED,
                'data': asdict(response),
                'request_id': user_request.request_id
            }
            
        except Exception as e:
            self.logger.error(f"Search request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'request_id': request_data.get('request_id', str(uuid.uuid4()))
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'agents': self.orchestrator.get_agent_stats(),
            'version': '1.0.0'
        }

# =============================================================================
# DEMO Y TESTING
# =============================================================================

async def demo_agents_system():
    """Demostración del sistema de agentes"""
    print("🤖 THE AGENTS - Demostración del Sistema")
    print("=" * 50)
    
    # Configuración
    config = {
        'default_agents': {
            'voice': {'enabled': True},
            'text': {'enabled': True},
            'search': {'enabled': True},
            'crm': {'enabled': True}
        }
    }
    
    # Inicializar servidor
    server = AgentsAPIServer(config)
    await server.orchestrator.start()
    
    print("✅ Sistema inicializado correctamente")
    print(f"📊 Agentes disponibles: {len(server.orchestrator.agents)}")
    
    # Demo 1: Agente de texto
    print("\n📝 Demo 1: Agente de Texto")
    text_request = {
        'user_id': 'user_123',
        'agent_type': 'text',
        'content': {
            'message': '¿Cuál es el status de mis campañas?'
        },
        'context': {
            'language': 'es',
            'channel': 'web'
        }
    }
    
    text_result = await server.handle_conversational_text(text_request)
    print(f"✅ Resultado: {text_result['success']}")
    if text_result['success']:
        response_data = text_result['data']['content']
        print(f"💬 Respuesta: {response_data.get('response', 'N/A')}")
        print(f"🎯 Confianza: {text_result['data']['confidence']:.2f}")
    
    # Demo 2: Agente de búsqueda
    print("\n🔍 Demo 2: Agente de Búsqueda")
    search_request = {
        'user_id': 'user_123',
        'agent_type': 'search',
        'content': {
            'query': 'campaign performance',
            'filters': {'date_range': 'last_7_days'}
        }
    }
    
    search_result = await server.handle_search_request(search_request)
    print(f"✅ Resultado: {search_result['success']}")
    if search_result['success']:
        response_data = search_result['data']['content']
        print(f"🔢 Resultados encontrados: {response_data.get('total_count', 0)}")
        print(f"⏱️ Tiempo de ejecución: {response_data.get('execution_time', 0):.3f}s")
    
    # Demo 3: Agente CRM
    print("\n👤 Demo 3: Agente CRM")
    crm_request = {
        'user_id': 'user_123',
        'agent_type': 'crm',
        'content': {
            'customer_id': 'customer_456',
            'interaction_type': 'follow_up',
            'interaction_data': {
                'call_duration': 300,
                'notes': 'Customer interested in premium features'
            }
        }
    }
    
    crm_result = await server.handle_conversational_text(crm_request)
    print(f"✅ Resultado: {crm_result['success']}")
    if crm_result['success']:
        response_data = crm_result['data']['content']
        print(f"👤 Cliente: {response_data.get('customer_profile', {}).get('name', 'N/A')}")
        print(f"⚡ Acciones ejecutadas: {len(response_data.get('actions_executed', []))}")
    
    # Estado del sistema
    print("\n📈 Estado del Sistema")
    system_status = await server.get_system_status()
    print(f"💚 Estado: {system_status['status']}")
    print(f"🤖 Total de agentes: {system_status['agents']['total_agents']}")
    
    await server.orchestrator.stop()
    print("\n🎉 Demostración completada")

if __name__ == "__main__":
    # Ejecutar demo
    asyncio.run(demo_agents_system())
