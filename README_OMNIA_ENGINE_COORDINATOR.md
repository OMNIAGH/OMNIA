# 🎯 OMNIA ENGINE COORDINATOR

## 📋 Descripción

El **OMNIA ENGINE COORDINATOR** es el **orquestador central** del ecosistema OMNIA, diseñado para coordinar y gestionar el flujo de datos entre los tres módulos principales:

- **🏗️ ANCHOR** (4,169 líneas) - Ingesta de datos de múltiples fuentes
- **🔍 CENSOR** (3,500+ líneas) - Supervisión y análisis ML  
- **📈 NOESIS** (6,700+ líneas) - Predicción y forecasting

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                OMNIA ENGINE COORDINATOR                     │
│              (Orquestador Central v1.0)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔐 OMNIA PROTOCOL (4 Capas de Seguridad)                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐│
│  │   SHIELD    │ │  GUARDIAN   │ │  SENTINEL   │ │ WATCHER ││
│  │(Perimetral) │ │(Prompts)    │ │(Contenido)  │ │(Behavioral)│
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘│
│                                                             │
│  🔄 FLUJO DE PROCESAMIENTO                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐           │
│  │   ANCHOR  │───▶│   CENSOR  │───▶│   NOESIS  │           │
│  │(Ingesta)  │    │(ML Superv)│    │(Forecast) │           │
│  └───────────┘    └───────────┘    └───────────┘           │
│         │                │                │                │
│         ▼                ▼                ▼                │
│  📊 Datos  │      🔍 Anomalías │      📈 Predicciones     │
│  📥 Fuentes│      🏷️ Auto-labels│      📉 Forecasting     │
│  ✅ Valida │      📋 Clasifica │      🎯 A/B Testing      │
│                                                             │
│  🎯 ORCHESTRATION FINAL                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Integración y Respuesta Final                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Características Principales

### ✅ **Orquestación Completa**
- **Flujo secuencial** optimizado: ANCHOR → CENSOR → NOESIS
- **Gestión de estado** centralizada con persistencia
- **Manejo de errores** robusto con recuperación automática
- **Monitoreo** en tiempo real de cada etapa

### 🔐 **OMNIA PROTOCOL - Seguridad de 4 Capas**
1. **🛡️ SHIELD** - Validación perimetral y filtrado de entrada
2. **🛡️ GUARDIAN** - Validación de prompts y contexto
3. **👁️ SENTINEL** - Análisis de contenido y filtrado
4. **👁️ WATCHER** - Monitoreo de comportamiento y telemetría

### 🔌 **Integración de Módulos**
- **ANCHOR**: Conectores para Google Ads, Meta Ads, LinkedIn, Twitter, TikTok, Pinterest, CSV/Excel
- **CENSOR**: Detección de anomalías, auto-etiquetado, clasificación ML, validación de integridad
- **NOESIS**: Forecasting (ARIMA, Prophet, XGBoost), A/B testing, análisis de tendencias

### 💾 **Almacenamiento Híbrido**
- **SQLite** para persistencia de requests y historial
- **Redis** para cache y colas de procesamiento
- **Logging completo** en archivo y consola

### 🌐 **API REST Completa**
- `/api/v1/omnia/process` - Procesamiento principal
- `/health` - Health check del sistema
- `/status/<request_id>` - Estado de requests específicas

## 📦 Instalación y Configuración

### Prerrequisitos
```bash
# Dependencias Python
pip install aiohttp asyncio requests redis sqlite3

# Servicios opcionales
sudo apt-get install redis-server  # Para cache distribuido
```

### Configuración Rápida
```bash
# 1. Clonar/descargar el coordinador
cd /workspace

# 2. Verificar dependencias
python3 -c "import aiohttp, requests, redis, sqlite3; print('✅ Dependencies OK')"

# 3. Iniciar el coordinador
python3 omnia_engine_coordinator.py
```

### Variables de Entorno (Opcional)
```bash
# Base de datos
export POSTGRES_HOST="localhost"  # Si conectas a PostgreSQL real
export REDIS_HOST="localhost"
export REDIS_PORT=6379

# Logging
export LOG_LEVEL="INFO"

# Módulos externos (si están disponibles)
export ANCHOR_API_URL="http://localhost:8001"
export CENSOR_API_URL="http://localhost:8002"
export NOESIS_API_URL="http://localhost:8003"
```

## 🚀 Uso Rápido

### 1. Iniciar el Servidor
```bash
python3 omnia_engine_coordinator.py
```

Salida esperada:
```
╔════════════════════════════════════════════════════════════════╗
║              OMNIA ENGINE COORDINATOR v1.0                    ║
║          Orquestador Central del Ecosistema OMNIA             ║
╚════════════════════════════════════════════════════════════════╝

🏗️  Arquitectura del Sistema:
   • ANCHOR - Ingesta de Datos
   • CENSOR - Supervisión ML
   • NOESIS - Forecasting
   • OMNIA PROTOCOL - 4 Capas de Seguridad

✅ Sistema inicializado correctamente
🚀 Servidor iniciado en http://localhost:8004
```

### 2. Probar Health Check
```bash
curl http://localhost:8004/health
```

### 3. Procesar Request
```bash
curl -X POST http://localhost:8004/api/v1/omnia/process \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analiza mis campañas de Google Ads y Meta Ads del último mes",
    "userId": "test_user",
    "sessionId": "test_session"
  }'
```

## 🧪 Testing y Validación

### Ejecutar Suite de Pruebas
```bash
# Suite completa de tests
python3 test_omnia_coordinator.py

# Tests con URL personalizada
python3 test_omnia_coordinator.py http://localhost:8004
```

**Tests incluidos:**
- ✅ Health Check
- 🔐 OMNIA Protocol Security
- 📥 ANCHOR Integration
- 🔍 CENSOR Integration  
- 📈 NOESIS Integration
- 🔄 Complete End-to-End Workflow
- ❌ Error Handling
- ⚡ Performance Benchmarks

### Ejecutar Demostración Completa
```bash
# Demo interactiva con ejemplos reales
python3 demo_omnia_coordinator.py

# Demo con URL personalizada
python3 demo_omnia_coordinator.py http://localhost:8004
```

**Demos incluidas:**
- 📊 Análisis Básico de Marketing
- 📈 Predicción y Forecasting
- 🎯 Optimización A/B Testing
- 🔄 Análisis Integral Completo
- 🔐 Escenarios de Seguridad
- ⚡ Comparación de Rendimiento

## 📊 Ejemplos de Uso

### Ejemplo 1: Análisis Básico
```python
import requests

response = requests.post('http://localhost:8004/api/v1/omnia/process', json={
    "message": "Analiza el rendimiento de mis campañas de Google Ads este trimestre",
    "userId": "user_123",
    "sessionId": "session_456"
})

result = response.json()
print(f"Procesado en {result['metadata']['processing_time']:.2f}s")
print(f"Registros procesados: {result['metadata']['anchor_data']['records_processed']}")
print(f"Anomalías detectadas: {result['metadata']['censor_analysis']['anomalies_detected']}")
```

### Ejemplo 2: Forecasting Completo
```python
import requests

response = requests.post('http://localhost:8004/api/v1/omnia/process', json={
    "message": """Necesito predecir la demanda de mis productos para los próximos 30 días.
                Considera tendencias estacionales, campañas activas y patrones históricos.""",
    "userId": "user_forecast",
    "sessionId": "session_forecast"
})

result = response.json()
forecast_data = result['response']['data_sources']['noesis']
print(f"Modelo: {forecast_data['forecast_model']}")
print(f"Tendencia: {forecast_data['trend_direction']}")
print(f"Horizonte: {forecast_data['prediction_horizon']} días")
```

### Ejemplo 3: Análisis de Seguridad
```python
import requests

# Query con PII (debería ser limpiada)
response = requests.post('http://localhost:8004/api/v1/omnia/process', json={
    "message": "Mi email es test@example.com, analiza las métricas de mis campañas",
    "userId": "security_test",
    "sessionId": "security_session"
})

result = response.json()
security = result['response']['security_validation']
print(f"Protocol levels: {security['protocol_levels']}")
print(f"Content filtered: {security['content_filtered']}")
print(f"Security score: {security['security_score']}")
```

## 🔧 Arquitectura Técnica

### Flujo de Procesamiento
1. **Validación** (OMNIA Protocol)
   - SHIELD: Validación perimetral
   - GUARDIAN: Análisis de prompts
   - SENTINEL: Filtrado de contenido
   - WATCHER: Monitoreo de comportamiento

2. **ANCHOR - Ingesta de Datos**
   - Conectores configurables
   - Validación automática
   - Rate limiting
   - Normalización de datos

3. **CENSOR - Supervisión ML**
   - Detección de anomalías
   - Auto-etiquetado inteligente
   - Clasificación automática
   - Validación de integridad

4. **NOESIS - Predicción**
   - Forecasting avanzado
   - A/B testing automático
   - Análisis de tendencias
   - Optimización de experimentos

5. **Orquestación Final**
   - Integración de resultados
   - Generación de insights
   - Respuesta estructurada

### Componentes Principales

#### `OmnIAEngineCoordinator`
```python
class OmnIAEngineCoordinator:
    """Orquestador principal del ecosistema OMNIA"""
    
    def __init__(self):
        self.omnia_protocol = OmnIAProtocol()
        self.anchor_client = AnchorClient()
        self.censor_client = CensorClient()
        self.noesis_client = NoesisClient()
```

#### `OmnIAProtocol`
```python
class OmnIAProtocol:
    """Implementación del protocolo de seguridad de 4 capas"""
    
    def shield_validate(self, data, user_id):
        """SHIELD - Validación perimetral"""
    
    def guardian_analyze(self, data, context):
        """GUARDIAN - Validación de prompts"""
    
    def sentinel_filter(self, content):
        """SENTINEL - Filtrado de contenido"""
    
    def watcher_monitor(self, user_id, action, data):
        """WATCHER - Monitoreo de comportamiento"""
```

### Estructura de Datos

#### `OmnIARequest`
```python
@dataclass
class OmnIARequest:
    request_id: str
    user_id: str
    session_id: str
    original_query: str
    processed_query: str
    security_level: SecurityLevel
    current_stage: ProcessingStage
    status: RequestStatus
    created_at: datetime
    updated_at: datetime
    context: Dict[str, Any]
    # Resultados de cada módulo
    anchor_data: Optional[Dict] = None
    censor_analysis: Optional[Dict] = None
    noesis_prediction: Optional[Dict] = None
    final_response: Optional[Dict] = None
```

#### Respuesta Estructurada
```json
{
  "success": true,
  "request_id": "req_abc123",
  "response": {
    "type": "omnia_coordinated",
    "content": "Análisis completo...",
    "summary": "Resumen ejecutivo",
    "insights": ["Insight 1", "Insight 2"],
    "recommendations": [
      {
        "test_name": "optimization_test",
        "expected_impact": "15-20% mejora"
      }
    ],
    "data_sources": {
      "anchor": {
        "records_processed": 150,
        "sources": ["google_ads", "meta_ads"]
      },
      "censor": {
        "anomalies_detected": 3,
        "quality_score": 0.85,
        "auto_labels": 12
      },
      "noesis": {
        "forecast_model": "XGBoost",
        "trend_direction": "increasing",
        "forecast_horizon": 30
      }
    },
    "security_validation": {
      "protocol_levels": ["SHIELD", "GUARDIAN", "SENTINEL", "WATCHER"],
      "content_filtered": false,
      "security_score": 0.95
    }
  },
  "metadata": {
    "processing_time": 8.45,
    "stages_completed": [
      "anchor_ingestion",
      "censor_supervision", 
      "noesis_forecasting",
      "final_orchestration"
    ]
  }
}
```

## 📈 Métricas y Monitoreo

### Métricas de Rendimiento
- **Throughput**: Requests procesadas por minuto
- **Latency**: Tiempo promedio de procesamiento (<15s objetivo)
- **Success Rate**: Tasa de éxito (>90% objetivo)
- **Error Rate**: Tasa de errores por tipo

### Métricas de Módulos
- **ANCHOR**: Registros procesados, fuentes conectadas, tiempo de ingesta
- **CENSOR**: Anomalías detectadas, score de calidad, auto-labels aplicados
- **NOESIS**: Horizonte de predicción, modelo utilizado, dirección de tendencia

### Métricas de Seguridad
- **Threats Blocked**: Amenazas bloqueadas por capa
- **PII Cleaned**: Datos personales limpiados
- **Security Score**: Score de seguridad general
- **Protocol Compliance**: Cumplimiento del protocolo

## 🔐 Seguridad

### OMNIA PROTOCOL - Detalles Técnicos

#### SHIELD (Nivel 1)
```python
# Validación perimetral
threat_indicators = [
    len(data) > 5000,  # Query muy larga
    'drop table' in data.lower(),
    'script' in data.lower()
]
threat_score = sum(indicators) / len(indicators)
```

#### GUARDIAN (Nivel 2)
```python
# Detección de prompt injection
injection_patterns = [
    r'ignore previous instructions',
    r'forget everything you know',
    r'you are now a different ai'
]
```

#### SENTINEL (Nivel 3)
```python
# Filtrado de contenido tóxico
toxic_patterns = ['hate speech', 'harassment', 'violent content']
toxicity_score = sum(pattern in content for pattern in toxic_patterns)
```

#### WATCHER (Nivel 4)
```python
# Monitoreo de comportamiento
behavioral_analysis = {
    'user_patterns': analyze_user_behavior(user_id),
    'anomaly_score': calculate_behavioral_anomaly(data),
    'telemetry': log_security_event(event)
}
```

### Configuración de Seguridad
```python
# Niveles de seguridad configurables
SECURITY_LEVELS = {
    'LOW': {'threshold': 0.3, 'actions': ['log']},
    'MEDIUM': {'threshold': 0.5, 'actions': ['log', 'notify']},
    'HIGH': {'threshold': 0.7, 'actions': ['log', 'notify', 'rate_limit']},
    'CRITICAL': {'threshold': 0.9, 'actions': ['log', 'notify', 'block']}
}
```

## 🛠️ Desarrollo y Extensión

### Agregar Nuevo Conector (ANCHOR)
```python
class NewConnector:
    async def fetch_data(self, config):
        # Implementar lógica de conexión
        return {"data": [], "sources": ["new_source"]}

# Registrar en AnchorClient
self.connectors['new_source'] = NewConnector()
```

### Agregar Nuevo Detector (CENSOR)
```python
class NewAnomalyDetector:
    def detect(self, data):
        # Implementar detección específica
        return {"anomalies": [], "score": 0.0}

# Registrar en CensorClient
self.detectors['new_type'] = NewAnomalyDetector()
```

### Agregar Nuevo Modelo (NOESIS)
```python
class NewForecastingModel:
    def predict(self, data, horizon):
        # Implementar modelo predictivo
        return {"predictions": [], "confidence": []}

# Registrar en NoesisClient
self.models['new_model'] = NewForecastingModel()
```

### Hooks de Procesamiento
```python
class OmnIAEngineCoordinator:
    async def _pre_anchor_hook(self, request):
        """Hook ejecutado antes de ANCHOR"""
        pass
    
    async def _post_anchor_hook(self, request, anchor_result):
        """Hook ejecutado después de ANCHOR"""
        pass
    
    async def _pre_censor_hook(self, request, anchor_data):
        """Hook ejecutado antes de CENSOR"""
        pass
```

## 📚 Documentación Adicional

### APIs de Módulos
- **[ANCHOR API](/omnia-anchor-module/README.md)** - Documentación completa del módulo de ingesta
- **[CENSOR API](/omnia-censor-module/README.md)** - Documentación del módulo de supervisión ML
- **[NOESIS API](/omnia-noesis-module/README.md)** - Documentación del módulo de forecasting

### Protocolo de Seguridad
- **[OMNIA PROTOCOL](/docs/omnia_protocol_security_v1.md)** - Especificación completa del protocolo
- **[Prompts Especializados](/docs/omnia_specialized_prompts_with_security_v1.md)** - Prompts con seguridad integrada

## 🤝 Contribución

### Flujo de Desarrollo
1. **Fork** del repositorio
2. **Crear branch** para feature: `git checkout -b feature/nueva-funcionalidad`
3. **Implementar cambios** con tests
4. **Ejecutar tests**: `python3 test_omnia_coordinator.py`
5. **Crear Pull Request** con descripción detallada

### Estándares de Código
- **Type hints** obligatorios
- **Docstrings** en español
- **Tests** para nuevas funcionalidades
- **Logging** descriptivo
- **Manejo de errores** robusto

## 📄 Licencia

Parte del ecosistema OMNIA - Todos los derechos reservados.

---

**🎯 Desarrollado por MiniMax Agent** - Coordinador del Ecosistema OMNIA

**📞 Soporte**: Para issues o preguntas, crear un issue en el repositorio.

**🔄 Versión**: 1.0.0 (Noviembre 2024)

**🌐 Servidor**: http://localhost:8004 (por defecto)