# MIDAS Platform Integration System

## Descripción

Sistema de integración unificada para Google Ads y Meta Ads que proporciona gestión cross-platform avanzada, optimización automática y análisis consolidado de campañas publicitarias.

## Características Principales

### 1. APIs Unificadas
- **UnifiedAdsManager**: Interfaz única para gestionar campañas en ambas plataformas
- Integración transparente con Google Ads API y Meta Ads API
- Manejo automático de autenticación y rate limits

### 2. Sincronización Bidireccional
- Sincronización automática de campañas entre Google Ads y Meta Ads
- Mapeo inteligente de targeting y configuraciones
- Respaldo y restauración de configuraciones de campaña

### 3. Mapeo Automático de Targeting
- **TargetingMapper**: Conversión automática entre formatos de targeting
- Soporte para edad, género, intereses, comportamientos, ubicaciones e idiomas
- Cálculo de similitud entre configuraciones de targeting

### 4. Gestión Unificada de Presupuestos
- **CrossPlatformOptimizer**: Redistribución inteligente de presupuestos
- Asignación basada en performance (ROAS, CTR, CPC)
- Estrategias: performance_based, equal, conservative

### 5. Reporting Consolidado
- Métricas unificadas cross-platform
- KPIs consolidados (CPA, ROAS, efficiency score)
- Insights automáticos de performance

### 6. Sistema de Alertas
- **AlertingSystem**: Monitoreo automático de performance
- Alertas por CTR bajo, CPC alto, caída de performance
- Notificaciones por webhook configurables

### 7. Optimización Cross-Platform
- Budget shifting automático basado en performance
- Rebalanceo dinámico de presupuestos
- Optimización basada en machine learning

### 8. Detección de Overlap de Audiencias
- **AudienceManager**: Análisis de solapamiento entre plataformas
- Identificación de audiencias únicas y compartidas
- Creación de audiencias unificadas

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    MIDAS Platform Integration               │
├─────────────────────────────────────────────────────────────┤
│  UnifiedAdsManager (Orquestador Principal)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Google Ads  │  │  Meta Ads    │  │   Targeting      │   │
│  │ Connector   │  │  Connector   │  │     Mapper       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ CrossPlat.  │  │  Audience    │  │    Alerting      │   │
│  │ Optimizer   │  │   Manager    │  │     System       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     Base de Datos OMNIA                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     ANCHOR Schema + MIDAS Extensions               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Instalación y Configuración

### Dependencias

```python
# Instalar dependencias requeridas
pip install asyncio logging json uuid hashlib datetime typing
pip install dataclasses enum collections statistics numpy
pip install psycopg2-binary requests concurrent.futures
```

### Configuración de Base de Datos

Crear tabla para audiencias unificadas:

```sql
CREATE TABLE IF NOT EXISTS midas_unified_audiences (
    audience_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    targeting JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(audience_id)
);
```

### Configuración de Credenciales

#### Google Ads
```python
google_config = GoogleAdsConfig(
    customer_id="1234567890",
    developer_token="your_developer_token",
    client_id="your_client_id",
    client_secret="your_client_secret",
    refresh_token="your_refresh_token"
)
```

#### Meta Ads
```python
meta_config = MetaAdsConfig(
    access_token="your_access_token",
    app_id="your_app_id",
    app_secret="your_app_secret",
    ad_account_id="123456789"
)
```

## Guía de Uso

### 1. Inicialización del Sistema

```python
import asyncio
from midas_platform_integration import UnifiedAdsManager, GoogleAdsConfig, MetaAdsConfig

# Configurar credenciales
google_config = GoogleAdsConfig(...)
meta_config = MetaAdsConfig(...)

# Crear manager unificado
manager = UnifiedAdsManager(
    db_connection="postgresql://user:pass@localhost/anchor",
    google_ads_config=google_config,
    meta_ads_config=meta_config,
    redis_connection="redis://localhost:6379"
)

# Inicializar
await manager.initialize()
```

### 2. Health Check

```python
# Verificar estado del sistema
health = await manager.health_check()
print(health)
```

### 3. Creación de Campaña Unificada

```python
campaign_config = {
    'name': 'Campaña Cross-Platform Test',
    'budget': 1000,
    'targeting': {
        'age_range': '25-54',
        'genders': ['male', 'female'],
        'interests': ['Technology', 'Business'],
        'locations': ['US']
    },
    'creatives': [
        {'type': 'image', 'url': 'https://example.com/image.jpg'},
        {'type': 'video', 'url': 'https://example.com/video.mp4'}
    ]
}

result = await manager.create_unified_campaign(campaign_config)
```

### 4. Sincronización Bidireccional

```python
# Sincronizar campañas existentes
google_campaigns = ['google_123', 'google_456']
meta_campaigns = ['meta_789', 'meta_101']

sync_result = await manager.sync_campaigns_bidirectional(
    google_campaigns=google_campaigns,
    meta_campaigns=meta_campaigns
)
```

### 5. Reporte Consolidado

```python
# Generar reporte de performance
campaign_ids = ['campaign_1', 'campaign_2', 'campaign_3']
report = await manager.get_consolidated_report(
    campaign_ids=campaign_ids,
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"CTR Consolidado: {report['kpis']['consolidated_ctr']}")
print(f"ROAS: {report['kpis']['return_on_ad_spend']}")
print(f"Grade: {report['kpis']['performance_grade']}")
```

### 6. Optimización Cross-Platform

```python
# Optimizar asignación de presupuesto
optimization_config = {
    'total_budget': 2000,
    'optimization_rules': {
        'strategy': 'performance_based'
    },
    'analysis_days': 30
}

optimization_result = await manager.optimize_cross_platform(
    campaign_ids=campaign_ids,
    optimization_config=optimization_config
)

# Aplicar redistribución de presupuesto
budget_allocation = optimization_result['budget_optimization']
print(f"Google: ${budget_allocation['google_allocation']}")
print(f"Meta: ${budget_allocation['meta_allocation']}")
```

### 7. Configuración de Monitoreo

```python
# Configurar sistema de alertas
alert_config = {
    'low_ctr_threshold': 1.5,        # Alert si CTR < 1.5%
    'high_cpc_threshold': 4.0,       # Alert si CPC > $4.00
    'low_conversion_rate': 2.5,      # Alert si conversion rate < 2.5%
    'budget_utilization': 0.85,      # Alert si budget > 85%
    'performance_drop': 0.25         # Alert si performance cae > 25%
}

monitoring_result = await manager.setup_monitoring(alert_config)
```

### 8. Análisis de Audiencias

```python
# Detectar overlap de audiencias
from midas_platform_integration import AudienceManager

audience_manager = AudienceManager(db_connection)

overlap_analysis = await audience_manager.detect_audience_overlap(
    google_campaign_ids=['google_123', 'google_456'],
    meta_campaign_ids=['meta_789', 'meta_101']
)

print(f"Overlap: {overlap_analysis.overlap_percentage}%")
print(f"Eficiencia: {overlap_analysis.efficiency_score}%")
```

## APIs Principales

### UnifiedAdsManager

| Método | Descripción | Parámetros |
|--------|-------------|------------|
| `initialize()` | Inicializa conectores | - |
| `create_unified_campaign()` | Crea campaña en ambas plataformas | campaign_config |
| `sync_campaigns_bidirectional()` | Sincroniza campañas entre plataformas | google_campaign_ids, meta_campaign_ids |
| `get_consolidated_report()` | Genera reporte consolidado | campaign_ids, start_date, end_date |
| `optimize_cross_platform()` | Optimiza performance cross-platform | campaign_ids, optimization_config |
| `setup_monitoring()` | Configura sistema de alertas | alert_config |
| `health_check()` | Verifica estado del sistema | - |

### CrossPlatformOptimizer

| Método | Descripción | Parámetros |
|--------|-------------|------------|
| `analyze_performance()` | Analiza performance de campañas | campaign_ids, days |
| `optimize_budget_allocation()` | Optimiza asignación de presupuesto | campaign_ids, target_budget, rules |
| `shift_budget()` | Redistribuye presupuesto | current_allocation, direction, amount |

### AudienceManager

| Método | Descripción | Parámetros |
|--------|-------------|------------|
| `detect_audience_overlap()` | Detecta overlap entre audiencias | google_campaign_ids, meta_campaign_ids |
| `sync_audiences()` | Sincroniza audiencias entre plataformas | source_platform, target_platform, mappings |
| `create_unified_audience()` | Crea audiencia unificada | google_campaign_ids, meta_campaign_ids, name |

### TargetingMapper

| Método | Descripción | Parámetros |
|--------|-------------|------------|
| `map_google_to_meta()` | Convierte targeting Google → Meta | google_targeting |
| `map_meta_to_google()` | Convierte targeting Meta → Google | meta_targeting |
| `calculate_similarity()` | Calcula similitud entre targeting | targeting1, targeting2 |

## Casos de Uso

### 1. Migración de Campañas
```python
# Migrar campañas exitosas de Google Ads a Meta Ads
google_campaigns = await manager._get_google_campaigns()
migration_result = await manager.sync_campaigns_bidirectional(
    google_campaigns=[c['id'] for c in google_campaigns]
)
```

### 2. Optimización Automática Diaria
```python
# Optimización diaria automática
async def daily_optimization():
    active_campaigns = await manager._get_active_campaigns()
    
    # Análisis de performance
    analysis = await manager.optimizer.analyze_performance(
        [c['id'] for c in active_campaigns], days=7
    )
    
    # Reoptimización de presupuesto si es necesario
    if analysis['optimization_opportunities']:
        await manager.optimize_cross_platform([c['id'] for c in active_campaigns])
```

### 3. Detección de Audiencias Subutilizadas
```python
# Identificar audiencias con poco overlap
overlap = await audience_manager.detect_audience_overlap(
    google_campaigns, meta_campaigns
)

if overlap.efficiency_score < 70:
    # Crear audiencia unificada
    unified = await audience_manager.create_unified_audience(
        google_campaigns, meta_campaigns, "Audiencia Optimizada"
    )
```

## Métricas y KPIs

### Métricas Consolidadas
- **CTR Consolidado**: Clicks / Impressions × 100
- **CPC Promedio**: Spend Total / Clicks Totales
- **CPM**: Spend Total / Impressions × 1000
- **Tasa de Conversión**: Conversiones / Clicks × 100

### KPIs Avanzados
- **CPA (Costo por Adquisición)**: Spend Total / Conversiones
- **ROAS (Return on Ad Spend)**: Revenue / Spend Total
- **Efficiency Score**: Score composito de performance (0-100)
- **Performance Grade**: A, B, C, D, F basado en efficiency

### Alertas Configurables
- CTR bajo: < 1.5% por defecto
- CPC alto: > $4.00 por defecto  
- Conversión baja: < 2.5% por defecto
- Caída de performance: > 25% por defecto

## Integración con OMNIA

### ANCHOR Integration
- Sincronización automática con `anchor_data` table
- Formato compatible con schema existente
- Validación a través de ANCHORValidator

### Real-time Processing
- Integración con Redis para queues
- Procesamiento asíncrono de campañas
- Event-driven architecture

### Monitoring y Alertas
- Webhook notifications configurables
- Integration con sistema de alertas OMNIA
- Dashboard metrics en tiempo real

## Consideraciones de Performance

### Rate Limiting
- Google Ads: 10,000 requests/day
- Meta Ads: 200 calls/hour
- Automatic throttling implementado

### Caching
- Campaign cache en memoria
- Targeting mappings cacheados
- Performance metrics cacheadas

### Concurrency
- Async/await para operaciones I/O
- ThreadPoolExecutor para operaciones CPU-intensive
- Batch processing para mejorar throughput

## Seguridad y Compliance

### Autenticación
- OAuth2 para Google Ads
- Access Tokens para Meta Ads
- Refresh token management automático

### Data Protection
- No storage de PII en logs
- Hash-based audit trails
- Compliance con GDPR/CCPA

### Rate Limiting Protection
- Automatic backoff en errores de rate limit
- Queue management para burst traffic
- Graceful degradation en high load

## Roadmap y Extensiones

### Próximas Características
- [ ] A/B Testing automático
- [ ] Machine Learning para optimization
- [ ] Integración con más plataformas (LinkedIn, TikTok)
- [ ] Real-time dashboard
- [ ] API REST completa
- [ ] GraphQL endpoints

### Mejoras de Performance
- [ ] Parallel campaign processing
- [ ] Advanced caching strategies
- [ ] Database optimization
- [ ] Microservices architecture

## Soporte y Contacto

Para soporte técnico y consultas:
- Documentación: `/workspace/midas_platform_integration.py`
- Logs: Sistema de logging integrado
- Health checks: `manager.health_check()`

---

**Versión**: 1.0  
**Fecha**: 2025-11-06  
**Autor**: OMNIA System  
**Licencia**: Propietaria OMNIA
