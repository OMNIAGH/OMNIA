# MIDAS Google OPAL - Documentación Técnica

## Resumen Ejecutivo

El sistema MIDAS Google OPAL es una plataforma integral de gestión automatizada de Google Ads que incorpora:

- **Integración completa** con Google Ads API v14
- **Optimización inteligente** basada en machine learning
- **Monitoreo en tiempo real** de Quality Scores
- **Dashboard interactivo** con métricas de performance
- **Automatización completa** de procesos de optimización

## Arquitectura del Sistema

### Componentes Principales

#### 1. GoogleAdsManager
```python
class GoogleAdsManager:
    - Gestión completa de Google Ads API v14
    - Operaciones CRUD de campañas, ad groups, keywords y ads
    - Sincronización con Google Analytics 4
    - Extracción de métricas y KPIs
```

**Funcionalidades Clave:**
- `get_active_campaigns()` - Lista campañas activas
- `create_campaign()` - Crea nuevas campañas
- `update_campaign_status()` - Pausa/activa campañas
- `get_campaign_metrics()` - Extrae métricas de performance
- `sync_with_google_analytics()` - Correlaciona datos GA4

#### 2. CampaignOptimizer
```python
class CampaignOptimizer:
    - Optimización automática de pujas
    - Gestión inteligente de negative keywords
    - Creación automática de anuncios
    - Ajuste dinámico de presupuestos
```

**Algoritmos de Optimización:**
- **Bid Optimization:** Ajusta pujas basándose en ROAS, CTR y Quality Score
- **Negative Keywords:** Bloquea términos irrelevantes automáticamente
- **Ad Creation:** Genera anuncios desde templates predefinidos
- **Budget Adjustment:** Optimiza distribución de presupuesto

#### 3. QualityScoreMonitor
```python
class QualityScoreMonitor:
    - Monitoreo continuo de Quality Scores
    - Generación de alertas automáticas
    - Recomendaciones de mejora
    - Historial y tendencias
```

**Características:**
- **Real-time Monitoring:** Verifica QS cada hora
- **Alert System:** Notificaciones por Quality Score bajo
- **Recommendations Engine:** Sugiere acciones de mejora
- **Historical Analysis:** Mantiene historial de puntuaciones

#### 4. RealTimeDashboard
```python
class RealTimeDashboard:
    - Métricas en tiempo real
    - Visualización de tendencias
    - Alertas y notificaciones
    - Predicciones de performance
```

**Visualizaciones Disponibles:**
- Overview general de campañas
- Tendencias semanales
- Top/Worst performing campaigns
- Alertas activas
- Predicciones con ML básico

## Flujo de Trabajo Automático

### 1. Inicialización
```python
config = {
    'customer_id': '1234567890',
    'developer_token': 'token',
    'client_id': 'client_id',
    'client_secret': 'client_secret',
    'refresh_token': 'refresh_token'
}

system = await initialize_midas_google_opal(config)
```

### 2. Optimización Automática
```python
# Se ejecuta cada 6 horas
await system['campaign_optimizer'].optimize_campaigns()
```

**Proceso de Optimización:**
1. Evalúa métricas de todas las campañas activas
2. Aplica reglas de optimización configuradas
3. Ajusta pujas basándose en performance
4. Agrega negative keywords si es necesario
5. Pausa anuncios de bajo rendimiento
6. Crea nuevos anuncios desde templates
7. Ajusta presupuestos dinámicamente

### 3. Monitoreo Continuo
```python
# Se ejecuta cada hora
await system['quality_monitor'].monitor_quality_scores()
```

**Proceso de Monitoreo:**
1. Extrae Quality Scores de todas las keywords
2. Compara con umbrales configurados
3. Genera alertas para scores bajos
4. Almacena datos en historial
5. Genera recomendaciones de mejora

## Configuración y Reglas de Negocio

### Reglas de Optimización (optimization_rules.json)

#### Bid Optimization Rules
```json
{
  "rule_name": "Alto ROAS - Aumentar Pujas",
  "conditions": {
    "min_roas": 4.0,
    "min_conversions": 10
  },
  "actions": {
    "optimize_bids": true,
    "bid_increase_percentage": 0.2
  }
}
```

#### Negative Keywords Rules
```json
{
  "rule_name": "Bloquear Términos Irrelevantes",
  "conditions": {
    "min_impressions": 1000,
    "click_through_rate_threshold": 0.01
  },
  "actions": {
    "add_negative_keywords": ["gratis", "free", "download"]
  }
}
```

### Templates de Anuncios (ad_templates.json)

#### Estructura de Template
```json
{
  "template_name": "Search Ads - Standard",
  "elements": {
    "headlines": [
      {
        "text": "Oferta Especial - {keyword} - {promotion_text}",
        "pinned": "HEADLINE_1"
      }
    ],
    "descriptions": [
      {
        "text": "Encuentra las mejores ofertas en {keyword}",
        "order": 1
      }
    ]
  }
}
```

**Variables Disponibles:**
- `{keyword}` - Palabra clave principal
- `{category}` - Categoría del producto
- `{location}` - Ubicación geográfica
- `{price}` - Precio específico
- `{promotion_text}` - Texto promocional

## Integración con Google Analytics 4

### Configuración
```python
ga4_integration = GoogleAnalyticsIntegration(
    property_id='GA4_PROPERTY_ID',
    credentials_path='path_to_credentials.json'
)
```

### Métricas Sincronizadas
- **Conversiones:** Compra, registro, descarga, etc.
- **Revenue:** Valor monetario de conversiones
- **Session Data:** Fuente, medio, campaign
- **User Behavior:** Tiempo en sitio, páginas vistas

### Correlación de Datos
El sistema correlaciona datos de Google Ads con GA4 usando:
- **Custom Dimensions:** Para tracking detallado
- **Attribution Models:** Para mejor análisis de conversión
- **Cross-platform Metrics:** Para análisis holístico

## Base de Datos y Almacenamiento

### Esquema de Base de Datos (SQLite)

#### Tabla: campaign_metrics
```sql
CREATE TABLE campaign_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    campaign_name TEXT NOT NULL,
    date TEXT NOT NULL,
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    cost_micros INTEGER DEFAULT 0,
    conversions INTEGER DEFAULT 0,
    conversion_value REAL DEFAULT 0.0,
    ctr REAL DEFAULT 0.0,
    cpc REAL DEFAULT 0.0,
    cpa REAL DEFAULT 0.0,
    roas REAL DEFAULT 0.0,
    quality_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'ENABLED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Tabla: quality_score_history
```sql
CREATE TABLE quality_score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    ad_group_id TEXT,
    keyword_id TEXT,
    quality_score REAL NOT NULL,
    date TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Tabla: optimization_rules
```sql
CREATE TABLE optimization_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    conditions TEXT NOT NULL,
    actions TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## APIs y Endpoints

### Google Ads API v14

#### Operaciones de Campaña
```python
# Crear campaña
campaign_id = await google_ads.create_campaign(campaign_data)

# Actualizar estado
await google_ads.update_campaign_status(campaign_id, 'PAUSED')

# Obtener métricas
metrics = await google_ads.get_campaign_metrics(campaign_id)
```

#### Operaciones de Keywords
```python
# Obtener keywords
keywords = await google_ads.get_keywords(ad_group_id)

# Actualizar puja
await google_ads.update_keyword_bid(keyword_id, new_bid)

# Agregar negative keyword
await google_ads.add_negative_keyword(campaign_id, 'negative_term')
```

#### Operaciones de Anuncios
```python
# Crear anuncio
result = await google_ads.create_ad(ad_group_id, ad_data)

# Pausar anuncio
await google_ads.pause_ad(ad_id)
```

### Google Analytics 4 API

#### Extracción de Datos
```python
# Obtener conversiones
ga4_data = await ga4_integration.get_conversion_data(start_date, end_date)

# Sincronizar con Google Ads
sync_result = await google_ads.sync_with_google_analytics(start_date, end_date)
```

## Configuración de Autenticación

### OAuth2 Setup
1. **Google Cloud Console:** Crear proyecto
2. **APIs:** Habilitar Google Ads API y GA4 API
3. **Credentials:** Crear OAuth 2.0 Client ID
4. **Authentication:** Configurar refresh token

### Archivos de Configuración

#### google_ads_config.json
```json
{
  "developer_token": "tu_developer_token",
  "client_id": "tu_client_id",
  "client_secret": "tu_client_secret",
  "refresh_token": "tu_refresh_token",
  "login_customer_id": "tu_login_customer_id",
  "use_proto_plus": true
}
```

#### Variables de Entorno
```bash
export GOOGLE_ADS_DEVELOPER_TOKEN="token"
export GOOGLE_ADS_CLIENT_ID="client_id"
export GOOGLE_ADS_CLIENT_SECRET="client_secret"
export GOOGLE_ADS_REFRESH_TOKEN="refresh_token"
export GOOGLE_ADS_CUSTOMER_ID="customer_id"
export GA4_PROPERTY_ID="property_id"
```

## Monitoreo y Alertas

### Sistema de Alertas
- **Quality Score Bajo:** QS < 6.0
- **Performance Degradado:** CTR < 1%, CPC > €5
- **Optimizaciones Aplicadas:** Tracking de cambios
- **Errores de API:** Fallos de conexión o rate limits

### Métricas de Sistema
- **API Response Times:** Latencia de llamadas
- **Success Rates:** Porcentaje de operaciones exitosas
- **Optimization Frequency:** Frecuencia de optimizaciones
- **Alert Volume:** Cantidad de alertas generadas

## Optimización de Performance

### Concurrencia
- **ThreadPoolExecutor:** Paralelización de optimizaciones
- **AsyncIO:** Operaciones no bloqueantes
- **Queue-based Processing:** Procesamiento asíncrono de alertas

### Caching
- **In-memory Cache:** Para métricas frecuentes
- **Database Cache:** Para datos históricos
- **API Rate Limiting:** Respeto de límites de Google

### Monitoring de Recursos
- **Memory Usage:** Seguimiento de consumo de RAM
- **Database Size:** Control de crecimiento de DB
- **Log Rotation:** Gestión automática de logs

## Seguridad y Compliance

### Protección de Credenciales
- **Environment Variables:** Para datos sensibles
- **File Permissions:** Control de acceso a archivos
- **Token Rotation:** Renovación automática de tokens

### GDPR Compliance
- **Data Minimization:** Solo datos necesarios
- **Anonymization:** Datos agregados para análisis
- **Data Retention:** Políticas de retención configurables

## Extensibilidad

### Plugin Architecture
```python
class CustomOptimizer:
    async def optimize(self, campaign_data):
        # Lógica personalizada de optimización
        pass

# Registrar plugin
system.register_optimizer('custom', CustomOptimizer())
```

### Custom Rules Engine
```python
class RuleEngine:
    def add_custom_rule(self, rule_function):
        # Agregar regla personalizada
        pass
```

## Troubleshooting

### Errores Comunes
1. **API Rate Limits:** Implementar retry logic
2. **Authentication Issues:** Verificar tokens y permisos
3. **Data Inconsistency:** Validar correlaciones de datos
4. **Performance Issues:** Optimizar queries de base de datos

### Logs y Debugging
- **File Logging:** `midas_google_opal.log`
- **Console Output:** Para desarrollo
- **Error Tracking:** Con timestamps y contexto
- **Performance Metrics:** Para optimización

## Roadmap y Futuras Mejoras

### Versión 2.1
- **Machine Learning Avanzado:** Modelos predictivos más sofisticados
- **Real-time Bidding:** Ajuste de pujas en tiempo real
- **Advanced Analytics:** Análisis de cohortes y atribución

### Versión 2.2
- **Multi-platform Support:** Integración con Facebook Ads, LinkedIn
- **A/B Testing Automático:** Testing continuo de creativos
- **Attribution Modeling:** Modelos de atribución avanzados

### Versión 3.0
- **AI-Powered Creative Generation:** Generación automática de creativos
- **Voice Search Optimization:** Optimización para búsqueda por voz
- **Privacy-First Architecture:** Enfoque en privacidad por diseño

---

**© 2025 MIDAS Google OPAL - Sistema de Gestión Automatizada de Google Ads**