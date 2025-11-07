# NOESIS Prediction APIs - Documentación

## Descripción General

Las NOESIS Prediction APIs proporcionan un sistema completo de predicciones para el ecosistema OMNIA, con integración a otros módulos como ANCHOR (datos históricos) y CENSOR (validación de predicciones).

## Características Principales

### 🚀 APIs REST
- **Predicción Individual**: `/predictions/single`
- **Predicciones en Lote**: `/predictions/batch`
- **Consulta de Estado**: `/predictions/{id}` y `/predictions/batch/{id}`

### 📊 Tipos de Predicción
- **Forecasting**: Predicción de series temporales
- **Demand**: Predicción de demanda
- **Trends**: Análisis de tendencias

### 🕐 Horizontes Temporales
- **Corto**: 1-7 días
- **Medio**: 1-4 semanas
- **Largo**: 1-12 meses

### 🗄️ Sistema de Cache
- Cache con Redis para predicciones frecuentes
- Estadísticas de cache (hits/misses)
- Invalidación por patrón

### 🔔 Sistema de Webhooks
- Registro de endpoints de notificación
- Eventos: `prediction_completed`, `prediction_failed`, `validation_required`
- Historial de envíos

### 🔐 Seguridad
- Autenticación JWT
- Rate limiting configurable
- Validación de permisos por roles

## Instalación

### Opción 1: Instalación Manual

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd noesis-prediction-apis

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env según necesidades

# 4. Iniciar servicios
./start.sh dev
```

### Opción 2: Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f noesis-prediction-api

# Detener servicios
docker-compose down
```

## Configuración

### Variables de Entorno Principales

| Variable | Descripción | Por Defecto |
|----------|-------------|-------------|
| `REDIS_HOST` | Host de Redis | localhost |
| `REDIS_PORT` | Puerto de Redis | 6379 |
| `JWT_SECRET` | Clave secreta JWT | omni-secret-key-2025 |
| `RATE_LIMIT_REQUESTS` | Límite de requests por hora | 100 |
| `CACHE_TTL` | Tiempo de vida del cache (segundos) | 3600 |
| `ANCHOR_API_URL` | URL de ANCHOR | http://anchor:8000 |
| `CENSOR_API_URL` | URL de CENSOR | http://censor:8000 |

## Uso de la API

### 1. Autenticación

```bash
# Obtener token JWT
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

### 2. Predicción Individual

```bash
curl -X POST "http://localhost:8000/predictions/single" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "forecasting",
    "horizon": "short",
    "data_source": "hybrid",
    "parameters": {
      "seasonal_period": 7
    },
    "confidence_level": 0.95
  }'
```

### 3. Predicciones en Lote

```bash
curl -X POST "http://localhost:8000/predictions/batch" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "batch-001",
    "requests": [
      {
        "type": "forecasting",
        "horizon": "short"
      },
      {
        "type": "demand",
        "horizon": "medium"
      }
    ]
  }'
```

### 4. Registro de Webhook

```bash
curl -X POST "http://localhost:8000/webhooks/register" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://my-api.com/webhooks/noesis",
    "events": ["prediction_completed", "prediction_failed"],
    "secret": "my-webhook-secret",
    "active": true
  }'
```

## Endpoints Disponibles

### Autenticación
- `POST /auth/login` - Autenticación de usuario

### Predicciones
- `POST /predictions/single` - Crear predicción individual
- `GET /predictions/{prediction_id}` - Obtener predicción por ID
- `POST /predictions/batch` - Crear predicciones en lote
- `GET /predictions/batch/{batch_id}` - Obtener estado de lote

### Webhooks
- `POST /webhooks/register` - Registrar webhook
- `DELETE /webhooks/{webhook_id}` - Desregistrar webhook
- `GET /webhooks/history` - Historial de webhooks

### Monitoreo
- `GET /health` - Health check
- `GET /metrics` - Métricas del sistema
- `DELETE /cache/clear` - Limpiar cache
- `GET /docs/types` - Tipos de predicción disponibles

## Modelos de Datos

### PredictionRequest
```json
{
  "type": "forecasting",     // forecasting|demand|trends
  "horizon": "short",        // short|medium|long
  "data_source": "hybrid",   // anchor|external|hybrid
  "parameters": {},          // Parámetros específicos
  "historical_period_days": 90,
  "confidence_level": 0.95
}
```

### PredictionResponse
```json
{
  "prediction_id": "uuid",
  "type": "forecasting",
  "status": "completed",
  "created_at": "2024-11-06T20:52:50",
  "data": {},
  "confidence_interval": {},
  "validation_status": "approved",
  "cached": false
}
```

## Integración con OMNIA

### ANCHOR (Datos Históricos)
- Obtención automática de datos históricos
- Soporte para múltiples fuentes de datos
- Cache inteligente de datos históricos

### CENSOR (Validación)
- Validación automática de predicciones
- Clasificación: approved|warning|rejected|error
- Cálculo de scores de confianza

## Monitoreo y Observabilidad

### Métricas Disponibles
- Estadísticas de cache (hits/misses)
- Requests por cliente (rate limiting)
- Estado de webhooks
- Salud de componentes

### Health Check
```bash
curl http://localhost:8000/health
```

### Endpoints de Debug
- `/docs` - Documentación Swagger UI
- `/docs/openapi.json` - Especificación OpenAPI
- `/openapi.json` - Schema JSON

## Ejemplos de Uso

### Ejemplo 1: Predicción de Ventas Semanal

```bash
# Autenticación
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')

# Crear predicción
curl -X POST "http://localhost:8000/predictions/single" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "demand",
    "horizon": "short",
    "data_source": "hybrid",
    "parameters": {
      "product_category": "electronics",
      "seasonal_adjustment": true
    },
    "historical_period_days": 90,
    "confidence_level": 0.95
  }'
```

### Ejemplo 2: Análisis de Tendencias Mensual

```bash
curl -X POST "http://localhost:8000/predictions/single" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "trends",
    "horizon": "long",
    "data_source": "anchor",
    "parameters": {
      "analysis_window": 180,
      "significance_level": 0.05
    }
  }'
```

## Troubleshooting

### Problemas Comunes

1. **Error de conexión a Redis**
   - Verificar que Redis esté ejecutándose
   - Revisar variables de entorno `REDIS_HOST` y `REDIS_PORT`
   - El sistema usará cache en memoria como fallback

2. **Rate Limiting activado**
   - Verificar límite actual con `/metrics`
   - Ajustar `RATE_LIMIT_REQUESTS` si es necesario

3. **Errores de autenticación**
   - Verificar token JWT válido
   - Revisar fecha de expiración
   - Verificar usuario y contraseña

4. **Webhooks no llegan**
   - Verificar URL accesible
   - Revisar historial en `/webhooks/history`
   - Verificar secrets correctos

### Logs

```bash
# Ver logs en tiempo real
tail -f logs/startup.log

# Logs de Docker
docker-compose logs -f noesis-prediction-api
```

## Rendimiento

### Optimizaciones
- Cache de predicciones frecuentes
- Procesamiento asíncrono
- Rate limiting por cliente
- Webhooks no bloqueantes

### Escalabilidad
- Workers múltiples configurables
- Cache distribuido con Redis
- Base de datos para persistencia (futuro)

## Seguridad

### Mejores Prácticas
1. **Variables de entorno**: Nunca hardcodear secrets
2. **JWT Secret**: Cambiar en producción
3. **Rate Limiting**: Ajustar según necesidades
4. **CORS**: Configurar orígenes permitidos
5. **HTTPS**: Usar en producción

### Usuarios por Defecto
- **admin/admin123**: Acceso completo
- **user/user123**: Solo lectura

## Contribución

### Estructura del Código
- `noesis_prediction_apis.py`: API principal
- `cache.py`: Sistema de cache
- `models.py`: Modelos de datos
- `webhooks.py`: Sistema de notificaciones

### Testing
```bash
# Tests unitarios (cuando estén implementados)
pytest tests/

# Test de integración
curl -X GET http://localhost:8000/health
```

## Licencia

OMNIA - Sistema de Predicciones v1.0.0