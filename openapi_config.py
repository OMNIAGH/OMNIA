"""
Configuración adicional para la documentación Swagger
Este archivo puede ser usado para mejorar la documentación de la API
"""

from fastapi.openapi.utils import get_openapi
from noesis_prediction_apis import app

def custom_openapi():
    """Generar OpenAPI schema personalizado"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="OMNIA NOESIS - Prediction APIs",
        version="1.0.0",
        description="""
        Sistema completo de APIs REST para predicciones del ecosistema OMNIA.
        
        ## Características Principales
        
        ### 🎯 Tipos de Predicción
        - **Forecasting**: Predicción de series temporales y valores futuros
        - **Demand**: Predicción de demanda con factores estacionales
        - **Trends**: Análisis de tendencias y patrones
        
        ### 🕐 Horizontes Temporales
        - **Corto**: 1-7 días para decisiones inmediatas
        - **Medio**: 1-4 semanas para planificación táctica
        - **Largo**: 1-12 meses para planificación estratégica
        
        ### 🔗 Integraciones
        - **ANCHOR**: Sistema de datos históricos
        - **CENSOR**: Validación y calidad de predicciones
        - **Webhooks**: Notificaciones en tiempo real
        
        ### 🚀 Características Técnicas
        - Cache inteligente con Redis
        - Procesamiento en lote
        - Rate limiting por cliente
        - Autenticación JWT
        - Documentación automática Swagger
        
        ## Autenticación
        
        1. Obtener token: `POST /auth/login`
        2. Usar en requests: `Authorization: Bearer <token>`
        
        ## Ejemplo de Uso
        
        ```bash
        # 1. Autenticación
        curl -X POST "http://localhost:8000/auth/login" \\
          -H "Content-Type: application/json" \\
          -d '{"username": "admin", "password": "admin123"}'
        
        # 2. Crear predicción
        curl -X POST "http://localhost:8000/predictions/single" \\
          -H "Authorization: Bearer <token>" \\
          -H "Content-Type: application/json" \\
          -d '{
            "type": "forecasting",
            "horizon": "short",
            "confidence_level": 0.95
          }'
        ```
        
        ## Errores Comunes
        
        - `401`: Token expirado o inválido
        - `429`: Límite de requests excedido
        - `500`: Error interno del servidor
        
        Para más información, consulte la documentación en `/docs`.
        """,
        routes=app.routes,
    )
    
    # Agregar información de contacto
    openapi_schema["info"]["contact"] = {
        "name": "OMNIA Team",
        "email": "omnia@company.com",
        "url": "https://omnia.company.com"
    }
    
    # Agregar licencia
    openapi_schema["info"]["license"] = {
        "name": "OMNIA License",
        "url": "https://omnia.company.com/license"
    }
    
    # Agregar servidores
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Servidor de desarrollo"
        },
        {
            "url": "https://api.omnia.company.com/predictions",
            "description": "Servidor de producción"
        }
    ]
    
    # Agregar tags con descripción
    openapi_schema["tags"] = [
        {
            "name": "Authentication",
            "description": "Endpoints de autenticación y autorización"
        },
        {
            "name": "Predictions",
            "description": "Endpoints para crear y consultar predicciones"
        },
        {
            "name": "Batch Operations",
            "description": "Operaciones de predicción en lote"
        },
        {
            "name": "Webhooks",
            "description": "Gestión de notificaciones webhooks"
        },
        {
            "name": "Monitoring",
            "description": "Endpoints de monitoreo y métricas"
        },
        {
            "name": "Cache",
            "description": "Gestión del sistema de cache"
        }
    ]
    
    # Personalizar respuestas de error
    openapi_schema["components"]["responses"] = {
        "UnauthorizedError": {
            "description": "Token JWT inválido o expirado",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string", "example": "Token expirado"}
                        }
                    }
                }
            }
        },
        "RateLimitExceeded": {
            "description": "Límite de requests excedido",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string", "example": "Límite de requests excedido"}
                        }
                    }
                }
            }
        },
        "ValidationError": {
            "description": "Error de validación en los datos de entrada",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string", "example": "Error de validación"}
                        }
                    }
                }
            }
        }
    }
    
    # Personalizar esquemas
    openapi_schema["components"]["schemas"]["PredictionRequest"]["example"] = {
        "type": "forecasting",
        "horizon": "short",
        "data_source": "hybrid",
        "parameters": {
            "seasonal_period": 7,
            "trend_factor": 1.0,
            "regression_method": "linear"
        },
        "historical_period_days": 90,
        "confidence_level": 0.95
    }
    
    openapi_schema["components"]["schemas"]["PredictionResponse"]["example"] = {
        "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
        "type": "forecasting",
        "status": "completed",
        "created_at": "2025-11-06T20:52:50Z",
        "data": {
            "predictions": [95.5, 96.2, 94.8, 97.1, 96.9, 95.7, 96.4],
            "dates": [
                "2025-11-07",
                "2025-11-08",
                "2025-11-09",
                "2025-11-10",
                "2025-11-11",
                "2025-11-12",
                "2025-11-13"
            ],
            "confidence_interval": {
                "lower": [92.0, 92.7, 91.3, 93.6, 93.4, 92.2, 92.9],
                "upper": [99.0, 99.7, 98.3, 100.6, 100.4, 99.2, 99.9]
            },
            "confidence": 0.89,
            "metrics": {
                "r2_score": 0.85,
                "mae": 12.5,
                "rmse": 15.2
            }
        },
        "validation_status": "approved",
        "cached": False
    }
    
    # Configurar seguridad JWT
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Token JWT obtenido del endpoint de autenticación"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Aplicar la configuración personalizada
if app:
    app.openapi = custom_openapi