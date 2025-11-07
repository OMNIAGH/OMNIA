# NOESIS - Sistema de Modelos de Forecasting Predictivo

## Descripción

Sistema completo de forecasting predictivo que combina múltiples modelos estadísticos y de machine learning para proporcionar predicciones precisas de series temporales. Desarrollado para NOESIS con capacidades avanzadas de validación, ensemble methods y API en tiempo real.

## Características Principales

### 🤖 Modelos Implementados

**Modelos Estadísticos:**
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA para datos estacionales
- **Prophet**: Modelo de Facebook para forecasting

**Modelos de Machine Learning:**
- **XGBoost**: Gradient Boosting optimizado
- **LightGBM**: Gradient Boosting eficiente
- **Random Forest**: Ensemble de árboles de decisión

**Ensemble Methods:**
- Weighted averaging con pesos optimizados
- Stacking methods
- Voting methods

### 🔧 Capacidades Avanzadas

- **Auto-optimización**: Selección automática de parámetros
- **Preprocesamiento**: Manejo automático de outliers y valores faltantes
- **Detección de estacionalidad**: Identificación automática de patrones
- **Validación robusta**: Walk-forward y time series cross-validation
- **API tiempo real**: Predicciones en tiempo real
- **Análisis diagnóstico**: Monitoreo y análisis de performance

## Instalación

### Dependencias Requeridas

```bash
# Dependencias básicas
pip install numpy pandas scipy scikit-learn xgboost lightgbm
pip install statsmodels joblib matplotlib seaborn

# Prophet (opcional)
pip install prophet

# Para instalación completa
pip install -r requirements.txt
```

### Configuración

```python
from noesis_forecasting_models import ForecastingConfig, NoesisForecastingAPI

# Configuración básica
config = ForecastingConfig(
    test_size=0.2,
    validation_size=0.1,
    ensemble_method='weighted',
    walk_forward=True
)

# Configuración avanzada
config_advanced = ForecastingConfig(
    seasonal_period=7,  # Para datos semanales
    n_splits=10,
    xgb_params={
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05
    }
)
```

## Uso Básico

### Ejemplo 1: Forecasting Simple

```python
import pandas as pd
from noesis_forecasting_models import NoesisForecastingAPI, create_sample_data

# Crear API
api = NoesisForecastingAPI()

# Generar datos de ejemplo
data = create_sample_data(periods=365, frequency='D')

# Entrenar todos los modelos
results = api.train_all_models(data)

# Hacer predicción
predictions = api.predict_ensemble(steps=12)
print(f"Predicciones: {predictions['predictions']}")
```

### Ejemplo 2: Análisis Completo

```python
# Análisis de la serie temporal
analysis = api.analyze_series(data)

print(f"Estacionalidad detectada: {analysis['seasonality']['has_seasonality']}")
print(f"Período óptimo: {analysis['seasonality']['best_period']}")
print(f"Es estacionaria: {analysis['stationarity']['is_stationary']}")

# Preprocesamiento
preprocessor = DataPreprocessor()
data_clean = preprocessor.handle_missing_values(data, 'interpolate')
data_clean = preprocessor.handle_outliers(data_clean, 'winsorize')
```

### Ejemplo 3: Validación Avanzada

```python
from noesis_forecasting_models import Validator

validator = Validator(config)

# Validación walk-forward
validation_results = validator.walk_forward_validation(arima_model, data)
print(f"MAE promedio: {validation_results['mae_mean']:.2f}")

# Cross-validation temporal
cv_results = validator.time_series_cross_validation(xgb_model, data)
print(f"R² promedio: {cv_results['r2_scores_mean']:.3f}")
```

## API Reference

### NoesisForecastingAPI

#### Métodos Principales

```python
api = NoesisForecastingAPI(config)

# Entrenar modelos
results = api.train_all_models(data, preprocessed=False)

# Predicciones
pred = api.predict(model_name='ensemble', steps=12)
ensemble_pred = api.predict_ensemble(steps=12, method='weighted')

# Análisis
analysis = api.analyze_series(data)
info = api.get_model_info('arima')

# Guardado/carga
api.save_models('./modelos_noesis')
api.load_models('./modelos_noesis')
```

### Modelos Individuales

#### ARIMA
```python
from noesis_forecasting_models import ARIMAModel, ForecastingConfig

config = ForecastingConfig()
model = ARIMAModel(config, p=1, d=1, q=1)  # o p=None para auto-selección
model.fit(data)
predictions = model.predict(steps=12)
```

#### Prophet
```python
from noesis_forecasting_models import ProphetModel

model = ProphetModel(config)
model.fit(data)
predictions = model.predict(steps=12)
```

#### XGBoost
```python
from noesis_forecasting_models import XGBoostModel

model = XGBoostModel(config)
model.fit(data)
predictions = model.predict(steps=12)
```

### Preprocesamiento

```python
preprocessor = DataPreprocessor(config)

# Manejo de valores faltantes
data_clean = preprocessor.handle_missing_values(data, method='interpolate')

# Detección de outliers
outliers = preprocessor.detect_outliers(data, method='iqr')

# Manejo de outliers
data_clean = preprocessor.handle_outliers(data, method='winsorize')

# Detección de estacionalidad
seasonality = preprocessor.detect_seasonality(data)

# Creación de features
features = preprocessor.create_features(data)
```

## Configuración Avanzada

### ForecastingConfig

```python
config = ForecastingConfig(
    # División de datos
    test_size=0.2,
    validation_size=0.1,
    random_state=42,
    
    # Parámetros ARIMA/SARIMA
    max_p=5, max_d=2, max_q=5,
    max_P=2, max_D=1, max_Q=2,
    seasonal_period=12,
    
    # Parámetros ML
    xgb_params={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    
    # Ensemble
    ensemble_method='weighted',  # 'weighted', 'stacking', 'voting'
    weights={
        'arima': 0.25,
        'sarima': 0.25,
        'prophet': 0.2,
        'xgboost': 0.15,
        'lightgbm': 0.1,
        'random_forest': 0.05
    },
    
    # Validación
    n_splits=5,
    walk_forward=True,
    horizon=12
)
```

## Casos de Uso

### 1. Forecasting de Demanda
```python
# Datos de ventas históricas
ventas = pd.read_csv('ventas.csv', parse_dates=['fecha'], index_col='fecha')

api = NoesisForecastingAPI()
api.train_all_models(ventas['ventas'])

# Predicción de próximos 30 días
prediccion = api.predict_ensemble(steps=30, method='weighted')
print(f"Demanda预测ada: {prediccion['predictions'].sum():.0f} unidades")
```

### 2. Análisis de Tendencias Financieras
```python
# Datos de precios
precios = pd.read_csv('precios.csv', parse_dates=['fecha'], index_col='fecha')['precio']

analysis = api.analyze_series(precios)
print(f"Fuerza de tendencia: {analysis['seasonality']['seasonal_strength']:.3f}")

# Predicción con confianza
pred_con_confianza = api.predict_ensemble(steps=7)
print(f"Confianza promedio: {pred_con_confianza['confidence']:.2f}")
```

### 3. Monitoreo en Tiempo Real
```python
# Simulación de sistema de monitoreo
def monitor_tiempo_real(nuevos_datos, api):
    # Actualizar modelo
    api.train_all_models(nuevos_datos)
    
    # Predicción inmediata
    pred = api.predict_ensemble(steps=1)
    
    # Alertas por confianza baja
    if pred['confidence'] < 0.6:
        print("⚠️  Alerta: Confianza baja en predicción")
    
    return pred

# Uso en producción
prediccion = monitor_tiempo_real(datos_nuevos, api)
```

## Métricas de Evaluación

### Métricas Implementadas

- **MAE (Mean Absolute Error)**: Error absoluto medio
- **RMSE (Root Mean Square Error)**: Raíz del error cuadrático medio
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual absoluto medio
- **R² (R-squared)**: Coeficiente de determinación

### Interpretación

```python
results = api.train_all_models(data)

for modelo, metricas in results.items():
    if 'error' not in metricas:
        print(f"{modelo}:")
        print(f"  MAE: {metricas['mae']:.2f}")
        print(f"  RMSE: {metricas['rmse']:.2f}")
        print(f"  R²: {metricas['r2']:.3f}")
        
        # Interpretación
        if metricas['r2'] > 0.8:
            print("  ✅ Excelente capacidad predictiva")
        elif metricas['r2'] > 0.6:
            print("  ✅ Buena capacidad predictiva")
        elif metricas['r2'] > 0.4:
            print("  ⚠️  Capacidad predictiva moderada")
        else:
            print("  ❌ Capacidad predictiva baja")
```

## Mejores Prácticas

### 1. Preparación de Datos
```python
# ✅ Buena práctica
analysis = api.analyze_series(data)
if analysis['missing_values'] > 0:
    data = preprocessor.handle_missing_values(data, 'interpolate')
if analysis['outliers_count'] > 0:
    data = preprocessor.handle_outliers(data, 'winsorize')

# ❌ Mala práctica
# data = pd.read_csv('data.csv')  # Sin análisis ni limpieza
```

### 2. Validación
```python
# ✅ Siempre validar
validator = Validator(config)
validation_results = validator.walk_forward_validation(model, data)

# ❌ No validar
# model.fit(data); model.predict(future)  # Sin validación
```

### 3. Ensemble vs Individual
```python
# ✅ Usar ensemble para mejor robustez
ensemble_pred = api.predict_ensemble(steps=12, method='weighted')

# ✅ Pero mantener modelo individual para comparación
individual_pred = api.predict('arima', steps=12)
```

### 4. Monitoreo Continuo
```python
# ✅ Monitorear performance
def evaluar_modelo_continuo(api, nuevos_datos):
    pred = api.predict_ensemble(steps=1)
    # Guardar métricas para análisis
    return pred

# ❌ No monitorear
# pred = api.predict(); # sin seguimiento
```

## Troubleshooting

### Error: "Modelo no entrenado"
```python
# ❌ Error
pred = api.predict('ensemble')  # Antes de entrenar

# ✅ Solución
api.train_all_models(data)  # Entrenar primero
pred = api.predict('ensemble')  # Luego predecir
```

### Error: "Prophet no disponible"
```python
# Instalar Prophet
pip install prophet

# O usar solo otros modelos
models = ['arima', 'sarima', 'xgboost', 'lightgbm']
results = api.train_all_models(data)  # Sin Prophet
```

### Error: "Parámetros no convergen"
```python
# Reducir complejidad de búsqueda
config = ForecastingConfig(
    max_p=2, max_q=2,  # Menos parámetros
    seasonal_period=12  # Fijar período conocido
)
```

## Estructura de Archivos

```
workspace/
├── noesis_forecasting_models.py    # Sistema principal
├── ejemplos_noesis_forecasting.py  # Ejemplos de uso
├── README.md                       # Esta documentación
├── requirements.txt                # Dependencias
├── modelos/                        # Modelos guardados
│   ├── arima_model.pkl
│   ├── sarima_model.pkl
│   └── ensemble_config.json
└── datos/                          # Datos de ejemplo
    ├── datos_entrenamiento.csv
    └── datos_prueba.csv
```

## Soporte y Contacto

- **Autor**: Sistema NOESIS
- **Versión**: 1.0
- **Documentación**: Este README
- **Ejemplos**: `ejemplos_noesis_forecasting.py`

## Licencia

Sistema desarrollado para NOESIS. Todos los derechos reservados.

---

**¡Sistema de Forecasting NOESIS listo para producción!** 🚀
