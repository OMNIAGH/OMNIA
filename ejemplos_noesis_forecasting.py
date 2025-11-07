"""
NOESIS Forecasting Models - Ejemplos de Uso
==========================================

Este archivo contiene ejemplos prácticos de uso del sistema de forecasting.

Autor: NOESIS
Versión: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from noesis_forecasting_models import (
    NoesisForecastingAPI, 
    ForecastingConfig, 
    create_sample_data,
    DataPreprocessor
)

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def ejemplo_basico():
    """Ejemplo básico de uso del sistema"""
    print("=== EJEMPLO BÁSICO ===")
    
    # 1. Configuración
    config = ForecastingConfig(
        test_size=0.2,
        validation_size=0.1,
        ensemble_method='weighted',
        walk_forward=True
    )
    
    # 2. Crear API
    api = NoesisForecastingAPI(config)
    
    # 3. Generar datos de muestra
    data = create_sample_data(
        start_date='2020-01-01',
        periods=365,
        frequency='D',
        trend=0.1,
        seasonality_amplitude=10
    )
    
    print(f"Datos generados: {len(data)} observaciones")
    print(f"Rango: {data.index[0]} a {data.index[-1]}")
    print(f"Estadísticas: Media={data.mean():.2f}, Std={data.std():.2f}")
    
    # 4. Análisis de la serie
    analysis = api.analyze_series(data)
    print(f"\nAnálisis de estacionalidad:")
    print(f"- Es estacionaria: {analysis['seasonality']['is_stationary']}")
    print(f"- Tiene estacionalidad: {analysis['seasonality']['has_seasonality']}")
    print(f"- Período detectado: {analysis['seasonality']['best_period']}")
    
    # 5. Entrenar modelos
    print("\nEntrenando modelos...")
    results = api.train_all_models(data)
    
    # Mostrar resultados
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{model_name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        else:
            print(f"{model_name}: Error - {metrics['error']}")
    
    # 6. Hacer predicciones
    print("\nGenerando predicciones para los próximos 12 días...")
    pred = api.predict('ensemble', steps=12)
    print(f"Predicciones: {pred.values}")
    
    # 7. Guardar modelos
    api.save_models("./ejemplo_modelos")
    print("Modelos guardados en ./ejemplo_modelos")
    
    return api, data, results

def ejemplo_avanzado():
    """Ejemplo avanzado con configuración personalizada"""
    print("\n=== EJEMPLO AVANZADO ===")
    
    # Configuración avanzada
    config = ForecastingConfig(
        test_size=0.3,
        validation_size=0.15,
        seasonal_period=7,  # Semanal
        ensemble_method='stacking',
        n_splits=10,
        walk_forward=True,
        xgb_params={
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.9
        },
        lgb_params={
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    )
    
    # Crear API
    api = NoesisForecastingAPI(config)
    
    # Crear datos más complejos con múltiples estacionalidades
    dates = pd.date_range('2020-01-01', periods=730, freq='D')
    
    # Componentes más complejos
    trend = np.linspace(0, 20, len(dates))
    daily_season = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Semanal
    monthly_season = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # Mensual
    noise = np.random.normal(0, 2, len(dates))
    
    data = pd.Series(trend + daily_season + monthly_season + noise, index=dates)
    
    print(f"Datos complejos generados: {len(data)} observaciones")
    print(f"Componentes: Tendencia + Estacionalidad diaria + Estacionalidad mensual + Ruido")
    
    # Preprocesamiento avanzado
    preprocessor = DataPreprocessor(config)
    
    # Detectar y manejar outliers
    outliers = preprocessor.detect_outliers(data, method='iqr')
    print(f"Outliers detectados: {outliers.sum()}")
    
    # Manejo de valores faltantes (simular algunos)
    data_with_missing = data.copy()
    data_with_missing.iloc[50:55] = np.nan
    data_with_missing.iloc[200:203] = np.nan
    
    data_processed = preprocessor.handle_missing_values(data_with_missing, 'interpolate')
    print(f"Valores faltantes procesados: {data_with_missing.isnull().sum()}")
    
    # Análisis completo
    analysis = api.analyze_series(data_processed)
    
    # Entrenar con validación avanzada
    print("\nEntrenando con validación walk-forward...")
    results = api.train_all_models(data_processed)
    
    return api, data_processed, analysis

def ejemplo_evaluacion_modelos():
    """Ejemplo de evaluación comparativa de modelos"""
    print("\n=== EVALUACIÓN COMPARATIVA ===")
    
    # Configuración
    config = ForecastingConfig(
        test_size=0.25,
        walk_forward=True,
        n_splits=7,
        horizon=14
    )
    
    api = NoesisForecastingAPI(config)
    
    # Datos de prueba
    data = create_sample_data(periods=400, trend=0.05, seasonality_amplitude=8)
    
    # Entrenar todos los modelos
    results = api.train_all_models(data)
    
    # Comparar modelos individuales
    print("Comparación de modelos individuales:")
    model_performance = []
    
    for model_name in ['arima', 'sarima', 'xgboost', 'lightgbm', 'random_forest']:
        if model_name in results and 'error' not in results[model_name]:
            model_performance.append({
                'model': model_name,
                'mae': results[model_name]['mae'],
                'rmse': results[model_name]['rmse'],
                'r2': results[model_name]['r2']
            })
    
    # Ordenar por R²
    model_performance.sort(key=lambda x: x['r2'], reverse=True)
    
    print("Ranking por R²:")
    for i, perf in enumerate(model_performance, 1):
        print(f"{i}. {perf['model']}: R²={perf['r2']:.3f}, MAE={perf['mae']:.2f}, RMSE={perf['rmse']:.2f}")
    
    # Comparar ensemble vs individual
    if 'ensemble' in results and 'error' not in results['ensemble']:
        best_individual = model_performance[0]
        ensemble = results['ensemble']
        
        print(f"\nEnsemble vs Mejor Individual ({best_individual['model']}):")
        print(f"Ensemble R²: {ensemble['r2']:.3f}")
        print(f"Individual R²: {best_individual['r2']:.3f}")
        print(f"Mejora: {((ensemble['r2'] - best_individual['r2']) / best_individual['r2'] * 100):.1f}%")
    
    return api, data, model_performance

def ejemplo_predicciones_tiempo_real():
    """Ejemplo de predicciones en tiempo real"""
    print("\n=== PREDICCIONES EN TIEMPO REAL ===")
    
    # Configuración optimizada para tiempo real
    config = ForecastingConfig(
        test_size=0.1,
        validation_size=0.05,
        ensemble_method='weighted',
        walk_forward=True,
        horizon=7
    )
    
    api = NoesisForecastingAPI(config)
    
    # Datos históricos
    data = create_sample_data(periods=200, trend=0.08, seasonality_amplitude=12)
    
    # Entrenar modelos
    api.train_all_models(data)
    
    # Simular predicciones en tiempo real
    print("Simulación de predicciones en tiempo real:")
    
    for day in range(5):
        # Predicción para el día siguiente
        pred = api.predict_ensemble(steps=1, method='weighted')
        
        print(f"Día {day+1}:")
        print(f"  Predicción: {pred['predictions'].iloc[0]:.2f}")
        print(f"  Confianza: {pred['confidence']:.2f}")
        print(f"  Modelos utilizados: {list(pred['individual_predictions'].keys())}")
        
        # Simular observación real (añadir ruido)
        real_value = pred['predictions'].iloc[0] + np.random.normal(0, 1)
        print(f"  Valor real: {real_value:.2f}")
        print(f"  Error: {abs(pred['predictions'].iloc[0] - real_value):.2f}")
        print()
    
    return api

def ejemplo_analisis_diagnostico():
    """Ejemplo de análisis de diagnóstico y monitoreo"""
    print("\n=== ANÁLISIS DE DIAGNÓSTICO ===")
    
    config = ForecastingConfig()
    api = NoesisForecastingAPI(config)
    
    # Crear datos con patrones conocidos
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # Datos con cambio de régimen
    regime1 = 50 + 0.1 * np.arange(150) + 5 * np.sin(2 * np.pi * np.arange(150) / 12)
    regime2 = 80 + 0.05 * np.arange(150) + 8 * np.sin(2 * np.pi * np.arange(150) / 12)
    combined_data = np.concatenate([regime1, regime2])
    noise = np.random.normal(0, 2, len(combined_data))
    data = pd.Series(combined_data + noise, index=dates)
    
    # Análisis
    analysis = api.analyze_series(data)
    
    print("Diagnóstico de la serie:")
    print(f"- Estacionariedad ADF p-value: {analysis['stationarity']['p_value']:.4f}")
    print(f"- Fuerza estacional: {analysis['seasonality']['seasonal_strength']:.3f}")
    print(f"- Detección de outliers: {analysis['outliers_count']}")
    print(f"- Valores faltantes: {analysis['missing_values']}")
    
    # Detectar cambio de régimen (simplificado)
    mid_point = len(data) // 2
    mean1 = data[:mid_point].mean()
    mean2 = data[mid_point:].mean()
    
    print(f"- Media primera mitad: {mean1:.2f}")
    print(f"- Media segunda mitad: {mean2:.2f}")
    print(f"- Cambio de nivel detectado: {abs(mean2 - mean1) > 2 * data.std()}")
    
    return api, data, analysis

def crear_dashboard_resultados(api, data, results):
    """Crear dashboard con resultados visuales"""
    print("\n=== CREANDO DASHBOARD ===")
    
    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dashboard de Resultados NOESIS Forecasting', fontsize=16)
    
    # 1. Serie temporal original
    axes[0, 0].plot(data.index, data.values, 'b-', alpha=0.7)
    axes[0, 0].set_title('Serie Temporal Original')
    axes[0, 0].set_xlabel('Fecha')
    axes[0, 0].set_ylabel('Valor')
    axes[0, 0].grid(True)
    
    # 2. Comparación de errores
    models = []
    errors = []
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            models.append(model_name)
            errors.append(metrics['rmse'])
    
    if models:
        axes[0, 1].bar(models, errors, color='skyblue')
        axes[0, 1].set_title('Comparación de RMSE por Modelo')
        axes[0, 1].set_xlabel('Modelo')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Predicciones
    if 'ensemble' in api.models:
        pred = api.predict('ensemble', steps=30)
        pred_index = pd.date_range(data.index[-1], periods=len(pred)+1, freq='D')[1:]
        
        # Mostrar últimos datos + predicciones
        recent_data = data.tail(50)
        axes[1, 0].plot(recent_data.index, recent_data.values, 'b-', label='Datos Históricos')
        axes[1, 0].plot(pred_index, pred.values, 'r--', label='Predicciones')
        axes[1, 0].set_title('Predicciones del Ensemble')
        axes[1, 0].set_xlabel('Fecha')
        axes[1, 0].set_ylabel('Valor')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. Distribución de errores (si hay validación)
    validation_errors = []
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            validation_errors.append(metrics['mae'])
    
    if validation_errors:
        axes[1, 1].hist(validation_errors, bins=10, alpha=0.7, color='lightgreen')
        axes[1, 1].set_title('Distribución de Errores MAE')
        axes[1, 1].set_xlabel('Error MAE')
        axes[1, 1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig('./dashboard_noesis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Dashboard guardado en ./dashboard_noesis.png")

def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("SISTEMA DE FORECASTING NOESIS - EJEMPLOS DE USO")
    print("=" * 50)
    
    try:
        # Ejemplo básico
        api1, data1, results1 = ejemplo_basico()
        
        # Ejemplo avanzado
        api2, data2, analysis2 = ejemplo_avanzado()
        
        # Evaluación comparativa
        api3, data3, performance = ejemplo_evaluacion_modelos()
        
        # Predicciones tiempo real
        api4 = ejemplo_predicciones_tiempo_real()
        
        # Análisis de diagnóstico
        api5, data5, analysis5 = ejemplo_analisis_diagnostico()
        
        # Crear dashboard (usando el primer ejemplo)
        crear_dashboard_resultados(api1, data1, results1)
        
        print("\n" + "=" * 50)
        print("¡TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE!")
        print("=" * 50)
        
        # Resumen de capacidades
        print("\nCapacidades del Sistema NOESIS:")
        print("✓ Modelos ARIMA y SARIMA para series temporales")
        print("✓ Modelos ML (XGBoost, LightGBM, Random Forest)")
        print("✓ Ensemble methods con pesos optimizados")
        print("✓ Validación walk-forward y cross-validation")
        print("✓ API para predicciones en tiempo real")
        print("✓ Manejo automático de outliers y valores faltantes")
        print("✓ Detección automática de estacionalidad")
        print("✓ Guardado/carga de modelos entrenados")
        print("✓ Análisis de diagnóstico completo")
        print("✓ Visualizaciones y dashboards")
        
    except Exception as e:
        print(f"Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
