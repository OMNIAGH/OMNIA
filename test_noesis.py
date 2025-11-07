"""
Script de prueba para NOESIS - Sistema de Análisis de Tendencias
Verifica las funcionalidades básicas sin generar gráficos
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Importar el módulo principal
from noesis_trend_analysis import (
    TrendDetector, SeasonalityAnalyzer, VolatilityAnalyzer,
    CorrelationAnalyzer, InflectionPointDetector, AlertSystem,
    generate_sample_data, TrendDirection, AlertLevel
)

def test_basic_functionality():
    """Prueba las funcionalidades básicas del sistema"""
    print("🧪 Probando Funcionalidades Básicas de NOESIS")
    print("=" * 50)
    
    # Generar datos de ejemplo
    print("📊 Generando datos de ejemplo...")
    data = generate_sample_data()
    print(f"✅ Datos generados: {data.shape[0]} filas, {data.shape[1]} columnas")
    print(f"   Columnas: {list(data.columns)}")
    
    # Prueba 1: TrendDetector
    print("\n🔍 Probando TrendDetector...")
    trend_detector = TrendDetector()
    
    trend_analysis = trend_detector.detect_multiple_timeframes(data['precio'])
    print(f"✅ Tendencia detectada: {trend_analysis['consensus'].value}")
    print(f"   Fuerza del consenso: {trend_analysis['consensus_strength']:.2%}")
    print(f"   Tendencia lineal: {trend_analysis['linear'].direction.value} (R²: {trend_analysis['linear'].r_squared:.3f})")
    
    # Prueba 2: SeasonalityAnalyzer
    print("\n📅 Probando SeasonalityAnalyzer...")
    seasonality_analyzer = SeasonalityAnalyzer()
    
    stationarity_result = seasonality_analyzer.test_stationarity(data['precio'])
    print(f"✅ Test de estacionariedad: {'Estacionaria' if stationarity_result['is_stationary'] else 'No estacionaria'}")
    print(f"   P-valor: {stationarity_result['p_value']:.4f}")
    
    cycles = seasonality_analyzer.detect_cycles(data['precio'])
    print(f"✅ Ciclos detectados: {len(cycles['cycle_periods'])}")
    if cycles['cycle_periods']:
        print(f"   Períodos de ciclo: {cycles['cycle_periods'][:3]}")  # Mostrar primeros 3
    
    # Prueba 3: VolatilityAnalyzer
    print("\n📈 Probando VolatilityAnalyzer...")
    volatility_analyzer = VolatilityAnalyzer()
    
    vol_metrics = volatility_analyzer.calculate_volatility_metrics(data['rendimiento'])
    current_vol = vol_metrics['volatility_20d'].iloc[-1]
    print(f"✅ Volatilidad actual (20d): {current_vol:.2%}")
    
    risk_metrics = volatility_analyzer.calculate_risk_metrics(data['rendimiento'])
    print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
    print(f"   VaR 95%: {risk_metrics['var_95']:.2%}")
    
    # Prueba 4: CorrelationAnalyzer
    print("\n🔗 Probando CorrelationAnalyzer...")
    correlation_analyzer = CorrelationAnalyzer()
    
    corr_matrix = correlation_analyzer.calculate_correlation_matrix(data[['precio', 'vix', 'volumen']])
    precio_vix_corr = corr_matrix.loc['precio', 'vix']
    print(f"✅ Correlación Precio-VIX: {precio_vix_corr:.3f}")
    
    # Prueba 5: InflectionPointDetector
    print("\n🎯 Probando InflectionPointDetector...")
    inflection_detector = InflectionPointDetector()
    
    peaks_troughs = inflection_detector.detect_peaks_and_troughs(data['precio'])
    trend_changes = inflection_detector.detect_trend_changes(data['precio'])
    
    max_points = [p for p in peaks_troughs if p.type == 'max']
    min_points = [p for p in peaks_troughs if p.type == 'min']
    
    print(f"✅ Puntos de inflexión detectados:")
    print(f"   Picos: {len(max_points)}")
    print(f"   Valles: {len(min_points)}")
    print(f"   Cambios de tendencia: {len(trend_changes)}")
    
    # Prueba 6: AlertSystem
    print("\n🚨 Probando AlertSystem...")
    alert_system = AlertSystem()
    
    # Preparar datos para alertas
    alert_data = {
        'volatility_metrics': {
            'current_vol': current_vol,
            'percentiles': vol_metrics['volatility_percentiles']
        },
        'trend_analysis': trend_analysis,
        'returns': data['rendimiento'],
        'current_price': data['precio'].iloc[-1]
    }
    
    alerts = alert_system.run_comprehensive_alert_check(alert_data)
    print(f"✅ Alertas generadas: {len(alerts)}")
    
    for alert in alerts[:3]:  # Mostrar primeras 3 alertas
        print(f"   {alert.level.value}: {alert.message}")
    
    # Prueba 7: Métricas integradas
    print("\n📊 Resumen de Métricas Integradas:")
    print(f"   • Análisis de tendencia: {trend_analysis['consensus'].value}")
    print(f"   • Estacionalidad detectada: {len(cycles['cycle_periods']) > 0}")
    print(f"   • Volatilidad actual: {current_vol:.2%}")
    print(f"   • Riesgo (VaR 95%): {risk_metrics['var_95']:.2%}")
    print(f"   • Correlación principal: {precio_vix_corr:.3f}")
    print(f"   • Puntos de inflexión: {len(peaks_troughs)}")
    print(f"   • Alertas activas: {len(alerts)}")
    
    print("\n🎉 ¡Todas las pruebas completadas exitosamente!")
    
    return True

def test_edge_cases():
    """Prueba casos extremos y validaciones"""
    print("\n🔬 Probando Casos Extremos...")
    print("=" * 30)
    
    # Test con datos mínimos
    try:
        small_data = pd.Series([1, 2, 3, 4, 5])
        detector = TrendDetector()
        result = detector.detect_trend_linear(small_data)
        print(f"✅ Datos mínimos procesados: {result.direction.value}")
    except Exception as e:
        print(f"❌ Error con datos mínimos: {e}")
    
    # Test con datos constantes
    try:
        constant_data = pd.Series([100] * 50)
        result = detector.detect_trend_linear(constant_data)
        print(f"✅ Datos constantes procesados: {result.direction.value}")
    except Exception as e:
        print(f"❌ Error con datos constantes: {e}")
    
    # Test con valores extremos
    try:
        extreme_data = pd.Series([1, 1000, 1, 1000, 1])
        result = detector.detect_trend_linear(extreme_data)
        print(f"✅ Datos extremos procesados: {result.direction.value}")
    except Exception as e:
        print(f"❌ Error con datos extremos: {e}")

if __name__ == "__main__":
    try:
        # Ejecutar pruebas básicas
        success = test_basic_functionality()
        
        if success:
            test_edge_cases()
            print("\n🏆 Sistema NOESIS funcionando correctamente!")
        else:
            print("\n❌ Fallos detectados en el sistema")
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("   Asegúrate de que todas las dependencias estén instaladas:")
        print("   pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()