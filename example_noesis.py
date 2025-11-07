"""
Ejemplo de Uso de NOESIS - Sistema de Análisis de Tendencias
Demostración de funcionalidades sin visualizaciones
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importar el sistema NOESIS
try:
    from noesis_trend_analysis import (
        TrendDetector, SeasonalityAnalyzer, VolatilityAnalyzer,
        CorrelationAnalyzer, InflectionPointDetector, AlertSystem,
        TrendDirection, AlertLevel
    )
    print("✅ Sistema NOESIS importado correctamente")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    exit(1)

def create_sample_data():
    """Crear datos de ejemplo realistas"""
    print("\n📊 Creando datos de ejemplo...")
    
    # Fechas
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=500, freq='D')
    
    # Simular precio con tendencia, estacionalidad y ruido
    trend = np.linspace(100, 150, 500)  # Tendencia alcista
    seasonal = 10 * np.sin(2 * np.pi * np.arange(500) / 365.25)  # Estacionalidad
    noise = np.random.normal(0, 3, 500)  # Ruido
    price = trend + seasonal + noise
    
    # Rendimientos
    returns = pd.Series(np.diff(np.log(price)), index=dates[1:])
    
    # Variables adicionales
    volume = np.random.lognormal(10, 1, 500)
    vix = np.maximum(10, 20 + 5 * np.sin(2 * np.pi * np.arange(500) / 252) + 
                    np.random.normal(0, 3, 500))
    
    # Crear DataFrame
    data = pd.DataFrame({
        'precio': pd.Series(price, index=dates),
        'rendimiento': returns,
        'volumen': volume,
        'vix': vix
    })
    
    print(f"✅ Datos creados: {data.shape[0]} observaciones, {data.shape[1]} variables")
    return data

def demonstrate_trend_detection(data):
    """Demostrar detección de tendencias"""
    print("\n🔍 DEMOSTRACIÓN: Detección de Tendencias")
    print("-" * 50)
    
    detector = TrendDetector()
    
    # Análisis de tendencia lineal
    linear_result = detector.detect_trend_linear(data['precio'])
    print(f"Tendencia Lineal:")
    print(f"  • Dirección: {linear_result.direction.value}")
    print(f"  • Pendiente: {linear_result.slope:.4f}")
    print(f"  • R²: {linear_result.r_squared:.3f}")
    print(f"  • Confianza: {linear_result.confidence:.3f}")
    
    # Análisis por medias móviles
    ma_signals = detector.detect_trend_moving_average(data['precio'])
    print(f"\nSeñales por Medias Móviles:")
    for timeframe, direction in ma_signals.items():
        print(f"  • {timeframe}: {direction.value}")
    
    # Análisis de momentum
    momentum = detector.detect_trend_momentum(data['precio'])
    print(f"\nIndicadores de Momentum:")
    print(f"  • RSI: {momentum['rsi']:.1f}")
    print(f"  • MACD: {momentum['macd']:.4f}")
    
    return {
        'linear': linear_result,
        'moving_averages': ma_signals,
        'momentum': momentum
    }

def demonstrate_seasonality_analysis(data):
    """Demostrar análisis de estacionalidad"""
    print("\n📅 DEMOSTRACIÓN: Análisis de Estacionalidad")
    print("-" * 50)
    
    analyzer = SeasonalityAnalyzer()
    
    # Test de estacionariedad
    stationarity = analyzer.test_stationarity(data['precio'])
    print(f"Test de Estacionariedad:")
    print(f"  • Es estacionaria: {'Sí' if stationarity['is_stationary'] else 'No'}")
    print(f"  • P-valor: {stationarity['p_value']:.4f}")
    
    # Detección de ciclos
    cycles = analyzer.detect_cycles(data['precio'])
    print(f"\nDetección de Ciclos:")
    print(f"  • Ciclos detectados: {len(cycles['cycle_periods'])}")
    if cycles['cycle_periods']:
        print(f"  • Períodos: {cycles['cycle_periods'][:3]}")
    
    # Análisis de estacionalidad
    seasonality = analyzer.detect_seasonality(data['precio'], period=365)
    if seasonality:
        yearly_decomp = seasonality[365]
        print(f"\nDescomposición Anual:")
        print(f"  • Fuerza estacional: {yearly_decomp['seasonal_strength']:.3f}")
        print(f"  • Fuerza tendencia: {yearly_decomp['trend_strength']:.3f}")
    
    return {
        'stationarity': stationarity,
        'cycles': cycles,
        'seasonality': seasonality
    }

def demonstrate_volatility_analysis(data):
    """Demostrar análisis de volatilidad"""
    print("\n📈 DEMOSTRACIÓN: Análisis de Volatilidad")
    print("-" * 50)
    
    analyzer = VolatilityAnalyzer()
    
    # Métricas de volatilidad
    vol_metrics = analyzer.calculate_volatility_metrics(data['rendimiento'])
    current_vol = vol_metrics['volatility_20d'].iloc[-1]
    avg_vol = vol_metrics['volatility_mean']
    
    print(f"Métricas de Volatilidad:")
    print(f"  • Volatilidad actual (20d): {current_vol:.2%}")
    print(f"  • Volatilidad promedio: {avg_vol:.2%}")
    
    # VaR y CVaR
    var_cvar = analyzer.calculate_var_cvar(data['rendimiento'])
    print(f"\nValue at Risk:")
    print(f"  • VaR 95%: {var_cvar['var_95']:.2%}")
    print(f"  • CVaR 95%: {var_cvar['cvar_95']:.2%}")
    
    # Métricas de riesgo
    risk_metrics = analyzer.calculate_risk_metrics(data['rendimiento'])
    print(f"\nRatios de Riesgo:")
    print(f"  • Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"  • Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
    print(f"  • Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
    print(f"  • Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}")
    
    # Regímenes de volatilidad
    regimes = analyzer.detect_volatility_regimes(data['rendimiento'])
    print(f"\nRegímenes de Volatilidad:")
    print(f"  • Regimen de baja volatilidad: {regimes['low_volatility_regime']}")
    print(f"  • Regimen de alta volatilidad: {regimes['high_volatility_regime']}")
    
    return {
        'volatility_metrics': vol_metrics,
        'risk_metrics': risk_metrics,
        'var_cvar': var_cvar,
        'regimes': regimes
    }

def demonstrate_correlation_analysis(data):
    """Demostrar análisis de correlaciones"""
    print("\n🔗 DEMOSTRACIÓN: Análisis de Correlaciones")
    print("-" * 50)
    
    analyzer = CorrelationAnalyzer()
    
    # Matriz de correlación
    corr_matrix = analyzer.calculate_correlation_matrix(data[['precio', 'vix', 'volumen']])
    print(f"Matriz de Correlación:")
    print(f"  • Precio vs VIX: {corr_matrix.loc['precio', 'vix']:.3f}")
    print(f"  • Precio vs Volumen: {corr_matrix.loc['precio', 'volumen']:.3f}")
    print(f"  • VIX vs Volumen: {corr_matrix.loc['vix', 'volumen']:.3f}")
    
    # Top correlaciones
    correlation_data = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            correlation_data.append({
                'pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                'correlation': corr_matrix.iloc[i, j]
            })
    
    correlation_data.sort(key=lambda x: abs(x['correlation']), reverse=True)
    print(f"\nTop 3 Correlaciones:")
    for i, item in enumerate(correlation_data[:3]):
        print(f"  {i+1}. {item['pair']}: {item['correlation']:.3f}")
    
    return {
        'correlation_matrix': corr_matrix,
        'top_correlations': correlation_data
    }

def demonstrate_inflection_points(data):
    """Demostrar detección de puntos de inflexión"""
    print("\n🎯 DEMOSTRACIÓN: Puntos de Inflexión")
    print("-" * 50)
    
    detector = InflectionPointDetector()
    
    # Detectar picos y valles
    peaks_troughs = detector.detect_peaks_and_troughs(data['precio'])
    max_points = [p for p in peaks_troughs if p.type == 'max']
    min_points = [p for p in peaks_troughs if p.type == 'min']
    
    print(f"Puntos de Inflexión Detectados:")
    print(f"  • Picos (máximos): {len(max_points)}")
    print(f"  • Valles (mínimos): {len(min_points)}")
    
    # Mostrar algunos puntos recientes
    if max_points:
        latest_peak = max(max_points, key=lambda x: x.date)
        print(f"\nÚltimo Pico:")
        print(f"  • Fecha: {latest_peak.date}")
        print(f"  • Precio: {latest_peak.price:.2f}")
        print(f"  • Significancia: {latest_peak.significance:.3f}")
    
    if min_points:
        latest_trough = min(min_points, key=lambda x: x.date)
        print(f"\nÚltimo Valle:")
        print(f"  • Fecha: {latest_trough.date}")
        print(f"  • Precio: {latest_trough.price:.2f}")
        print(f"  • Significancia: {latest_trough.significance:.3f}")
    
    # Detectar cambios de tendencia
    trend_changes = detector.detect_trend_changes(data['precio'])
    print(f"\nCambios de Tendencia:")
    print(f"  • Cambios detectados: {len(trend_changes)}")
    
    if trend_changes:
        latest_change = trend_changes[-1]
        print(f"  • Último cambio: {latest_change.type}")
        print(f"  • Fecha: {latest_change.date}")
        print(f"  • Precio: {latest_change.price:.2f}")
    
    return {
        'peaks_troughs': peaks_troughs,
        'trend_changes': trend_changes
    }

def demonstrate_alert_system(data, results):
    """Demostrar sistema de alertas"""
    print("\n🚨 DEMOSTRACIÓN: Sistema de Alertas")
    print("-" * 50)
    
    alert_system = AlertSystem()
    
    # Preparar datos para alertas
    vol_20d = results['volatility']['volatility_metrics']['volatility_20d']
    vol_percentiles = results['volatility']['volatility_metrics']['volatility_percentiles']
    
    alert_data = {
        'volatility_metrics': {
            'current_vol': vol_20d.iloc[-1],
            'percentiles': vol_percentiles
        },
        'trend_analysis': results['trend'],
        'returns': data['rendimiento'],
        'current_price': data['precio'].iloc[-1]
    }
    
    # Generar alertas
    alerts = alert_system.run_comprehensive_alert_check(alert_data)
    
    print(f"Alertas Generadas: {len(alerts)}")
    
    if alerts:
        print(f"\nDetalle de Alertas:")
        for i, alert in enumerate(alerts[:5]):  # Mostrar primeras 5
            print(f"  {i+1}. [{alert.level.value}] {alert.message}")
            print(f"     Métrica: {alert.metric}, Valor: {alert.value:.4f}")
    else:
        print("  • No se generaron alertas (situación normal)")
    
    # Generar reporte
    report = alert_system.generate_alert_report(alerts)
    print(f"\nReporte de Alertas Generado (primeros 200 caracteres):")
    print(f"  {report[:200]}...")
    
    return alerts

def main():
    """Función principal de demostración"""
    print("🏗️  NOESIS - SISTEMA DE ANÁLISIS DE TENDENCIAS")
    print("=" * 60)
    print("Demostración completa de funcionalidades")
    print("Sin visualizaciones para optimizar el rendimiento")
    
    # Crear datos
    data = create_sample_data()
    
    # Ejecutar análisis
    print("\n🚀 Iniciando análisis completo...")
    
    results = {}
    
    # 1. Detección de tendencias
    results['trend'] = demonstrate_trend_detection(data)
    
    # 2. Análisis de estacionalidad
    results['seasonality'] = demonstrate_seasonality_analysis(data)
    
    # 3. Análisis de volatilidad
    results['volatility'] = demonstrate_volatility_analysis(data)
    
    # 4. Análisis de correlaciones
    results['correlation'] = demonstrate_correlation_analysis(data)
    
    # 5. Puntos de inflexión
    results['inflection'] = demonstrate_inflection_points(data)
    
    # 6. Sistema de alertas
    results['alerts'] = demonstrate_alert_system(data, results)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE ANÁLISIS COMPLETADO")
    print("=" * 60)
    
    print(f"✅ Detección de Tendencias: {results['trend']['linear'].direction.value}")
    print(f"✅ Estacionariedad: {'Sí' if results['seasonality']['stationarity']['is_stationary'] else 'No'}")
    print(f"✅ Volatilidad Actual: {results['volatility']['volatility_metrics']['volatility_20d'].iloc[-1]:.2%}")
    print(f"✅ Sharpe Ratio: {results['volatility']['risk_metrics']['sharpe_ratio']:.2f}")
    print(f"✅ Correlación Principal: {abs(results['correlation']['correlation_matrix'].loc['precio', 'vix']):.3f}")
    print(f"✅ Puntos de Inflexión: {len(results['inflection']['peaks_troughs'])}")
    print(f"✅ Alertas Generadas: {len(results['alerts'])}")
    
    print("\n🎉 ¡Demostración completada exitosamente!")
    print("   El sistema NOESIS está completamente funcional.")
    
    return results

if __name__ == "__main__":
    # Ejecutar demostración
    try:
        results = main()
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()