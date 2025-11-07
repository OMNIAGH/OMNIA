# NOESIS - Sistema de Análisis de Tendencias

## Descripción

Sistema completo para el análisis de tendencias en datos financieros y económicos. Incluye detección automática de tendencias, análisis de estacionalidad, identificación de puntos de inflexión, correlación entre variables, análisis de volatilidad y sistema de alertas.

## Instalación

```bash
# Instalar dependencias básicas
pip install -r requirements.txt

# O instalar manualmente
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

## Uso Rápido

```python
from noesis_trend_analysis import *

# Crear analizadores
trend_detector = TrendDetector()
volatility_analyzer = VolatilityAnalyzer()
alert_system = AlertSystem()

# Análisis básico
trend_analysis = trend_detector.detect_multiple_timeframes(prices)
volatility_results = volatility_analyzer.calculate_risk_metrics(returns)

# Generar alertas
alerts = alert_system.run_comprehensive_alert_check(data)
```

## Componentes Principales

### 1. TrendDetector
Detección automática de tendencias alcistas, bajistas y laterales.

```python
detector = TrendDetector()

# Análisis completo
results = detector.detect_multiple_timeframes(prices)
print(f"Tendencia: {results['consensus'].value}")
print(f"Fuerza: {results['consensus_strength']:.2%}")

# Visualización
detector.plot_trend_analysis(prices, save_path='trend_analysis.png')
```

### 2. SeasonalityAnalyzer
Análisis de estacionalidad y patrones cíclicos.

```python
analyzer = SeasonalityAnalyzer()

# Test de estacionariedad
is_stationary = analyzer.test_stationarity(series)
print(f"Estacionaria: {is_stationary['is_stationary']}")

# Detección de ciclos
cycles = analyzer.detect_cycles(series)
print(f"Ciclos: {cycles['cycle_periods']}")

# Descomposición estacional
seasonality = analyzer.detect_seasonality(series, period=365)
```

### 3. VolatilityAnalyzer
Análisis de volatilidad y métricas de riesgo.

```python
analyzer = VolatilityAnalyzer()

# Métricas de volatilidad
vol_metrics = analyzer.calculate_volatility_metrics(returns)
print(f"Volatilidad: {vol_metrics['volatility_20d'].iloc[-1]:.2%}")

# VaR y CVaR
risk_metrics = analyzer.calculate_var_cvar(returns)
print(f"VaR 95%: {risk_metrics['var_95']:.2%}")

# Ratios de riesgo
risk = analyzer.calculate_risk_metrics(returns)
print(f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
```

### 4. CorrelationAnalyzer
Análisis de correlaciones entre múltiples variables.

```python
analyzer = CorrelationAnalyzer()

# Matriz de correlación
corr_matrix = analyzer.calculate_correlation_matrix(data)
print(f"Correlación: {corr_matrix.loc['var1', 'var2']:.3f}")

# Análisis completo
results = analyzer.plot_correlation_analysis(data, save_path='correlations.png')
```

### 5. InflectionPointDetector
Identificación de puntos de inflexión y cambios de tendencia.

```python
detector = InflectionPointDetector()

# Detectar picos y valles
inflection_points = detector.detect_peaks_and_troughs(series)
print(f"Puntos detectados: {len(inflection_points)}")

# Cambios de tendencia
trend_changes = detector.detect_trend_changes(series)

# Visualización
detector.plot_inflection_points(series, save_path='inflection_points.png')
```

### 6. AlertSystem
Sistema de alertas para cambios significativos.

```python
alert_system = AlertSystem()

# Verificación completa
alerts = alert_system.run_comprehensive_alert_check(alert_data)

# Generar reporte
report = alert_system.generate_alert_report(alerts)
print(report)

# Guardar log
alert_system.save_alert_log('alerts.json')
```

## Ejemplo Completo

Ver `example_noesis.py` para un ejemplo completo de uso del sistema.

```bash
python example_noesis.py
```

## Estructura de Datos

### Enums
- `TrendDirection`: ALCISTA, BAJISTA, LATERAL, INDETERMINADO
- `AlertLevel`: BAJO, MEDIO, ALTO, CRITICO

### Dataclasses
- `TrendMetrics`: Métricas completas de tendencia
- `InflectionPoint`: Puntos de inflexión
- `Alert`: Sistema de alertas

## Visualizaciones

El sistema incluye visualizaciones profesionales para:

- Análisis de tendencias con indicadores técnicos
- Descomposición estacional y ciclos
- Volatilidad y distribuciones de riesgo
- Matrices de correlación y redes
- Puntos de inflexión y cambios de tendencia

Todas las visualizaciones se pueden guardar usando el parámetro `save_path`.

## Métricas Disponibles

### Tendencias
- Dirección de tendencia (alcista/bajista/lateral)
- Fuerza y confianza de tendencia
- R² de regresión lineal
- Señales por marcos temporales

### Estacionalidad
- Test de estacionariedad (ADF)
- Componentes estacionales
- Períodos de ciclos
- Fuerza de estacionalidad

### Volatilidad
- Volatilidad histórica y móvil
- VaR y CVaR (múltiples niveles)
- Ratios: Sharpe, Sortino, Calmar
- Máximo drawdown

### Correlaciones
- Matrices multidimensionales
- Rupturas estructurales
- Correlaciones por régimen

### Alertas
- Volatilidad extrema
- Cambios de tendencia
- Movimientos extremos
- Rupturas de soporte/resistencia

## Configuración Avanzada

```python
# Personalizar analizador de tendencias
detector = TrendDetector(window_size=30, min_trend_length=15)

# Personalizar períodos de estacionalidad
analyzer = SeasonalityAnalyzer(periods=[7, 30, 90, 252])

# Personalizar ventanas de volatilidad
vol_analyzer = VolatilityAnalyzer(window_sizes=[10, 20, 60, 120])

# Personalizar reglas de alerta
alert_rules = {
    'volatility_spike': {
        'threshold': 0.90,
        'level': AlertLevel.ALTO,
        'message': 'Spike de volatilidad detectado'
    }
}
alert_system = AlertSystem(alert_rules=alert_rules)
```

## Troubleshooting

### Errores Comunes

1. **ImportError**: Verificar que todas las dependencias estén instaladas
2. **MemoryError**: Reducir el tamaño de los datos o usar ventanas más pequeñas
3. **ConvergenceWarning**: Normalizar los datos antes del análisis

### Optimización

- Usar datos diarios para mejor rendimiento
- Reducir número de visualizaciones si hay problemas de memoria
- Ajustar parámetros de ventana según el tamaño de datos

## Contribución

Para contribuir al sistema:

1. Fork del repositorio
2. Crear rama de feature
3. Implementar cambios con tests
4. Submit pull request

## Licencia

Sistema desarrollado para NOESIS - Todos los derechos reservados.

## Soporte

Para soporte técnico o preguntas, contactar al equipo de desarrollo de NOESIS.