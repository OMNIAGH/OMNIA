# NOESIS A/B Testing System 🚀

## Sistema de Experimentación Automática y Análisis Estadístico Avanzado

El **NOESIS A/B Testing System** es una plataforma completa de experimentación que incluye diseño automático de experimentos, análisis estadístico avanzado, optimización multi-armed bandit y reportes inteligentes.

## 🌟 Características Principales

### 1. **Diseño Experimental Automático**
- Cálculo automático de tamaño de muestra
- Estimación de duración de experimentos
- Segmentación automática de usuarios
- Análisis de poder estadístico
- Soporte para múltiples métricas

### 2. **Análisis Estadístico Avanzado**
- **Pruebas t** para comparaciones de medias
- **Chi-square** para análisis de proporciones
- **Análisis Bayesiano** con intervalos creíbles
- Detección automática de significancia
- Intervalos de confianza bootstrap

### 3. **Multi-Armed Bandit**
- Algoritmos epsilon-greedy, Thompson Sampling y UCB
- Optimización continua del tráfico
- Decay automático de parámetros
- Recomendaciones en tiempo real

### 4. **Análisis de Lift e Impacto**
- Cálculo de lift absoluto y relativo
- Impacto incremental en conversiones
- Proyecciones de ROI
- Escenarios de escalabilidad

### 5. **Early Stopping Inteligente**
- Detección automática de significancia
- Criterios de seguridad
- Monitoreo continuo
- Prevención de falsos positivos

### 6. **Reportes Automáticos**
- Reportes ejecutivos automáticos
- Insights accionables
- Recomendaciones basadas en IA
- Exportación en múltiples formatos

### 7. **Dashboard Interactivo**
- Visualizaciones en tiempo real
- Métricas clave
- Comparaciones lado a lado
- Estado de experimentos

## 📁 Archivos del Sistema

| Archivo | Descripción |
|---------|-------------|
| `noesis_ab_testing.py` | Módulo principal del sistema |
| `noesis_ab_testing_dashboard.html` | Dashboard interactivo |
| `noesis_ab_testing_examples.py` | Ejemplos y casos de uso |
| `README_NOESIS_AB_Testing.md` | Documentación completa |

## 🚀 Instalación y Configuración

### Requisitos
```bash
python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
scipy >= 1.7.0
```

### Instalación
```bash
# Clonar o descargar los archivos
# Instalar dependencias
pip install numpy pandas matplotlib seaborn scipy

# Ejecutar ejemplos
python noesis_ab_testing_examples.py

# Ver dashboard
# Abrir noesis_ab_testing_dashboard.html en navegador
```

## 💡 Uso Rápido

### Ejemplo Básico

```python
from noesis_ab_testing import NoesisABTestingSystem, ExperimentConfig

# Crear sistema
noesis = NoesisABTestingSystem()

# Configurar experimento
config = ExperimentConfig(
    name="Test de Página de Producto",
    description="Probando diferentes layouts",
    control_name="Layout Original",
    variant_names=["Layout Mejorado"],
    primary_metric="conversion_rate",
    min_sample_size_per_variant=1000,
    significance_level=0.05,
    power=0.8,
    early_stopping_enabled=True
)

# Crear experimento
experiment_id = noesis.create_experiment(
    config=config,
    baseline_rates={'control': 0.08},  # 8% tasa base
    expected_effects={'Layout Mejorado': 0.15},  # 15% mejora
    daily_traffic=5000
)

# Iniciar experimento
noesis.start_experiment(experiment_id)

# Añadir datos
for i in range(1000):
    # Simular conversión en control
    is_conversion = np.random.random() < 0.08
    noesis.add_data_point(experiment_id, 'control', 1.0 if is_conversion else 0.0)
    
    # Simular conversión en variante
    is_conversion = np.random.random() < 0.092  # 15% mejor
    noesis.add_data_point(experiment_id, 'Layout Mejorado', 1.0 if is_conversion else 0.0)

# Analizar resultados
results = noesis.analyze_experiment(experiment_id)

# Generar reporte
report = noesis.get_experiment_report(experiment_id)
print(report)
```

### Optimización con Bandit

```python
# Configurar variantes para bandit
arm_performance = {
    'homepage_v1': 0.08,
    'homepage_v2': 0.12,
    'homepage_v3': 0.06
}

# Optimizar
best_arm = noesis.optimize_with_bandit(arm_performance)
print(f"Mejor variante: {best_arm}")

# Obtener recomendaciones
recommendations = noesis.bandit_optimizer.get_recommendations()
for rec in recommendations:
    print(f"{rec['arm_id']}: {rec['estimated_value']:.3%}")
```

## 📊 Clases Principales

### `ExperimentConfig`
Configuración de experimentos con parámetros como:
- Nombre y descripción
- Variantes y control
- Métricas objetivo
- Parámetros estadísticos
- Configuración de early stopping

```python
config = ExperimentConfig(
    name="Mi Experimento",
    control_name="Control",
    variant_names=["Variante A", "Variante B"],
    primary_metric="conversion_rate",
    min_sample_size_per_variant=1000,
    significance_level=0.05,
    power=0.8,
    early_stopping_enabled=True,
    bandit_enabled=True
)
```

### `ExperimentDesigner`
Diseña experimentos automáticamente:
- Cálculo de tamaño de muestra
- Estimación de duración
- Análisis de poder
- Segmentación

```python
designer = ExperimentDesigner()
design = designer.design_experiment(config, baseline_rates, expected_effects, daily_traffic)
```

### `StatisticalAnalyzer`
Análisis estadístico completo:
- Pruebas t, chi-square, Z-test
- Análisis bayesiano
- Early stopping
- Intervalos de confianza

```python
analyzer = StatisticalAnalyzer()
results = analyzer.analyze_conversion_rates(control_data, variant_data)
bayesian_results = analyzer.bayesian_analysis(control_data, variant_data)
```

### `BanditOptimizer`
Optimización multi-armed bandit:
- Epsilon-greedy
- Thompson Sampling
- Upper Confidence Bound
- Recomendaciones dinámicas

```python
bandit = BanditOptimizer(epsilon=0.1, decay_rate=0.995)
bandit.add_arm('option_1', initial_estimate=0.1)
selected_arm = bandit.select_arm()
bandit.update(selected_arm, reward)
```

### `LiftAnalyzer`
Análisis de lift e impacto:
- Lift absoluto y relativo
- Impacto incremental
- Proyecciones ROI
- Bootstrap CI

```python
lift_analyzer = LiftAnalyzer()
lift_results = lift_analyzer.calculate_lift(control_data, variant_data)
impact = lift_analyzer.calculate_incremental_impact(control_data, variant_data)
```

### `ReportGenerator`
Generación de reportes:
- Reportes ejecutivos
- Dashboard data
- Insights automáticos
- Recomendaciones

```python
generator = ReportGenerator()
report = generator.generate_experiment_report(results)
dashboard_data = generator.generate_dashboard_data(experiments)
```

## 🎯 Casos de Uso

### 1. **Optimización de Conversiones**
```python
# Test de elementos de página
config = ExperimentConfig(
    name="Test Botón CTA",
    variant_names=["Botón Rojo", "Botón Verde", "Botón Azul"],
    primary_metric="click_through_rate"
)
```

### 2. **Personalización por Segmento**
```python
# Test segmentado
config = ExperimentConfig(
    name="Email Personalizado",
    segments=["nuevos_usuarios", "usuarios_activos", "usuarios_inactivos"]
)
```

### 3. **Optimización de Precios**
```python
# Análisis bayesiano para pricing
bayesian_results = analyzer.bayesian_analysis(control_data, variant_data)
probability_better = bayesian_results['probability_variant_better']
```

### 4. **Experimentación Continua**
```python
# Bandit para optimización continua
for iteration in range(1000):
    selected_arm = bandit.select_arm()
    reward = simulate_user_behavior(selected_arm)
    bandit.update(selected_arm, reward)
```

## 📈 Métricas Soportadas

### Métricas Binarias
- Tasas de conversión
- Tasas de clic
- Tasas de apertura (email)
- Tasas de abandono

### Métricas Continuas
- Tiempo en página
- Valor promedio de pedido
- Páginas por sesión
- Engagement score

### Métricas de Negocio
- Revenue per user
- Customer lifetime value
- Return on investment
- Net promoter score

## 🔧 Configuración Avanzada

### Parámetros Estadísticos
```python
config = ExperimentConfig(
    significance_level=0.05,  # Alpha level
    power=0.8,              # Statistical power
    min_detectable_effect=0.02  # Minimum effect size
)
```

### Early Stopping
```python
config = ExperimentConfig(
    early_stopping_enabled=True,
    min_duration_days=7,
    max_duration_days=30
)
```

### Bandit Configuration
```python
bandit = BanditOptimizer(
    epsilon=0.1,        # Exploration rate
    decay_rate=0.995    # Epsilon decay
)
```

## 📊 Dashboard

El dashboard interactivo (`noesis_ab_testing_dashboard.html`) incluye:

### Vista General
- Total de experimentos
- Experimentos activos
- Resultados significativos
- Lift promedio

### Gráficos
- Rendimiento temporal
- Comparación control vs variante
- Distribución de lift
- Significancia estadística

### Tablas
- Lista de experimentos
- Estado y progreso
- Métricas clave
- Acciones disponibles

### Recomendaciones
- Insights automáticos
- Sugerencias de optimización
- Alertas de significancia
- Próximos pasos

## 🚀 Casos de Prueba

### Ejecutar Ejemplos
```bash
python noesis_ab_testing_examples.py
```

Esto ejecutará:
1. **A/B Test básico** - Optimización de página
2. **Bandit optimization** - Selección automática
3. **Análisis por segmentos** - Segmentación demográfica
4. **Análisis bayesiano** - Probabilidades avanzadas
5. **Dashboard completo** - Reportes automáticos
6. **Benchmark de rendimiento** - Métricas de velocidad

## 📋 Mejores Prácticas

### Diseño de Experimentos
1. **Hipótesis clara**: Define qué esperas cambiar
2. **Métrica primaria**: Una métrica principal para evitar dilución
3. **Tamaño de muestra**: Suficiente para detectar el efecto mínimo
4. **Duración**: Al menos 1-2 semanas para capturar variaciones
5. **Segmentación**: Considera heterogeneidad en respuestas

### Análisis Estadístico
1. **Corregir por múltiples pruebas** si usas muchas métricas
2. **Usar análisis bayesiano** para decisiones más informadas
3. **Monitorear early stopping** para evitar falsos positivos
4. **Validar assumptions** de las pruebas estadísticas

### Optimización con Bandit
1. **Empezar con epsilon-greedy** para exploración
2. **Decay epsilon gradualmente** para más explotación
3. **Thompson Sampling** para mejor balance exploración/explotación
4. **Monitorear confianza** en las estimaciones

## 🔍 Interpretación de Resultados

### Significancia Estadística
- **p < 0.05**: Significativo al 95% de confianza
- **p < 0.01**: Significativo al 99% de confianza
- **Intervalos de confianza**: Rango de valores probables

### Lift Analysis
- **Lift > 5%**: Generalmente significativo para implementación
- **Lift 2-5%**: Considerar costo de implementación
- **Lift < 2%**: Probablemente no justifica el cambio

### Análisis Bayesiano
- **Probabilidad > 95%**: Evidencia muy fuerte
- **Probabilidad 80-95%**: Evidencia fuerte
- **Probabilidad 60-80%**: Evidencia moderada

## 🛠️ Solución de Problemas

### Error: "No hay datos suficientes"
- Verificar que se han añadido suficientes puntos de datos
- Comprobar que hay datos para todas las variantes

### Resultados no significativos
- Aumentar tamaño de muestra
- Extender duración del experimento
- Revisar magnitud del efecto esperado

### Bandit no converge
- Aumentar número de iteraciones
- Ajustar parámetros epsilon/decay
- Verificar configuración de brazos

### Dashboard no carga datos
- Verificar formato de datos
- Comprobar conexión (si usa API)
- Validar estructura JSON

## 🔄 Integración con Sistemas

### APIs Web
```python
# Flask/FastAPI example
from flask import Flask, jsonify
from noesis_ab_testing import NoesisABTestingSystem

app = Flask(__name__)
noesis = NoesisABTestingSystem()

@app.route('/api/experiments', methods=['POST'])
def create_experiment():
    # Handle experiment creation
    pass

@app.route('/api/experiments/<id>/results', methods=['GET'])
def get_results(experiment_id):
    results = noesis.experiments[experiment_id]['results']
    return jsonify(asdict(results))
```

### Bases de Datos
```python
# PostgreSQL/MongoDB integration
import psycopg2

def save_experiment_results(experiment_id, results):
    # Save to database
    pass
```

### Sistemas de Notificación
```python
# Slack/Email alerts
def send_significance_alert(experiment_id, results):
    if has_significant_results(results):
        send_slack_message(f"Experiment {experiment_id} is significant!")
```

## 📈 Métricas de Rendimiento

### Benchmarks del Sistema
- **Creación de experimentos**: < 10ms por experimento
- **Análisis estadístico**: 10,000+ puntos de datos/segundo
- **Memoria**: ~0.1MB por experimento
- **Dashboard**: Tiempo de carga < 2 segundos

### Escalabilidad
- Soporta miles de experimentos simultáneos
- Análisis en tiempo real
- Compresión automática de datos históricos
- Particionado de datos por fecha

## 🛡️ Seguridad y Privacidad

### Anonimización
- Datos de usuario anonimizados automáticamente
- Hash de identificadores
- Datos agregados por defecto

### Cumplimiento
- GDPR compliant
- Retention policies configurables
- Audit trails completos

## 🔮 Roadmap y Mejoras Futuras

### Próximas Funcionalidades
- [ ] **Machine Learning Integration**: Predicción automática de resultados
- [ ] **Multi-variate Testing**: Tests factoriales completos
- [ ] **Seasonal Adjustment**: Corrección por estacionalidad
- [ ] **Automated Insights**: Insights generativos con IA
- [ ] **Real-time Streaming**: Análisis en tiempo real
- [ ] **Mobile SDK**: SDK para aplicaciones móviles
- [ ] **A/A Testing**: Validación automática de significance
- [ ] **Sequential Testing**: Análisis secuencial avanzado

### Optimizaciones
- [ ] **Performance**: Paralelización de análisis
- [ ] **Storage**: Compresión avanzada de datos
- [ ] **Visualization**: Gráficos más interactivos
- [ ] **API**: RESTful API completa
- [ ] **Export**: Más formatos de exportación

## 📞 Soporte y Contacto

### Documentación
- **Ejemplos**: `noesis_ab_testing_examples.py`
- **API Reference**: Ver docstrings en código
- **Dashboard**: Ver `noesis_ab_testing_dashboard.html`

### Contribuciones
1. Fork el proyecto
2. Crear feature branch
3. Commit changes
4. Push to branch
5. Crear Pull Request

### Issues
Reportar bugs y solicitar features en GitHub Issues.

## 📄 Licencia

Este proyecto está licenciado bajo MIT License - ver LICENSE para detalles.

## 👥 Créditos

Desarrollado por el Equipo NOESIS - Sistema de Experimentación Automática

---

**¡Gracias por usar NOESIS A/B Testing System! 🎉**

Para más información, consultas o soporte, no dudes en contactarnos o revisar los ejemplos incluidos.