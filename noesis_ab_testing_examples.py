#!/usr/bin/env python3
"""
NOESIS A/B Testing - Ejemplos de Uso y Pruebas

Este archivo demuestra el uso completo del sistema de A/B testing
incluyendo creación de experimentos, análisis estadístico, 
optimización bandit y generación de reportes.

Uso:
    python noesis_ab_testing_examples.py

Autor: Sistema NOESIS
Fecha: 2025-11-06
"""

import sys
import os
import json
import time
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Añadir el directorio actual al path para importar el módulo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from noesis_ab_testing import (
    NoesisABTestingSystem,
    ExperimentConfig,
    ExperimentResults,
    create_dashboard_visualization
)

def print_separator(title=""):
    """Imprime un separador visual"""
    print("\n" + "="*80)
    if title:
        print(f" {title} ".center(80, "="))
    print("="*80)

def simulate_user_behavior(base_conversion_rate, noise_std=0.02):
    """Simula el comportamiento de un usuario"""
    return max(0, min(1, base_conversion_rate + np.random.normal(0, noise_std)))

def run_example_1_basic_ab_test():
    """Ejemplo 1: A/B Test básico de página de producto"""
    print_separator("EJEMPLO 1: A/B Test Básico - Página de Producto")
    
    # Crear sistema
    noesis = NoesisABTestingSystem()
    
    # Configurar experimento
    config = ExperimentConfig(
        name="Optimización de Página de Producto",
        description="Probando diferentes layouts de página de producto para mejorar conversiones",
        control_name="Layout Original",
        variant_names=["Layout Mejorado", "Layout Minimalista"],
        primary_metric="conversion_rate",
        secondary_metrics=["add_to_cart_rate", "time_on_page"],
        min_sample_size_per_variant=1000,
        significance_level=0.05,
        power=0.8,
        min_detectable_effect=0.1,
        early_stopping_enabled=True,
        segments=["new_users", "returning_users", "all"]
    )
    
    # Simular datos de baseline
    baseline_rates = {
        'control': 0.08,  # 8% tasa de conversión base
        'Layout Mejorado': 0.12,  # Mejora esperada 50%
        'Layout Minimalista': 0.09  # Mejora esperada 12.5%
    }
    
    expected_effects = {
        'Layout Mejorado': 0.5,
        'Layout Minimalista': 0.125
    }
    
    # Crear experimento
    experiment_id = noesis.create_experiment(
        config=config,
        baseline_rates=baseline_rates,
        expected_effects=expected_effects,
        daily_traffic=5000
    )
    
    print(f"✓ Experimento creado: {experiment_id}")
    print(f"  - Tamaño de muestra estimado: {noesis.experiments[experiment_id]['design']['sample_sizes']}")
    print(f"  - Duración estimada: {noesis.experiments[experiment_id]['design']['estimated_duration_days']} días")
    
    # Iniciar experimento
    noesis.start_experiment(experiment_id)
    print(f"✓ Experimento iniciado")
    
    # Simular datos durante 7 días
    print("\n📊 Simulando recolección de datos (7 días)...")
    for day in range(7):
        print(f"  Día {day + 1}: ", end="")
        
        # Simular datos para cada variante
        for variant_name, baseline_rate in baseline_rates.items():
            if variant_name == 'control':
                continue
                
            daily_users = 5000 // len(baseline_rates)
            
            for _ in range(daily_users):
                # Simular conversión con ligera variación diaria
                daily_variation = np.random.normal(1.0, 0.1)
                conversion_prob = baseline_rate * daily_variation
                
                # Añadir punto de datos
                is_conversion = np.random.random() < conversion_prob
                noesis.add_data_point(
                    experiment_id=experiment_id,
                    variant="Layout Mejorado" if variant_name == "Layout Mejorado" else "Layout Minimalista",
                    data_point=1.0 if is_conversion else 0.0
                )
            
            print(f"{variant_name}: {daily_users} usuarios", end=" | ")
        
        print()
        
        # Simular datos para control
        daily_control_users = 5000 // len(baseline_rates)
        for _ in range(daily_control_users):
            is_conversion = np.random.random() < baseline_rates['control']
            noesis.add_data_point(
                experiment_id=experiment_id,
                variant="control",
                data_point=1.0 if is_conversion else 0.0
            )
    
    # Analizar resultados
    print("\n🔍 Analizando resultados...")
    results = noesis.analyze_experiment(experiment_id)
    
    print(f"✓ Análisis completado")
    print(f"  - Estado: {results.status}")
    print(f"  - Early stopping: {'Sí' if results.early_stopping_triggered else 'No'}")
    
    # Mostrar resultados clave
    if results.statistical_results:
        control_rate = results.statistical_results.get('control', {}).get('rate', 0)
        variant_rate = results.statistical_results.get('variant', {}).get('rate', 0)
        lift = results.lift_analysis.get('lift_analysis', {}).get('percent_lift', 0)
        
        print(f"  - Tasa control: {control_rate:.4f}")
        print(f"  - Tasa variante: {variant_rate:.4f}")
        print(f"  - Lift: {lift:.2f}%")
    
    # Generar reporte
    print("\n📄 Generando reporte...")
    report = noesis.get_experiment_report(experiment_id)
    
    # Guardar reporte
    report_file = f"reporte_ejemplo_1_{experiment_id}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Reporte guardado: {report_file}")
    
    return experiment_id, results

def run_example_2_bandit_optimization():
    """Ejemplo 2: Optimización con Multi-Armed Bandit"""
    print_separator("EJEMPLO 2: Optimización Multi-Armed Bandit")
    
    noesis = NoesisABTestingSystem()
    
    # Configurar variantes para bandit
    arm_performance = {
        'homepage_v1': 0.08,  # Original
        'homepage_v2': 0.12,  # Mejorado
        'homepage_v3': 0.06,  # Peor
        'homepage_v4': 0.10   # Bueno
    }
    
    print("🎯 Simulando optimización con Multi-Armed Bandit")
    print("Variantes disponibles:")
    for arm, performance in arm_performance.items():
        print(f"  - {arm}: {performance:.1%} conversión esperada")
    
    # Simular proceso de optimización
    total_traffic = 10000
    results_by_arm = {arm: {'traffic': 0, 'conversions': 0} for arm in arm_performance.keys()}
    
    print(f"\n📊 Simulando {total_traffic:,} usuarios...")
    
    for i in range(total_traffic):
        # Seleccionar brazo usando bandit
        selected_arm = noesis.optimize_with_bandit(arm_performance)
        
        # Simular conversión
        expected_rate = arm_performance[selected_arm]
        is_conversion = np.random.random() < expected_rate
        
        # Actualizar bandit con resultado
        noesis.bandit_optimizer.update(selected_arm, 1.0 if is_conversion else 0.0)
        
        # Registrar estadísticas
        results_by_arm[selected_arm]['traffic'] += 1
        if is_conversion:
            results_by_arm[selected_arm]['conversions'] += 1
        
        # Mostrar progreso cada 1000 usuarios
        if (i + 1) % 1000 == 0:
            print(f"  Progreso: {i + 1:,}/{total_traffic:,} usuarios")
    
    # Mostrar resultados finales
    print("\n📈 Resultados finales del Bandit:")
    recommendations = noesis.bandit_optimizer.get_recommendations()
    
    for rec in recommendations:
        arm_id = rec['arm_id']
        stats = results_by_arm[arm_id]
        actual_rate = stats['conversions'] / stats['traffic'] if stats['traffic'] > 0 else 0
        
        print(f"\n{arm_id}:")
        print(f"  - Tráfico asignado: {stats['traffic']:,} usuarios")
        print(f"  - Conversiones: {stats['conversions']:,}")
        print(f"  - Tasa real: {actual_rate:.3%}")
        print(f"  - Estimación bandit: {rec['estimated_value']:.3%}")
        print(f"  - Score de confianza: {rec['confidence_score']:.2f}")
        print(f"  - ¿Recomendado?: {'Sí' if rec['is_recommended'] else 'No'}")
    
    # Identificar mejor brazo
    best_arm = max(recommendations, key=lambda x: x['estimated_value'])
    print(f"\n🏆 Mejor brazo identificado: {best_arm['arm_id']}")
    print(f"   Tasa estimada: {best_arm['estimated_value']:.3%}")
    
    return recommendations

def run_example_3_advanced_segmentation():
    """Ejemplo 3: Análisis avanzado por segmentos"""
    print_separator("EJEMPLO 3: Análisis por Segmentos Avanzado")
    
    noesis = NoesisABTestingSystem()
    
    # Configurar experimento con múltiples segmentos
    config = ExperimentConfig(
        name="Test de Email Marketing por Segmento",
        description="Probando diferentes líneas de asunto por segmento demográfico",
        control_name="Email Original",
        variant_names=["Email Personalizado", "Email Urgente"],
        primary_metric="open_rate",
        segments=["edad_18_25", "edad_26_40", "edad_41_55", "edad_55_plus", "all"],
        min_sample_size_per_variant=500,
        early_stopping_enabled=False,  # Para análisis completo
        bandit_enabled=True
    )
    
    # Tasas de conversión por segmento (simuladas)
    segment_rates = {
        'edad_18_25': {
            'control': 0.15, 'Email Personalizado': 0.22, 'Email Urgente': 0.18
        },
        'edad_26_40': {
            'control': 0.20, 'Email Personalizado': 0.28, 'Email Urgente': 0.25
        },
        'edad_41_55': {
            'control': 0.25, 'Email Personalizado': 0.30, 'Email Urgente': 0.22
        },
        'edad_55_plus': {
            'control': 0.18, 'Email Personalizado': 0.20, 'Email Urgente': 0.15
        },
        'all': {
            'control': 0.20, 'Email Personalizado': 0.26, 'Email Urgente': 0.21
        }
    }
    
    print("👥 Simulando análisis por segmentos demográficos")
    print("Segmentos configurados:")
    for segment in config.segments:
        print(f"  - {segment}")
    
    # Crear y ejecutar experimento
    baseline_rates = {k: v['control'] for k, v in segment_rates.items() if k != 'all'}
    expected_effects = {
        'Email Personalizado': 0.3,  # 30% mejora esperada
        'Email Urgente': 0.1        # 10% mejora esperada
    }
    
    experiment_id = noesis.create_experiment(
        config=config,
        baseline_rates=baseline_rates,
        expected_effects=expected_effects,
        daily_traffic=2000
    )
    
    noesis.start_experiment(experiment_id)
    
    # Simular datos para cada segmento
    print(f"\n📊 Simulando datos por segmento...")
    
    for segment, rates in segment_rates.items():
        print(f"\nSegmento: {segment}")
        
        segment_sample_size = 1000  # Por variante
        
        for variant_name in ['Email Personalizado', 'Email Urgente']:
            for _ in range(segment_sample_size):
                conversion_prob = rates[variant_name]
                is_conversion = np.random.random() < conversion_prob
                
                noesis.add_data_point(
                    experiment_id=experiment_id,
                    variant=variant_name,
                    data_point=1.0 if is_conversion else 0.0
                )
        
        # También para control
        for _ in range(segment_sample_size):
            conversion_prob = rates['control']
            is_conversion = np.random.random() < conversion_prob
            
            noesis.add_data_point(
                experiment_id=experiment_id,
                variant='control',
                data_point=1.0 if is_conversion else 0.0
            )
    
    # Analizar por segmento
    print(f"\n🔍 Analizando resultados por segmento...")
    results = noesis.analyze_experiment(experiment_id)
    
    # Mostrar resultados por segmento
    print("\n📈 Resultados por Segmento:")
    if results.segment_results:
        for segment, segment_data in results.segment_results.items():
            print(f"\n{segment}:")
            print(f"  - Lift: {segment_data.get('lift', 0):.3%}")
            print(f"  - Tasa conversión: {segment_data.get('conversion_rate', 0):.3%}")
            print(f"  - Tamaño muestra: {segment_data.get('sample_size', 0):,}")
    
    # Generar insights segmentados
    print(f"\n💡 Insights por Segmento:")
    for segment, segment_data in results.segment_results.items():
        if segment == 'all':
            continue
            
        lift = segment_data.get('lift', 0)
        if lift > 0.05:  # Lift significativo
            print(f"  - {segment}: Excelente respuesta (+{lift:.1%}) - Recomendado para targeting")
        elif lift > 0.02:
            print(f"  - {segment}: Buena respuesta (+{lift:.1%}) - Considerar implementación")
        elif lift < -0.02:
            print(f"  - {segment}: Respuesta negativa ({lift:.1%}) - Evitar en este segmento")
        else:
            print(f"  - {segment}: Sin diferencia significativa ({lift:.1%})")
    
    return experiment_id, results

def run_example_4_bayesian_analysis():
    """Ejemplo 4: Análisis Bayesiano Avanzado"""
    print_separator("EJEMPLO 4: Análisis Bayesiano y Probabilidades")
    
    noesis = NoesisABTestingSystem()
    
    # Configuración
    config = ExperimentConfig(
        name="Test de Precio Dinámico",
        description="Probando diferentes estrategias de precios con análisis bayesiano",
        control_name="Precio Fijo",
        variant_names=["Precio Dinámico"],
        primary_metric="purchase_rate",
        significance_level=0.05,
        power=0.8
    )
    
    # Crear experimento
    experiment_id = noesis.create_experiment(
        config=config,
        baseline_rates={'control': 0.05},  # 5% tasa base
        expected_effects={'Precio Dinámico': 0.2},  # 20% mejora
        daily_traffic=3000
    )
    
    noesis.start_experiment(experiment_id)
    
    # Simular con efectos variables en el tiempo
    print("💰 Simulando test de precio dinámico...")
    
    control_rates_over_time = []
    variant_rates_over_time = []
    
    for day in range(10):
        # Simular variación en el tiempo
        time_factor = 1.0 + 0.1 * np.sin(day / 2)  # Variación sinusoidal
        
        control_daily_rate = 0.05 * time_factor
        variant_daily_rate = 0.06 * time_factor  # Mejora consistente
        
        control_rates_over_time.append(control_daily_rate)
        variant_rates_over_time.append(variant_daily_rate)
        
        # Simular usuarios diarios
        daily_users_per_variant = 1500
        
        for _ in range(daily_users_per_variant):
            # Control
            is_conversion = np.random.random() < control_daily_rate
            noesis.add_data_point(experiment_id, 'control', 
                                1.0 if is_conversion else 0.0)
            
            # Variante
            is_conversion = np.random.random() < variant_daily_rate
            noesis.add_data_point(experiment_id, 'Precio Dinámico', 
                                1.0 if is_conversion else 0.0)
        
        print(f"  Día {day + 1}: Control {control_daily_rate:.3%}, Variante {variant_daily_rate:.3%}")
    
    # Análisis bayesiano
    print("\n🧮 Realizando análisis bayesiano...")
    
    # Obtener datos para análisis bayesiano
    experiment_data = noesis.experiments[experiment_id]['data']
    control_data = [d['value'] for d in experiment_data['control']]
    variant_data = [d['value'] for d in experiment_data['variants']['Precio Dinámico']]
    
    # Realizar análisis bayesiano
    bayesian_results = noesis.statistical_analyzer.bayesian_analysis(
        control_data, variant_data,
        prior_alpha=1, prior_beta=1  # Prior uniforme
    )
    
    print("✓ Análisis bayesiano completado")
    
    # Mostrar resultados bayesianos
    print("\n📊 Resultados Bayesianos:")
    
    prob_variant_better = bayesian_results['probability_variant_better']
    expected_lift = bayesian_results['expected_lift']
    confidence_level = bayesian_results['confidence_level']
    
    print(f"  - Probabilidad de que la variante sea mejor: {prob_variant_better:.1%}")
    print(f"  - Lift esperado: {expected_lift:.2%}")
    print(f"  - Nivel de confianza: {confidence_level}")
    
    # Intervalos creíbles
    control_ci = bayesian_results['posterior_distributions']['control']['credible_interval']
    variant_ci = bayesian_results['posterior_distributions']['variant']['credible_interval']
    
    print(f"  - Intervalo creíble Control 95%: [{control_ci[0]:.3%}, {control_ci[1]:.3%}]")
    print(f"  - Intervalo creíble Variante 95%: [{variant_ci[0]:.3%}, {variant_ci[1]:.3%}]")
    
    # Interpretación
    print(f"\n💡 Interpretación Bayesiana:")
    if prob_variant_better > 0.95:
        print("  → Hay evidencia muy fuerte de que la variante es superior")
    elif prob_variant_better > 0.8:
        print("  → Hay evidencia fuerte de que la variante es superior")
    elif prob_variant_better > 0.6:
        print("  → Hay evidencia moderada de que la variante es superior")
    else:
        print("  → La evidencia no es concluyente")
    
    return experiment_id, bayesian_results

def run_example_5_comprehensive_dashboard():
    """Ejemplo 5: Dashboard y Reportes Completos"""
    print_separator("EJEMPLO 5: Dashboard y Reportes Automáticos")
    
    # Crear múltiples experimentos para el dashboard
    noesis = NoesisABTestingSystem()
    
    print("📈 Generando dashboard con múltiples experimentos...")
    
    # Simular varios experimentos
    experiments_created = []
    
    # Experimento 1: Landing page
    config1 = ExperimentConfig(
        name="Test Landing Page Hero",
        control_name="Hero Original",
        variant_names=["Hero Video", "Hero Imagen"],
        primary_metric="signup_rate"
    )
    
    exp1_id = noesis.create_experiment(
        config1, {'control': 0.12}, {'Hero Video': 0.25, 'Hero Imagen': 0.15}, 2000
    )
    noesis.start_experiment(exp1_id)
    
    # Simular datos exitosos
    for _ in range(2000):
        noesis.add_data_point(exp1_id, 'control', np.random.binomial(1, 0.12))
        noesis.add_data_point(exp1_id, 'Hero Video', np.random.binomial(1, 0.15))
    
    results1 = noesis.analyze_experiment(exp1_id)
    experiments_created.append(results1)
    
    # Experimento 2: Checkout process
    config2 = ExperimentConfig(
        name="Optimización Checkout",
        control_name="Checkout Largo",
        variant_names=["Checkout Corto"],
        primary_metric="completion_rate"
    )
    
    exp2_id = noesis.create_experiment(
        config2, {'control': 0.65}, {'Checkout Corto': 0.78}, 1500
    )
    noesis.start_experiment(exp2_id)
    
    for _ in range(1500):
        noesis.add_data_point(exp2_id, 'control', np.random.binomial(1, 0.65))
        noesis.add_data_point(exp2_id, 'Checkout Corto', np.random.binomial(1, 0.78))
    
    results2 = noesis.analyze_experiment(exp2_id)
    experiments_created.append(results2)
    
    # Experimento 3: Email subject
    config3 = ExperimentConfig(
        name="Asunto Email",
        control_name="Asunto Formal",
        variant_names=["Asunto Casual", "Asunto Urgente"],
        primary_metric="open_rate"
    )
    
    exp3_id = noesis.create_experiment(
        config3, {'control': 0.18}, {'Asunto Casual': 0.22, 'Asunto Urgente': 0.19}, 3000
    )
    noesis.start_experiment(exp3_id)
    
    for _ in range(1500):
        noesis.add_data_point(exp3_id, 'control', np.random.binomial(1, 0.18))
        noesis.add_data_point(exp3_id, 'Asunto Casual', np.random.binomial(1, 0.22))
    
    results3 = noesis.analyze_experiment(exp3_id)
    experiments_created.append(results3)
    
    # Generar datos del dashboard
    dashboard_data = noesis.get_dashboard_data()
    
    print("✓ Dashboard generado")
    print(f"\n📊 Resumen del Dashboard:")
    print(f"  - Total experimentos: {dashboard_data['summary']['total_experiments']}")
    print(f"  - Experimentos activos: {dashboard_data['summary']['active_experiments']}")
    print(f"  - Experimentos completados: {dashboard_data['summary']['completed_experiments']}")
    print(f"  - Resultados significativos: {dashboard_data['summary']['significant_results']}")
    
    # Generar visualización del dashboard
    dashboard_viz = create_dashboard_visualization(dashboard_data)
    
    # Guardar reporte del dashboard
    dashboard_report = f"dashboard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(dashboard_report, 'w', encoding='utf-8') as f:
        f.write(dashboard_viz)
    
    print(f"✓ Reporte del dashboard guardado: {dashboard_report}")
    
    # Generar resumen ejecutivo
    print(f"\n📋 Resumen Ejecutivo de Experimentos:")
    for i, exp in enumerate(experiments_created, 1):
        lift = exp.lift_analysis.get('lift_analysis', {}).get('percent_lift', 0)
        status = exp.status
        significant = any(test.get('significant', False) 
                         for test in exp.statistical_results.get('statistical_tests', {}).values())
        
        print(f"\nExperimento {i} ({exp.experiment_id}):")
        print(f"  - Estado: {status}")
        print(f"  - Lift: {lift:.2f}%")
        print(f"  - Significativo: {'Sí' if significant else 'No'}")
        
        if exp.recommendations:
            print(f"  - Recomendación principal: {exp.recommendations[0]}")
    
    return dashboard_data, experiments_created

def run_performance_benchmark():
    """Benchmark de rendimiento del sistema"""
    print_separator("BENCHMARK DE RENDIMIENTO")
    
    noesis = NoesisABTestingSystem()
    
    # Test de velocidad de creación de experimentos
    start_time = time.time()
    for i in range(100):
        config = ExperimentConfig(
            name=f"Benchmark Test {i}",
            variant_names=["Variant A", "Variant B"]
        )
        exp_id = noesis.create_experiment(
            config, {'control': 0.1}, {'Variant A': 0.15, 'Variant B': 0.12}, 1000
        )
    creation_time = time.time() - start_time
    
    print(f"⚡ Creación de 100 experimentos: {creation_time:.2f} segundos")
    print(f"   Promedio: {creation_time/100*1000:.1f} ms por experimento")
    
    # Test de velocidad de análisis
    config = ExperimentConfig(name="Performance Test", variant_names=["Test Variant"])
    exp_id = noesis.create_experiment(config, {'control': 0.1}, {'Test Variant': 0.15}, 1000)
    noesis.start_experiment(exp_id)
    
    # Añadir datos
    for i in range(10000):
        noesis.add_data_point(exp_id, 'control', np.random.binomial(1, 0.1))
        noesis.add_data_point(exp_id, 'Test Variant', np.random.binomial(1, 0.15))
    
    # Benchmark de análisis
    start_time = time.time()
    results = noesis.analyze_experiment(exp_id)
    analysis_time = time.time() - start_time
    
    print(f"⚡ Análisis de 10,000 puntos de datos: {analysis_time:.3f} segundos")
    print(f"   Throughput: {10000/analysis_time:.0f} puntos de datos/segundo")
    
    # Test de memoria (estimado)
    total_experiments = len(noesis.experiments)
    print(f"⚡ Memoria estimada para {total_experiments} experimentos: {total_experiments * 0.1:.1f} MB")
    
    return {
        'creation_time': creation_time,
        'analysis_time': analysis_time,
        'total_experiments': total_experiments
    }

def main():
    """Función principal - ejecuta todos los ejemplos"""
    print("🚀 NOESIS A/B Testing System - Ejemplos Completos")
    print("Sistema de experimentación automática y análisis estadístico avanzado")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Ejemplos principales
        examples_results = {}
        
        # Ejemplo 1: A/B Test básico
        examples_results['basic'] = run_example_1_basic_ab_test()
        
        # Ejemplo 2: Bandit optimization
        examples_results['bandit'] = run_example_2_bandit_optimization()
        
        # Ejemplo 3: Segmentación
        examples_results['segmentation'] = run_example_3_advanced_segmentation()
        
        # Ejemplo 4: Análisis bayesiano
        examples_results['bayesian'] = run_example_4_bayesian_analysis()
        
        # Ejemplo 5: Dashboard
        examples_results['dashboard'] = run_example_5_comprehensive_dashboard()
        
        # Benchmark
        benchmark_results = run_performance_benchmark()
        
        # Resumen final
        print_separator("RESUMEN FINAL")
        print("✅ Todos los ejemplos ejecutados exitosamente")
        print("\n📊 Archivos generados:")
        
        import glob
        report_files = glob.glob("reporte_ejemplo_*.md") + glob.glob("dashboard_report_*.md")
        for file in report_files:
            print(f"  - {file}")
        
        print(f"\n⚡ Resultados de rendimiento:")
        print(f"  - Creación de experimentos: {benchmark_results['creation_time']:.2f}s")
        print(f"  - Análisis estadístico: {benchmark_results['analysis_time']:.3f}s")
        print(f"  - Total experimentos: {benchmark_results['total_experiments']}")
        
        print(f"\n🎯 Funcionalidades demostradas:")
        print("  ✓ Diseño automático de experimentos")
        print("  ✓ Análisis estadístico (t-test, chi-square, bayesiano)")
        print("  ✓ Multi-armed bandit para optimización")
        print("  ✓ Análisis de lift e impacto incremental")
        print("  ✓ Detección de early stopping")
        print("  ✓ Segmentación de usuarios")
        print("  ✓ Reportes automáticos")
        print("  ✓ Dashboard interactivo")
        
        print(f"\n🌐 Para ver el dashboard interactivo:")
        print("  1. Abrir noesis_ab_testing_dashboard.html en el navegador")
        print("  2. Los datos se simulan automáticamente")
        print("  3. Incluir archivos de reporte para datos reales")
        
        print(f"\n✨ El sistema NOESIS A/B Testing está listo para producción!")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()