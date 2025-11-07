#!/usr/bin/env python3
"""
NOESIS A/B Testing - Prueba Rápida del Sistema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from noesis_ab_testing import (
    NoesisABTestingSystem,
    ExperimentConfig,
    StatisticalAnalyzer,
    BanditOptimizer
)

def test_basic_functionality():
    """Prueba básica de funcionalidad"""
    print("🧪 NOESIS A/B Testing - Prueba del Sistema")
    print("=" * 50)
    
    # Test 1: Crear sistema
    print("\n1. ✅ Inicializando sistema...")
    noesis = NoesisABTestingSystem()
    print("   Sistema inicializado correctamente")
    
    # Test 2: Configuración de experimento
    print("\n2. ✅ Configurando experimento...")
    config = ExperimentConfig(
        name="Test de Botón",
        description="Probando diferentes colores de botón",
        control_name="Botón Azul",
        variant_names=["Botón Rojo", "Botón Verde"],
        primary_metric="conversion_rate",
        min_sample_size_per_variant=100,
        significance_level=0.05
    )
    print("   Configuración creada")
    
    # Test 3: Crear experimento
    print("\n3. ✅ Creando experimento...")
    experiment_id = noesis.create_experiment(
        config=config,
        baseline_rates={'control': 0.1},
        expected_effects={'Botón Rojo': 0.15, 'Botón Verde': 0.12},
        daily_traffic=1000
    )
    print(f"   Experimento creado: {experiment_id}")
    
    # Test 4: Iniciar experimento
    print("\n4. ✅ Iniciando experimento...")
    noesis.start_experiment(experiment_id)
    print("   Experimento iniciado")
    
    # Test 5: Añadir datos
    print("\n5. ✅ Añadiendo datos simulados...")
    import numpy as np
    
    # Simular datos para control
    for i in range(200):
        is_conversion = np.random.random() < 0.1
        noesis.add_data_point(experiment_id, 'control', 1.0 if is_conversion else 0.0)
    
    # Simular datos para variantes
    for variant in ['Botón Rojo', 'Botón Verde']:
        for i in range(200):
            if variant == 'Botón Rojo':
                prob = 0.115  # 15% mejor
            else:
                prob = 0.112  # 12% mejor
            
            is_conversion = np.random.random() < prob
            noesis.add_data_point(experiment_id, variant, 1.0 if is_conversion else 0.0)
    
    print("   Datos añadidos (200 por variante)")
    
    # Test 6: Análisis estadístico
    print("\n6. ✅ Analizando resultados...")
    results = noesis.analyze_experiment(experiment_id)
    print("   Análisis completado")
    
    # Test 7: Mostrar resultados
    print("\n7. 📊 Resultados:")
    if results.statistical_results:
        control_rate = results.statistical_results.get('control', {}).get('rate', 0)
        variant_rate = results.statistical_results.get('variant', {}).get('rate', 0)
        lift = results.lift_analysis.get('lift_analysis', {}).get('percent_lift', 0)
        
        print(f"   - Tasa control: {control_rate:.4f}")
        print(f"   - Tasa variante: {variant_rate:.4f}")
        print(f"   - Lift: {lift:.2f}%")
        print(f"   - Estado: {results.status}")
    
    # Test 8: Bandit optimizer
    print("\n8. 🤖 Probando Bandit Optimizer...")
    bandit = BanditOptimizer()
    
    # Añadir brazos
    bandit.add_arm('option_1', 0.1)
    bandit.add_arm('option_2', 0.15)
    bandit.add_arm('option_3', 0.08)
    
    # Simular selección y actualización
    selected = bandit.select_arm()
    print(f"   - Brazo seleccionado: {selected}")
    
    # Simular recompensas
    for i in range(10):
        arm = bandit.select_arm()
        reward = 1.0 if np.random.random() < 0.12 else 0.0
        bandit.update(arm, reward)
    
    recommendations = bandit.get_recommendations()
    print("   - Recomendaciones bandit generadas")
    
    # Test 9: Generar reporte
    print("\n9. 📄 Generando reporte...")
    report = noesis.get_experiment_report(experiment_id)
    print("   - Reporte generado")
    
    # Test 10: Dashboard data
    print("\n10. 📈 Generando datos de dashboard...")
    dashboard_data = noesis.get_dashboard_data()
    print("   - Datos de dashboard generados")
    
    print("\n" + "=" * 50)
    print("🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE!")
    print("=" * 50)
    
    # Resumen de archivos creados
    print(f"\n📁 Archivos disponibles:")
    print(f"   - noesis_ab_testing.py (sistema principal)")
    print(f"   - noesis_ab_testing_dashboard.html (dashboard)")
    print(f"   - noesis_ab_testing_examples.py (ejemplos completos)")
    print(f"   - README_NOESIS_AB_Testing.md (documentación)")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\n✅ Sistema NOESIS A/B Testing funcionando correctamente!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()