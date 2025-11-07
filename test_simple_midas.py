#!/usr/bin/env python3
"""
Script de prueba simple para MIDAS Auto Optimization
"""

import sys
import traceback
from datetime import datetime, timedelta

# Importar el módulo principal
try:
    from midas_auto_optimization import (
        MIDASAutoOptimization, 
        CampaignPerformance, 
        CreativeVariant,
        OptimizationRule
    )
    print("✓ Importación exitosa del módulo MIDAS")
except ImportError as e:
    print(f"✗ Error importando módulo: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Prueba funcionalidad básica"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # 1. Inicializar sistema
        midas = MIDASAutoOptimization()
        print("✓ Sistema inicializado")
        
        # 2. Crear datos de prueba
        performance_data = []
        for i in range(30):
            perf = CampaignPerformance(
                campaign_id="test_camp",
                date=datetime.now() - timedelta(days=i),
                impressions=1000 + i * 20,
                clicks=50 + i,
                conversions=5 + i // 5,
                cost=100.0 + i * 2,
                revenue=200.0 + i * 5,
                ctr=0.05,
                cvr=0.04,
                cpa=20.0,
                roas=2.0,
                hour=10 + i % 12,
                day_of_week=i % 7
            )
            performance_data.append(perf)
        
        print(f"✓ Datos de prueba creados: {len(performance_data)} registros")
        
        # 3. Probar Bid Optimizer
        training_result = midas.bid_optimizer.train_bid_model("test_camp", performance_data)
        print(f"✓ Bid Optimizer entrenado: {training_result['status']}")
        
        # 4. Probar optimización de bid
        context = {
            'hour': 14,
            'day_of_week': 2,
            'impressions': 1200,
            'clicks': 60,
            'conversions': 6,
            'cost': 120.0,
            'ctr': 0.05,
            'cvr': 0.05,
            'cpa': 20.0,
            'roas': 2.0
        }
        
        bid_result = midas.bid_optimizer.optimize_bids("test_camp", 1.50, context)
        print(f"✓ Bid optimizado: ${bid_result['optimized_bid']} (cambio: {bid_result['change_percentage']}%)")
        
        # 5. Probar Performance Predictor
        prediction = midas.performance_predictor.predict_performance("test_camp", context, days_ahead=7)
        print(f"✓ Performance predicha: ROAS {prediction['average_roas']:.3f}, CTR {prediction['average_ctr']:.4f}")
        
        # 6. Probar Rule Engine
        rule = OptimizationRule(
            rule_id="test_rule",
            name="Test Rule",
            condition="roas < 2.0",
            action="reduce_bid",
            priority=5,
            is_active=True
        )
        
        rule_result = midas.rule_engine.add_rule(rule)
        print(f"✓ Regla añadida: {rule_result['status']}")
        
        # 7. Probar Creative Optimizer
        creative_variants = [
            CreativeVariant("creative_1", "test_camp", "Variant A", 5000, 250, 25, 500.0, 0.05, 0.05),
            CreativeVariant("creative_2", "test_camp", "Variant B", 4800, 216, 20, 480.0, 0.045, 0.042)
        ]
        
        ab_test = midas.creative_optimizer.create_ab_test("test_camp", creative_variants)
        print(f"✓ Test A/B creado: {ab_test['status']}")
        
        # 8. Probar optimizador de horarios
        analysis = midas.dayparting_optimizer.analyze_hourly_performance(performance_data)
        print(f"✓ Análisis horario completado: {analysis['total_hours_analyzed']} horas analizadas")
        
        print("\n🎉 Todas las pruebas básicas pasaron exitosamente!")
        return True
        
    except Exception as e:
        print(f"✗ Error en pruebas básicas: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Prueba de integración completa"""
    print("\n=== Testing Full Integration ===")
    
    try:
        midas = MIDASAutoOptimization()
        
        # Crear datos de muestra
        campaign_data = {
            "camp_001": {
                "current_bid": 1.50,
                "context": {
                    "hour": 14, "day_of_week": 2, "impressions": 1200, "clicks": 60,
                    "conversions": 6, "cost": 120.0, "ctr": 0.05, "cvr": 0.05, "cpa": 20.0, "roas": 2.0
                },
                "performance_data": {"roas": 2.0, "ctr": 0.05, "cvr": 0.05, "cpa": 20.0, "impressions": 1200},
                "current_budget": 500.0
            }
        }
        
        # Ejecutar optimización completa
        results = midas.run_full_optimization(campaign_data)
        
        if results['status'] == 'success':
            print(f"✓ Optimización completa exitosa:")
            print(f"  - Campañas procesadas: {results['campaigns_processed']}")
            print(f"  - Campañas optimizadas: {results['summary']['optimized_campaigns']}")
            print(f"  - Tasa de optimización: {results['summary']['optimization_rate']:.1%}")
        
        # Generar dashboard
        dashboard = midas.get_optimization_dashboard(1)
        print(f"✓ Dashboard generado para {dashboard['period']}")
        
        print("🎉 Integración completa exitosa!")
        return True
        
    except Exception as e:
        print(f"✗ Error en integración: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== MIDAS Auto Optimization - Test Suite ===")
    print(f"Fecha: {datetime.now()}")
    
    # Ejecutar pruebas
    basic_test = test_basic_functionality()
    integration_test = test_integration()
    
    # Resumen
    total_tests = 2
    passed_tests = sum([basic_test, integration_test])
    
    print(f"\n=== Resultados ===")
    print(f"Pruebas ejecutadas: {total_tests}")
    print(f"Pruebas exitosas: {passed_tests}")
    print(f"Tasa de éxito: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ¡Sistema MIDAS listo para producción!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} pruebas fallaron")
    
    sys.exit(0 if passed_tests == total_tests else 1)