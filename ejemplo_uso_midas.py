#!/usr/bin/env python3
"""
Ejemplo práctico de uso del sistema MIDAS Auto Optimization
Este script demuestra cómo usar todas las funcionalidades principales del sistema
"""

import sys
import os
from datetime import datetime, timedelta

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from midas_auto_optimization import (
    MIDASAutoOptimization,
    CampaignPerformance,
    CreativeVariant,
    OptimizationRule,
    create_sample_campaign_data
)

def ejemplo_optimizacion_completa():
    """
    Ejemplo completo de optimización automática
    """
    print("🚀 === EJEMPLO COMPLETO: MIDAS AUTO OPTIMIZATION ===\n")
    
    # 1. INICIALIZACIÓN DEL SISTEMA
    print("1️⃣ Inicializando sistema MIDAS...")
    midas = MIDASAutoOptimization(noesis_api_key="demo_key_123")
    print("✓ Sistema inicializado con integración NOESIS")
    
    # 2. CREAR DATOS DE CAMPAÑA
    print("\n2️⃣ Creando datos de campañas de ejemplo...")
    campaign_data = create_sample_campaign_data()
    
    # Añadir más campañas para demostrar
    campaign_data["camp_002"] = {
        "current_bid": 2.00,
        "context": {
            "hour": 16, "day_of_week": 3, "impressions": 2000, "clicks": 80,
            "conversions": 8, "cost": 200.0, "ctr": 0.04, "cvr": 0.04, "cpa": 25.0, "roas": 2.4
        },
        "performance_data": {
            "roas": 2.4, "ctr": 0.04, "cvr": 0.04, "cpa": 25.0, "impressions": 2000
        },
        "current_budget": 800.0,
        "historical_performance": [
            CampaignPerformance(
                campaign_id="camp_002",
                date=datetime.now() - timedelta(days=i),
                impressions=1500 + i * 30,
                clicks=60 + i * 2,
                conversions=6 + i // 8,
                cost=150.0 + i * 3,
                revenue=360.0 + i * 6,
                ctr=0.04,
                cvr=0.05,
                cpa=25.0,
                roas=2.4,
                hour=14 + i % 8,
                day_of_week=i % 7
            ) for i in range(30)
        ]
    }
    
    campaign_data["camp_003"] = {
        "current_bid": 1.20,
        "context": {
            "hour": 20, "day_of_week": 5, "impressions": 800, "clicks": 32,
            "conversions": 3, "cost": 80.0, "ctr": 0.04, "cvr": 0.038, "cpa": 27.0, "roas": 1.8
        },
        "performance_data": {
            "roas": 1.8, "ctr": 0.04, "cvr": 0.038, "cpa": 27.0, "impressions": 800
        },
        "current_budget": 300.0
    }
    
    print(f"✓ {len(campaign_data)} campañas configuradas")
    
    # 3. CONFIGURAR REGLAS DE OPTIMIZACIÓN
    print("\n3️⃣ Configurando reglas de optimización automáticas...")
    
    reglas = [
        OptimizationRule(
            rule_id="rule_001",
            name="🔴 ALERTA: ROAS Crítico",
            condition="roas < 1.5",
            action="pause_campaign",
            priority=10,
            is_active=True
        ),
        OptimizationRule(
            rule_id="rule_002",
            name="🟡 ADVERTENCIA: ROAS Bajo",
            condition="roas < 2.0",
            action="reduce_bid",
            priority=8,
            is_active=True
        ),
        OptimizationRule(
            rule_id="rule_003",
            name="🟢 OPORTUNIDAD: Alto ROAS",
            condition="roas > 3.0",
            action="increase_bid",
            priority=7,
            is_active=True
        ),
        OptimizationRule(
            rule_id="rule_004",
            name="🔵 CTR Bajo",
            condition="ctr < 0.03",
            action="scale_budget",
            priority=6,
            is_active=True
        )
    ]
    
    for regla in reglas:
        resultado = midas.rule_engine.add_rule(regla)
        print(f"  ✓ {resultado['rule_name']}: {resultado['status']}")
    
    # 4. OPTIMIZACIÓN AUTOMÁTICA COMPLETA
    print("\n4️⃣ Ejecutando optimización automática completa...")
    resultados = midas.run_full_optimization(campaign_data)
    
    if resultados['status'] == 'success':
        print(f"🎯 OPTIMIZACIÓN COMPLETADA:")
        print(f"  • Campañas procesadas: {resultados['campaigns_processed']}")
        print(f"  • Campañas optimizadas: {resultados['summary']['optimized_campaigns']}")
        print(f"  • Tasa de optimización: {resultados['summary']['optimization_rate']:.1%}")
        print(f"  • Tipos aplicados: {', '.join(resultados['summary']['optimization_types_applied'])}")
        
        # Mostrar detalles por campaña
        print(f"\n📊 DETALLES POR CAMPAÑA:")
        for camp_id, opt in resultados['optimizations'].items():
            print(f"\n  🔸 {camp_id}:")
            print(f"    - Optimizaciones aplicadas: {', '.join(opt['optimizations_applied']) if opt['optimizations_applied'] else 'Ninguna'}")
            
            # Bid optimization
            if 'bid_optimization' in opt:
                bid = opt['bid_optimization']
                print(f"    - Bid: ${bid['current_bid']} → ${bid['optimized_bid']} ({bid['change_percentage']:+.1f}%)")
                print(f"    - Justificación: {bid['justification']}")
            
            # Performance prediction
            if 'performance_prediction' in opt:
                pred = opt['performance_prediction']
                print(f"    - ROAS predicho: {pred['average_roas']:.3f}")
                print(f"    - CTR predicho: {pred['average_ctr']:.4f}")
                print(f"    - Confianza: {pred['confidence']:.1%}")
            
            # Alertas
            if opt.get('recommendations'):
                print(f"    - Alertas/Recomendaciones: {len(opt['recommendations'])}")
                for rec in opt['recommendations'][:2]:  # Mostrar primeras 2
                    print(f"      • {rec.get('level', 'info').upper()}: {rec.get('message', '')}")
    
    # 5. ALLOCACIÓN DE BUDGET
    print("\n5️⃣ Analizando reallocation de budget...")
    if 'budget_reallocation' in resultados:
        budget_data = resultados['budget_reallocation']
        print(f"💰 BUDGET REALLOCATION:")
        print(f"  • Total budget: ${budget_data['total_budget']:.2f}")
        
        if 'current_allocations' in budget_data:
            print(f"  • Asignaciones actuales:")
            for camp, budget in budget_data['current_allocations'].items():
                print(f"    - {camp}: ${budget:.2f}")
        
        if 'optimal_allocations' in budget_data:
            print(f"  • Asignaciones optimizadas:")
            for camp, budget in budget_data['optimal_allocations'].items():
                current = budget_data['current_allocations'].get(camp, 0)
                cambio = (budget - current) / current * 100 if current > 0 else 0
                print(f"    - {camp}: ${budget:.2f} ({cambio:+.1f}%)")
        
        if 'reallocation_recommendations' in budget_data:
            realloc = budget_data['reallocation_recommendations']
            if realloc.get('recommendations'):
                print(f"  • Recomendaciones de reallocation: {realloc['total_recommendations']}")
                for rec in realloc['recommendations']:
                    print(f"    - {rec['campaign_id']}: {rec['change_percentage']:+.1f}% ({rec['reason']})")
    
    # 6. DASHBOARD Y MÉTRICAS
    print("\n6️⃣ Generando dashboard de métricas...")
    dashboard = midas.get_optimization_dashboard(7)
    
    print(f"📈 DASHBOARD DE OPTIMIZATION ({dashboard['period']}):")
    print(f"  • Total optimizaciones: {dashboard['total_optimizations']}")
    print(f"  • Promedio campañas/optimización: {dashboard['avg_campaigns_per_optimization']}")
    print(f"  • Accuracy de predicciones:")
    pred_acc = dashboard['prediction_accuracy']
    print(f"    - ROAS: {pred_acc['roas_prediction_accuracy']:.1%}")
    print(f"    - CTR: {pred_acc['ctr_prediction_accuracy']:.1%}")
    print(f"    - Overall: {pred_acc['overall_confidence']:.1%}")
    
    print(f"  • System Health:")
    health = dashboard['system_health']
    for component, status in health.items():
        icon = "✅" if status else "❌"
        print(f"    - {component}: {icon}")
    
    # 7. DEMOSTRACIÓN DE FUNCIONALIDADES AVANZADAS
    print("\n7️⃣ Demostrando funcionalidades avanzadas...")
    
    # Dayparting
    print("🕐 DAYPARTING OPTIMIZATION:")
    example_performance = [
        CampaignPerformance(
            campaign_id="demo",
            date=datetime.now() - timedelta(days=i),
            impressions=500 + i * 50,
            clicks=25 + i * 2,
            conversions=2 + i // 5,
            cost=50.0 + i * 2,
            revenue=100.0 + i * 4,
            ctr=0.05,
            cvr=0.04,
            cpa=25.0,
            roas=2.0,
            hour=i % 24,
            day_of_week=i % 7
        ) for i in range(72)  # 3 días de datos horarios
    ]
    
    dayparting_analysis = midas.dayparting_optimizer.analyze_hourly_performance(example_performance)
    optimal_schedule = midas.dayparting_optimizer.generate_optimal_schedule(target_budget=500.0)
    
    print(f"  • Horas pico identificadas: {dayparting_analysis['peak_hours'][:5]}")
    print(f"  • Horas pobres identificadas: {dayparting_analysis['poor_hours'][:5]}")
    if 'optimal_schedule' in optimal_schedule:
        peak_hour = max(optimal_schedule['optimal_schedule'], key=optimal_schedule['optimal_schedule'].get)
        print(f"  • Hora pico de budget: {peak_hour}:00")
    
    # A/B Testing
    print("🧪 A/B TESTING DE CREATIVOS:")
    creative_variants = [
        CreativeVariant("creative_1", "demo", "Llamada a la acción A", 5000, 250, 25, 500.0, 0.05, 0.05),
        CreativeVariant("creative_2", "demo", "Llamada a la acción B", 4800, 216, 20, 480.0, 0.045, 0.042),
        CreativeVariant("creative_3", "demo", "Llamada a la acción C", 5200, 260, 22, 520.0, 0.05, 0.042)
    ]
    
    ab_test = midas.creative_optimizer.create_ab_test("demo", creative_variants)
    if ab_test['status'] == 'success':
        test_id = ab_test['test_id']
        print(f"  • Test creado: {test_id}")
        print(f"  • Variantes: {len(creative_variants)}")
        print(f"  • Criterio de ganador: {ab_test['test_config']['winning_criteria']}")
        print(f"  • Nivel de confianza: {ab_test['test_config']['confidence_level']:.0%}")
    
    # 8. REPORTE FINAL Y RECOMENDACIONES
    print("\n8️⃣ Generando reporte final...")
    reporte_final = {
        'timestamp': datetime.now(),
        'sistema_version': '1.0.0',
        'campanas_analizadas': len(campaign_data),
        'optimizaciones_aplicadas': len(resultados.get('optimizations', {})),
        'reglas_activas': len([r for r in midas.rule_engine.rules.values() if r.is_active]),
        'modelos_entrenados': len(midas.bid_optimizer.models) + len(midas.performance_predictor.models),
        'tests_ab_activos': len(midas.creative_optimizer.active_tests),
        'mejoras_recomendadas': [
            "Considerar aumentar frecuencia de optimización a cada 4 horas",
            "Implementar alertas por email para cambios significativos",
            "Analizar audiencias específicas para mejorar targeting",
            "Configurar integración completa con NOESIS para mejor forecasting"
        ]
    }
    
    print("📋 REPORTE FINAL:")
    for key, value in reporte_final.items():
        if key != 'timestamp':
            print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎉 === OPTIMIZACIÓN AUTOMÁTICA COMPLETADA ===")
    print(f"✅ Sistema MIDAS funcionando al 100%")
    print(f"📊 {len(campaign_data)} campañas optimizadas")
    print(f"🔄 {len(midas.rule_engine.rules)} reglas de optimización activas")
    print(f"🤖 Machine Learning: {len(midas.bid_optimizer.models)} modelos entrenados")
    print(f"⏰ Último reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return midas, resultados

def ejemplo_uso_individual():
    """
    Ejemplo de uso de componentes individuales
    """
    print("\n" + "="*60)
    print("🔧 === USO INDIVIDUAL DE COMPONENTES ===")
    print("="*60)
    
    # Solo Bid Optimization
    print("\n💰 BID OPTIMIZATION SOLO:")
    midas = MIDASAutoOptimization()
    
    # Datos históricos para entrenar modelo
    historical_data = [
        CampaignPerformance(
            campaign_id="solo_test",
            date=datetime.now() - timedelta(days=i),
            impressions=1000 + i * 25,
            clicks=50 + i,
            conversions=5 + i // 7,
            cost=100.0 + i * 2,
            revenue=200.0 + i * 5,
            ctr=0.05,
            cvr=0.05,
            cpa=20.0,
            roas=2.0,
            hour=10 + i % 10,
            day_of_week=i % 7
        ) for i in range(60)
    ]
    
    # Entrenar y optimizar
    training = midas.bid_optimizer.train_bid_model("solo_test", historical_data)
    print(f"  • Entrenamiento: {training['status']}")
    
    context = {
        'hour': 15, 'day_of_week': 3, 'impressions': 1500, 'clicks': 75,
        'conversions': 7, 'cost': 150.0, 'ctr': 0.05, 'cvr': 0.046, 'cpa': 21.4, 'roas': 2.1
    }
    
    bid_opt = midas.bid_optimizer.optimize_bids("solo_test", 2.50, context)
    print(f"  • Bid actual: ${bid_opt['current_bid']}")
    print(f"  • Bid optimizado: ${bid_opt['optimized_bid']}")
    print(f"  • Cambio: {bid_opt['change_percentage']:+.1f}%")
    print(f"  • Justificación: {bid_opt['justification']}")
    
    # Solo Performance Prediction
    print("\n🔮 PERFORMANCE PREDICTION SOLO:")
    prediction = midas.performance_predictor.predict_performance("solo_test", context, days_ahead=14)
    print(f"  • ROAS predicho (14 días): {prediction['average_roas']:.3f}")
    print(f"  • CTR predicho: {prediction['average_ctr']:.4f}")
    print(f"  • Confianza: {prediction['confidence']:.1%}")
    print(f"  • Basado en modelo: {prediction['model_based']}")
    
    if prediction.get('alerts'):
        print(f"  • Alertas generadas: {len(prediction['alerts'])}")
    
    print("\n✅ Componentes individuales funcionando correctamente")

if __name__ == "__main__":
    print("🚀 Iniciando ejemplos del Sistema MIDAS Auto Optimization")
    print(f"⏰ Tiempo de inicio: {datetime.now()}")
    
    try:
        # Ejecutar ejemplo completo
        midas_system, resultados = ejemplo_optimizacion_completa()
        
        # Ejecutar ejemplo individual
        ejemplo_uso_individual()
        
        print("\n" + "="*60)
        print("🎊 ¡TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE!")
        print("="*60)
        print("📖 Para usar en producción:")
        print("  1. Configurar credenciales NOESIS")
        print("  2. Conectar con base de datos de campañas")
        print("  3. Configurar reglas de negocio específicas")
        print("  4. Implementar monitoring y alertas")
        print("  5. Ejecutar optimización en schedule automático")
        
    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)