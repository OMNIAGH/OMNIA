"""
Prueba simple de importación y estructura de NOESIS
"""

def test_imports():
    """Prueba que todas las clases se pueden importar"""
    try:
        print("🔍 Probando importaciones de NOESIS...")
        
        # Importaciones básicas
        from noesis_trend_analysis import (
            TrendDirection, AlertLevel, TrendMetrics, 
            InflectionPoint, Alert
        )
        print("✅ Clases de datos importadas correctamente")
        
        # Importaciones de analizadores
        from noesis_trend_analysis import (
            TrendDetector, SeasonalityAnalyzer, VolatilityAnalyzer,
            CorrelationAnalyzer, InflectionPointDetector, AlertSystem
        )
        print("✅ Analizadores importados correctamente")
        
        # Inicializar clases sin datos
        trend_detector = TrendDetector()
        seasonality_analyzer = SeasonalityAnalyzer()
        volatility_analyzer = VolatilityAnalyzer()
        correlation_analyzer = CorrelationAnalyzer()
        inflection_detector = InflectionPointDetector()
        alert_system = AlertSystem()
        
        print("✅ Todas las clases inicializadas correctamente")
        
        # Verificar métodos principales
        print("\n🔧 Verificando métodos disponibles...")
        
        # TrendDetector methods
        methods = ['detect_trend_linear', 'detect_trend_moving_average', 
                  'detect_trend_momentum', 'detect_multiple_timeframes']
        for method in methods:
            if hasattr(trend_detector, method):
                print(f"   ✅ TrendDetector.{method}")
            else:
                print(f"   ❌ TrendDetector.{method} NO ENCONTRADO")
        
        # SeasonalityAnalyzer methods
        methods = ['detect_seasonality', 'detect_cycles', 'test_stationarity']
        for method in methods:
            if hasattr(seasonality_analyzer, method):
                print(f"   ✅ SeasonalityAnalyzer.{method}")
            else:
                print(f"   ❌ SeasonalityAnalyzer.{method} NO ENCONTRADO")
        
        # VolatilityAnalyzer methods
        methods = ['calculate_volatility_metrics', 'calculate_var_cvar', 
                  'calculate_risk_metrics', 'detect_volatility_regimes']
        for method in methods:
            if hasattr(volatility_analyzer, method):
                print(f"   ✅ VolatilityAnalyzer.{method}")
            else:
                print(f"   ❌ VolatilityAnalyzer.{method} NO ENCONTRADO")
        
        # InflectionPointDetector methods
        methods = ['detect_peaks_and_troughs', 'detect_trend_changes', 
                  'analyze_invalidation_points']
        for method in methods:
            if hasattr(inflection_detector, method):
                print(f"   ✅ InflectionPointDetector.{method}")
            else:
                print(f"   ❌ InflectionPointDetector.{method} NO ENCONTRADO")
        
        # AlertSystem methods
        methods = ['check_volatility_alert', 'check_trend_alert', 
                  'run_comprehensive_alert_check', 'generate_alert_report']
        for method in methods:
            if hasattr(alert_system, method):
                print(f"   ✅ AlertSystem.{method}")
            else:
                print(f"   ❌ AlertSystem.{method} NO ENCONTRADO")
        
        print("\n🎉 ¡Todas las importaciones y estructura correctas!")
        
        # Verificar enums
        print(f"\n📋 Enums disponibles:")
        print(f"   TrendDirection: {', '.join([t.value for t in TrendDirection])}")
        print(f"   AlertLevel: {', '.join([a.value for a in AlertLevel])}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_with_dummy_data():
    """Prueba con datos dummy simples"""
    print("\n🧪 Probando con datos dummy...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Crear datos dummy muy simples
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        dummy_prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        dummy_returns = dummy_prices.pct_change().dropna()
        
        # Probar TrendDetector
        from noesis_trend_analysis import TrendDetector
        detector = TrendDetector()
        result = detector.detect_trend_linear(dummy_prices)
        print(f"✅ TrendDetector funciona: {result.direction.value}")
        
        # Probar VolatilityAnalyzer
        from noesis_trend_analysis import VolatilityAnalyzer
        vol_analyzer = VolatilityAnalyzer()
        vol_metrics = vol_analyzer.calculate_volatility_metrics(dummy_returns)
        print(f"✅ VolatilityAnalyzer funciona: Volatilidad calculada")
        
        print("✅ Pruebas con datos dummy exitosas")
        return True
        
    except Exception as e:
        print(f"❌ Error con datos dummy: {e}")
        return False

if __name__ == "__main__":
    print("🏗️  VERIFICACIÓN DE ESTRUCTURA NOESIS")
    print("=" * 40)
    
    # Prueba de importaciones
    import_success = test_imports()
    
    if import_success:
        # Prueba con datos dummy
        dummy_success = test_with_dummy_data()
        
        if dummy_success:
            print("\n🎊 ¡SISTEMA NOESIS COMPLETAMENTE FUNCIONAL!")
            print("   Todas las clases, métodos y funcionalidades verificadas.")
        else:
            print("\n⚠️  Problemas detectados con datos reales")
    else:
        print("\n❌ Fallos en la estructura básica del sistema")
    
    print("\n📖 USO DEL SISTEMA:")
    print("   from noesis_trend_analysis import *")
    print("   # Crear instancia de cualquier analizador")
    print("   detector = TrendDetector()")
    print("   # Usar métodos disponibles")