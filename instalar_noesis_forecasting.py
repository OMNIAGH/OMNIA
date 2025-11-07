#!/usr/bin/env python3
"""
NOESIS Forecasting Models - Script de Instalación y Configuración
================================================================

Script automatizado para instalar y configurar el sistema de forecasting
desarrollado para NOESIS.

Autor: NOESIS
Versión: 1.0
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def print_header():
    """Imprimir encabezado del script"""
    print("=" * 60)
    print("🚀 NOESIS FORECASTING MODELS - INSTALADOR")
    print("=" * 60)
    print("Sistema completo de modelos de forecasting predictivo")
    print("Autor: NOESIS")
    print("Versión: 1.0")
    print("=" * 60)
    print()

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    
    if sys.version_info < (3, 7):
        print("❌ Error: Se requiere Python 3.7 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Versión OK")
    return True

def install_dependencies():
    """Instalar dependencias del sistema"""
    print("\n📦 Instalando dependencias...")
    
    # Lista de dependencias esenciales para forecasting
    dependencies = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.12.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.2.0",
        "joblib>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "python-dateutil>=2.8.0",
        "pytz>=2021.1"
    ]
    
    failed_deps = []
    
    for dep in dependencies:
        try:
            print(f"   Instalando {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep, "--quiet"
            ])
            print(f"   ✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Error instalando {dep}: {e}")
            failed_deps.append(dep)
    
    if failed_deps:
        print(f"\n⚠️  Dependencias fallidas: {len(failed_deps)}")
        print("   Algunas funciones pueden no estar disponibles")
    else:
        print(f"\n✅ Todas las dependencias básicas instaladas correctamente")
    
    return len(failed_deps) == 0

def install_prophet():
    """Instalar Prophet (opcional)"""
    print("\n🔮 Instalando Prophet (opcional)...")
    
    try:
        # Intentar instalar Prophet
        print("   Instalando Prophet...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "prophet>=1.1.0", "--quiet"
        ])
        print("   ✅ Prophet instalado correctamente")
        return True
    except subprocess.CalledProcessError:
        print("   ⚠️  Error instalando Prophet")
        print("   Para instalar manualmente: pip install prophet")
        print("   Prophet es opcional, el sistema funcionará sin él")
        return False

def test_imports():
    """Probar importación de módulos principales"""
    print("\n🧪 Probando importaciones...")
    
    modules_to_test = {
        'numpy': 'np',
        'pandas': 'pd', 
        'scipy': 'scipy',
        'sklearn': 'sklearn',
        'statsmodels': 'sm',
        'xgboost': 'xgb',
        'lightgbm': 'lgb',
        'matplotlib': 'plt',
        'joblib': 'joblib'
    }
    
    successful_imports = 0
    failed_imports = []
    
    for module, alias in modules_to_test.items():
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
            successful_imports += 1
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    # Probar Prophet por separado
    try:
        from prophet import Prophet
        print("   ✅ prophet")
        successful_imports += 1
    except ImportError:
        print("   ⚠️  prophet (opcional)")
    
    print(f"\n📊 Importaciones exitosas: {successful_imports}/{len(modules_to_test)+1}")
    
    if failed_imports:
        print(f"⚠️  Módulos faltantes: {', '.join(failed_imports)}")
        return False
    
    return True

def create_example_environment():
    """Crear directorio de ejemplos y datos"""
    print("\n📁 Creando estructura de directorios...")
    
    directories = [
        "ejemplos",
        "datos", 
        "modelos_guardados",
        "resultados"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Creado: {directory}/")
        else:
            print(f"   📁 Ya existe: {directory}/")
    
    return True

def create_quick_start_script():
    """Crear script de inicio rápido"""
    print("\n📝 Creando script de inicio rápido...")
    
    script_content = '''#!/usr/bin/env python3
"""
NOESIS Forecasting - Inicio Rápido
=================================

Script de ejemplo para empezar rápidamente con el sistema.
"""

from noesis_forecasting_models import (
    NoesisForecastingAPI, 
    ForecastingConfig, 
    create_sample_data
)

def main():
    print("🚀 NOESIS Forecasting - Inicio Rápido")
    print("=" * 40)
    
    # 1. Configuración básica
    config = ForecastingConfig(
        test_size=0.2,
        ensemble_method='weighted'
    )
    
    # 2. Crear API
    api = NoesisForecastingAPI(config)
    
    # 3. Generar datos de ejemplo
    print("📊 Generando datos de ejemplo...")
    data = create_sample_data(
        start_date='2020-01-01',
        periods=365,
        frequency='D',
        trend=0.1,
        seasonality_amplitude=10
    )
    
    # 4. Analizar serie
    print("🔍 Analizando serie temporal...")
    analysis = api.analyze_series(data)
    print(f"   Estacionalidad detectada: {analysis['seasonality']['has_seasonality']}")
    
    # 5. Entrenar modelos
    print("🤖 Entrenando modelos...")
    results = api.train_all_models(data)
    
    # 6. Mostrar resultados
    print("📈 Resultados:")
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"   {model_name}: R² = {metrics['r2']:.3f}")
    
    # 7. Predicción
    print("🔮 Generando predicciones...")
    pred = api.predict_ensemble(steps=12)
    print(f"   Predicción próxima semana: {pred['predictions'].iloc[0]:.2f}")
    print(f"   Confianza: {pred['confidence']:.2f}")
    
    # 8. Guardar modelos
    print("💾 Guardando modelos...")
    api.save_models("./modelos_noesis_demo")
    print("   Modelos guardados en ./modelos_noesis_demo/")
    
    print("\\n🎉 ¡Demostración completada!")
    print("   Ejecutar: python ejemplos_noesis_forecasting.py")
    print("   para ver más ejemplos.")

if __name__ == "__main__":
    main()
'''
    
    with open("inicio_rapido_noesis.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("   ✅ Creado: inicio_rapido_noesis.py")
    return True

def test_system():
    """Probar el sistema completo"""
    print("\n🧪 Probando sistema completo...")
    
    try:
        # Importar sistema principal
        from noesis_forecasting_models import NoesisForecastingAPI, create_sample_data
        
        # Crear API y datos de prueba
        api = NoesisForecastingAPI()
        data = create_sample_data(periods=100)  # Datos pequeños para prueba rápida
        
        # Entrenar al menos un modelo básico
        from noesis_forecasting_models import ARIMAModel
        model = ARIMAModel(api.config)
        model.fit(data)
        pred = model.predict(5)
        
        print("   ✅ Sistema básico funcionando")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en prueba del sistema: {e}")
        return False

def print_final_instructions():
    """Imprimir instrucciones finales"""
    print("\n" + "=" * 60)
    print("🎉 ¡INSTALACIÓN COMPLETADA!")
    print("=" * 60)
    
    print("\n📚 PRÓXIMOS PASOS:")
    print("1. Probar el sistema:")
    print("   python inicio_rapido_noesis.py")
    print()
    print("2. Ver ejemplos completos:")
    print("   python ejemplos_noesis_forecasting.py")
    print()
    print("3. Importar en tu código:")
    print("   from noesis_forecasting_models import NoesisForecastingAPI")
    print()
    
    print("📖 DOCUMENTACIÓN:")
    print("- README_NOESIS_Forcasting.md - Documentación completa")
    print("- ejemplos_noesis_forecasting.py - Ejemplos de uso")
    print("- noesis_forecasting_models.py - API completa")
    print()
    
    print("🛠️  ARCHIVOS PRINCIPALES:")
    print("- noesis_forecasting_models.py    : Sistema principal")
    print("- ejemplos_noesis_forecasting.py  : Ejemplos de uso")
    print("- inicio_rapido_noesis.py        : Demo rápida")
    print("- requirements.txt               : Dependencias")
    print()
    
    print("💡 CARACTERÍSTICAS DISPONIBLES:")
    print("✓ Modelos ARIMA, SARIMA, Prophet")
    print("✓ ML: XGBoost, LightGBM, Random Forest")
    print("✓ Ensemble methods con pesos optimizados")
    print("✓ Validación walk-forward y cross-validation")
    print("✓ API para predicciones en tiempo real")
    print("✓ Manejo automático de outliers y missing values")
    print("✓ Detección automática de estacionalidad")
    print()
    
    print("🎯 ¡Listo para producción!")
    print("=" * 60)

def main():
    """Función principal del instalador"""
    print_header()
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependencias
    basic_success = install_dependencies()
    
    # Instalar Prophet (opcional)
    prophet_success = install_prophet()
    
    # Probar importaciones
    import_success = test_imports()
    
    # Crear estructura de directorios
    dirs_created = create_example_environment()
    
    # Crear script de inicio rápido
    script_created = create_quick_start_script()
    
    # Probar sistema (básico)
    if basic_success:
        system_working = test_system()
    else:
        system_working = False
    
    # Instrucciones finales
    print_final_instructions()
    
    # Código de salida
    if basic_success and import_success and dirs_created and script_created:
        print("\n✅ Instalación exitosa")
        sys.exit(0)
    else:
        print("\n⚠️  Instalación con advertencias")
        print("   Revisar los mensajes anteriores para detalles")
        sys.exit(1)

if __name__ == "__main__":
    main()
