"""
Ejemplo de uso del sistema MIDAS Google OPAL
Demonstración de funcionalidades principales
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from midas_google_opal import (
    initialize_midas_google_opal,
    run_automatic_optimization,
    GoogleAdsManager,
    CampaignOptimizer,
    QualityScoreMonitor,
    RealTimeDashboard
)

async def ejemplo_basico():
    """Ejemplo básico de uso del sistema"""
    print("=" * 60)
    print("    MIDAS Google OPAL - Ejemplo Básico")
    print("=" * 60)
    
    # Configuración del sistema
    config = {
        'customer_id': '1234567890',  # Su Customer ID sin guiones
        'developer_token': 'su_developer_token',
        'client_id': 'su_client_id',
        'client_secret': 'su_client_secret',
        'refresh_token': 'su_refresh_token',
        'ga4_property_id': 'propiedad_ga4'  # Opcional
    }
    
    try:
        # 1. Inicializar sistema
        print("\n1. Inicializando sistema...")
        system = await initialize_midas_google_opal(config)
        
        if system['status'] != 'initialized':
            print(f"Error: {system.get('error')}")
            return
        
        print("✓ Sistema inicializado correctamente")
        
        # 2. Obtener campañas activas
        print("\n2. Obteniendo campañas activas...")
        campaigns = await system['google_ads_manager'].get_active_campaigns()
        print(f"✓ {len(campaigns)} campañas encontradas")
        
        for campaign in campaigns:
            print(f"   - {campaign['name']} (ID: {campaign['id']})")
        
        # 3. Obtener métricas de la primera campaña
        if campaigns:
            first_campaign = campaigns[0]
            print(f"\n3. Obteniendo métricas de '{first_campaign['name']}'...")
            
            metrics = await system['google_ads_manager'].get_campaign_metrics(first_campaign['id'])
            if metrics:
                print(f"✓ Métricas obtenidas:")
                print(f"   - Impresiones: {metrics.impressions:,}")
                print(f"   - Clics: {metrics.clicks:,}")
                print(f"   - CTR: {metrics.ctr_percentage:.2f}%")
                print(f"   - CPC: €{metrics.cpc/1_000_000:.2f}")
                print(f"   - Conversiones: {metrics.conversions}")
                print(f"   - ROAS: {metrics.roas:.2f}")
            else:
                print("No se pudieron obtener métricas")
        
        # 4. Generar dashboard
        print("\n4. Generando dashboard de performance...")
        dashboard_data = await system['dashboard'].get_dashboard_data()
        print("✓ Dashboard generado")
        
        if 'overview' in dashboard_data:
            overview = dashboard_data['overview']
            print(f"   - Total campañas: {overview['total_campaigns']}")
            print(f"   - Impresiones totales: {overview['total_impressions']:,}")
            print(f"   - Clics totales: {overview['total_clicks']:,}")
            print(f"   - Costo total: €{overview['total_cost']:.2f}")
            print(f"   - CTR general: {overview['ctr']:.2f}%")
        
    except Exception as e:
        print(f"Error en ejemplo básico: {e}")

async def ejemplo_optimizacion():
    """Ejemplo de optimización de campañas"""
    print("\n" + "=" * 60)
    print("    MIDAS Google OPAL - Ejemplo de Optimización")
    print("=" * 60)
    
    config = {
        'customer_id': '1234567890',
        'developer_token': 'su_developer_token',
        'client_id': 'su_client_id', 
        'client_secret': 'su_client_secret',
        'refresh_token': 'su_refresh_token'
    }
    
    try:
        # Inicializar sistema
        system = await initialize_midas_google_opal(config)
        
        if system['status'] != 'initialized':
            print(f"Error: {system.get('error')}")
            return
        
        # Ejecutar optimización automática
        print("\nEjecutando optimización automática...")
        await system['campaign_optimizer'].optimize_campaigns()
        print("✓ Optimización completada")
        
        # Obtener ad groups de una campaña
        campaigns = await system['google_ads_manager'].get_active_campaigns()
        if campaigns:
            campaign = campaigns[0]
            print(f"\nObteniendo ad groups de '{campaign['name']}'...")
            
            ad_groups = await system['google_ads_manager'].get_ad_groups(campaign['id'])
            print(f"✓ {len(ad_groups)} ad groups encontrados")
            
            for ad_group in ad_groups:
                print(f"   - {ad_group['name']} (ID: {ad_group['id']})")
                
                # Obtener keywords del ad group
                keywords = await system['google_ads_manager'].get_keywords(ad_group['id'])
                print(f"     Keywords: {len(keywords)} encontradas")
                
                for keyword in keywords[:3]:  # Mostrar solo las primeras 3
                    print(f"     - {keyword.text} (QS: {keyword.quality_score})")
                
                if len(keywords) > 3:
                    print(f"     ... y {len(keywords) - 3} más")
        
        # Sincronizar con Google Analytics
        print("\nSincronizando con Google Analytics...")
        sync_result = await system['google_ads_manager'].sync_with_google_analytics(
            (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        print("✓ Sincronización completada")
        print(f"   Score de correlación: {sync_result.get('data_correlation_score', 0):.2f}")
        
    except Exception as e:
        print(f"Error en ejemplo de optimización: {e}")

async def ejemplo_monitoreo():
    """Ejemplo de monitoreo de Quality Score"""
    print("\n" + "=" * 60)
    print("    MIDAS Google OPAL - Ejemplo de Monitoreo QS")
    print("=" * 60)
    
    config = {
        'customer_id': '1234567890',
        'developer_token': 'su_developer_token',
        'client_id': 'su_client_id',
        'client_secret': 'su_client_secret', 
        'refresh_token': 'su_refresh_token'
    }
    
    try:
        system = await initialize_midas_google_opal(config)
        
        if system['status'] != 'initialized':
            print(f"Error: {system.get('error')}")
            return
        
        # Obtener reporte de Quality Score
        print("\nGenerando reporte de Quality Score...")
        qs_report = system['quality_monitor'].get_quality_score_report()
        print("✓ Reporte generado")
        
        if 'statistics' in qs_report:
            print("\nEstadísticas de Quality Score:")
            for rango, stats in qs_report['statistics'].items():
                print(f"   - {rango}: {stats['count']} keywords (promedio: {stats['avg_score']})")
        
        print(f"\nTotal keywords monitoreadas: {qs_report.get('total_keywords', 0)}")
        print(f"Alertas activas: {qs_report.get('alerts_count', 0)}")
        
        # Obtener recomendaciones para una keyword específica (simulado)
        print("\nEjemplo de recomendaciones:")
        fake_keyword_id = "123456789"
        recommendations = system['quality_monitor'].get_quality_score_recommendations(fake_keyword_id)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
    except Exception as e:
        print(f"Error en ejemplo de monitoreo: {e}")

async def ejemplo_creacion_campana():
    """Ejemplo de creación de nueva campaña"""
    print("\n" + "=" * 60)
    print("    MIDAS Google OPAL - Ejemplo Creación Campaña")
    print("=" * 60)
    
    config = {
        'customer_id': '1234567890',
        'developer_token': 'su_developer_token',
        'client_id': 'su_client_id',
        'client_secret': 'su_client_secret',
        'refresh_token': 'su_refresh_token'
    }
    
    try:
        system = await initialize_midas_google_opal(config)
        
        if system['status'] != 'initialized':
            print(f"Error: {system.get('error')}")
            return
        
        # Datos de la nueva campaña
        campaign_data = {
            'name': f'Campaña Automática {datetime.now().strftime("%Y%m%d_%H%M")}',
            'status': 'PAUSED',  # Iniciar pausada para revisión
            'advertising_channel_type': 'SEARCH',
            'budget': '1000000',  # 1000 EUR (en micros)
            'bidding_strategy_type': 'MANUAL_CPC'
        }
        
        print(f"\nCreando nueva campaña: '{campaign_data['name']}'...")
        campaign_id = await system['google_ads_manager'].create_campaign(campaign_data)
        
        if campaign_id:
            print(f"✓ Campaña creada exitosamente (ID: {campaign_id})")
            
            # Ejemplo de actualización de estado
            print("\nActivando campaña...")
            await system['google_ads_manager'].update_campaign_status(campaign_id, 'ENABLED')
            print("✓ Campaña activada")
            
        else:
            print("Error creando campaña")
        
    except Exception as e:
        print(f"Error en ejemplo de creación: {e}")

def mostrar_estructura_archivos():
    """Muestra la estructura de archivos del proyecto"""
    print("\n" + "=" * 60)
    print("    MIDAS Google OPAL - Estructura del Proyecto")
    print("=" * 60)
    
    archivos = [
        ("midas_google_opal.py", "Sistema principal con todas las clases"),
        ("optimization_rules.json", "Reglas de optimización configurables"),
        ("ad_templates.json", "Templates de anuncios para auto-creación"),
        ("google_ads_config.json.example", "Ejemplo de configuración de Google Ads"),
        ("requirements.txt", "Dependencias del proyecto"),
        ("run_midas_opal.py", "Script de ejecución principal"),
        ("setup_credentials.py", "Asistente de configuración"),
        ("install.sh", "Script de instalación automática"),
        ("ejemplo_uso.py", "Este archivo con ejemplos")
    ]
    
    print("\nArchivos del proyecto:")
    for archivo, descripcion in archivos:
        print(f"   {archivo:<30} - {descripcion}")
    
    print("\n" + "="*60)
    print("    Funcionalidades Principales")
    print("="*60)
    
    funcionalidades = [
        "✓ Integración completa Google Ads API v14",
        "✓ Gestión de campañas (crear, modificar, pausar)",
        "✓ Optimización automática de pujas con ML",
        "✓ Sistema de negative keywords inteligente",
        "✓ Creación automática de anuncios desde templates",
        "✓ Monitoreo de Quality Score en tiempo real",
        "✓ Integración con Google Analytics 4",
        "✓ Dashboard de performance en tiempo real",
        "✓ Sistema de alertas y recomendaciones",
        "✓ Base de datos para historial y análisis"
    ]
    
    for func in funcionalidades:
        print(f"   {func}")
    
    print("\n" + "="*60)
    print("    Para empezar:")
    print("="*60)
    print("   1. python3 setup_credentials.py")
    print("   2. python3 run_midas_opal.py")
    print("   3. Seguir las opciones del menú interactivo")
    print("="*60)

async def main():
    """Función principal que ejecuta todos los ejemplos"""
    
    # Mostrar estructura del proyecto
    mostrar_estructura_archivos()
    
    print("\n¿Desea ejecutar los ejemplos? (esto requiere credenciales válidas)")
    print("Los ejemplos simularán las operaciones principales del sistema.")
    print()
    
    try:
        # Ejecutar ejemplos solo si las credenciales están configuradas
        config_exists = os.path.exists('google_ads_config.json')
        
        if config_exists:
            print("✓ Archivo de configuración encontrado, ejecutando ejemplos...")
            
            # Ejecutar ejemplos (comentados para evitar errores con credenciales)
            # await ejemplo_basico()
            # await ejemplo_optimizacion()
            # await ejemplo_monitoreo()
            # await ejemplo_creacion_campana()
            
            print("\n⚠️  Ejemplos comentados - Configure credenciales válidas para ejecutarlos")
            
        else:
            print("⚠️  No se encontró archivo de configuración")
            print("   Ejecute: python3 setup_credentials.py")
            
    except Exception as e:
        print(f"\nError ejecutando ejemplos: {e}")
    
    print("\n" + "="*60)
    print("    ¡Gracias por usar MIDAS Google OPAL!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())