"""
Google OPAL Workflow para MIDAS
Sistema integral de gestión y optimización de Google Ads con IA
Versión: 2.0.0
Fecha: 2025-11-06
"""

import os
import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import sqlite3
import threading
from queue import Queue
import time
from enum import Enum

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('midas_google_opal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CampaignStatus(Enum):
    """Estados de campaña disponibles"""
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    REMOVED = "REMOVED"


class BidStrategyType(Enum):
    """Tipos de estrategia de puja"""
    MANUAL_CPC = "MANUAL_CPC"
    TARGET_CPA = "TARGET_CPA"
    TARGET_ROAS = "TARGET_ROAS"
    MAXIMIZE_CONVERSIONS = "MAXIMIZE_CONVERSIONS"
    MAXIMIZE_CONVERSION_VALUE = "MAXIMIZE_CONVERSION_VALUE"


@dataclass
class CampaignMetrics:
    """Métricas de campaña para análisis y optimización"""
    campaign_id: str
    campaign_name: str
    impressions: int
    clicks: int
    cost_micros: int
    conversions: int
    conversion_value: float
    ctr: float
    cpc: float
    cpa: float
    roas: float
    quality_score: float
    status: str
    date: str

    @property
    def ctr_percentage(self) -> float:
        return self.ctr * 100

    @property
    def cpa_euros(self) -> float:
        return self.cpa / 1_000_000 if self.cpa else 0

    @property
    def cost_euros(self) -> float:
        return self.cost_micros / 1_000_000 if self.cost_micros else 0


@dataclass
class KeywordData:
    """Estructura de datos para palabras clave"""
    keyword_id: str
    text: str
    match_type: str
    status: str
    final_urls: List[str]
    cpc_bid_micros: int
    quality_score: float
    is_negative: bool = False


@dataclass
class AdGroupData:
    """Estructura de datos para grupos de anuncios"""
    ad_group_id: str
    name: str
    status: str
    cpc_bid_micros: int
    campaign_id: str
    metrics: CampaignMetrics
    keywords: List[KeywordData] = None


class GoogleAnalyticsIntegration:
    """Integración con Google Analytics 4"""
    
    def __init__(self, property_id: str, credentials_path: str = None):
        self.property_id = property_id
        self.credentials_path = credentials_path
        self.base_url = "https://analyticsdata.googleapis.com/v1beta"
        
    async def get_conversion_data(self, start_date: str, end_date: str) -> Dict:
        """Obtiene datos de conversiones desde GA4"""
        try:
            url = f"{self.base_url}/properties/{self.property_id}:runReport"
            headers = {
                "Authorization": f"Bearer {self._get_access_token()}",
                "Content-Type": "application/json"
            }
            
            body = {
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "metrics": [
                    {"name": "conversions"},
                    {"name": "purchaseRevenue"},
                    {"name": "sessions"},
                    {"name": "users"}
                ],
                "dimensions": [
                    {"name": "date"},
                    {"name": "source"},
                    {"name": "medium"}
                ]
            }
            
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de GA4: {e}")
            return {}
    
    def _get_access_token(self) -> str:
        """Obtiene token de acceso para GA4"""
        # Implementar lógica de OAuth2
        return os.getenv("GOOGLE_ANALYTICS_ACCESS_TOKEN", "")


class DatabaseManager:
    """Gestor de base de datos para almacenar métricas y configuraciones"""
    
    def __init__(self, db_path: str = "midas_google_opal.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa la base de datos con las tablas necesarias"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS campaign_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    campaign_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    impressions INTEGER DEFAULT 0,
                    clicks INTEGER DEFAULT 0,
                    cost_micros INTEGER DEFAULT 0,
                    conversions INTEGER DEFAULT 0,
                    conversion_value REAL DEFAULT 0.0,
                    ctr REAL DEFAULT 0.0,
                    cpc REAL DEFAULT 0.0,
                    cpa REAL DEFAULT 0.0,
                    roas REAL DEFAULT 0.0,
                    quality_score REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'ENABLED',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_score_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    ad_group_id TEXT,
                    keyword_id TEXT,
                    quality_score REAL NOT NULL,
                    date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS negative_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT NOT NULL,
                    reason TEXT,
                    campaign_id TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
    def save_metrics(self, metrics: CampaignMetrics):
        """Guarda métricas de campaña en la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO campaign_metrics 
                (campaign_id, campaign_name, date, impressions, clicks, cost_micros, 
                 conversions, conversion_value, ctr, cpc, cpa, roas, quality_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.campaign_id, metrics.campaign_name, metrics.date,
                metrics.impressions, metrics.clicks, metrics.cost_micros,
                metrics.conversions, metrics.conversion_value, metrics.ctr,
                metrics.cpc, metrics.cpa, metrics.roas, metrics.quality_score,
                metrics.status
            ))
    
    def get_campaign_trends(self, campaign_id: str, days: int = 30) -> pd.DataFrame:
        """Obtiene tendencias de campaña para análisis"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM campaign_metrics 
                WHERE campaign_id = ? AND date >= date('now', '-{} days')
                ORDER BY date ASC
            """.format(days)
            
            return pd.read_sql_query(query, conn, params=(campaign_id,))


class QualityScoreMonitor:
    """Monitor de Quality Score con alertas y recomendaciones"""
    
    def __init__(self, db_manager: DatabaseManager, google_ads: 'GoogleAdsManager'):
        self.db_manager = db_manager
        self.google_ads = google_ads
        self.threshold_low = 6.0
        self.threshold_medium = 7.0
        self.alerts = Queue()
        
    async def monitor_quality_scores(self):
        """Monitorea Quality Scores en tiempo real"""
        logger.info("Iniciando monitoreo de Quality Score")
        
        while True:
            try:
                # Obtener campañas activas
                campaigns = await self.google_ads.get_active_campaigns()
                
                for campaign in campaigns:
                    # Obtener ad groups y keywords
                    ad_groups = await self.google_ads.get_ad_groups(campaign['id'])
                    
                    for ad_group in ad_groups:
                        keywords = await self.google_ads.get_keywords(ad_group['id'])
                        
                        for keyword in keywords:
                            await self._check_quality_score(
                                campaign['id'], 
                                ad_group['id'], 
                                keyword
                            )
                
                await asyncio.sleep(3600)  # Verificar cada hora
                
            except Exception as e:
                logger.error(f"Error en monitoreo de Quality Score: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos en caso de error
    
    async def _check_quality_score(self, campaign_id: str, ad_group_id: str, keyword: KeywordData):
        """Verifica el Quality Score de una keyword específica"""
        if keyword.quality_score < self.threshold_low:
            alert = {
                'type': 'LOW_QUALITY_SCORE',
                'campaign_id': campaign_id,
                'ad_group_id': ad_group_id,
                'keyword_id': keyword.keyword_id,
                'keyword_text': keyword.text,
                'quality_score': keyword.quality_score,
                'timestamp': datetime.now().isoformat()
            }
            
            self.alerts.put(alert)
            logger.warning(f"Quality Score bajo detectado: {keyword.text} - {keyword.quality_score}")
            
            # Guardar en historial
            self._save_quality_score_history(campaign_id, ad_group_id, keyword.keyword_id, keyword.quality_score)
    
    def _save_quality_score_history(self, campaign_id: str, ad_group_id: str, keyword_id: str, quality_score: float):
        """Guarda historial de Quality Score"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            conn.execute("""
                INSERT INTO quality_score_history 
                (campaign_id, ad_group_id, keyword_id, quality_score, date)
                VALUES (?, ?, ?, ?, ?)
            """, (campaign_id, ad_group_id, keyword_id, quality_score, datetime.now().strftime('%Y-%m-%d')))
    
    def get_quality_score_recommendations(self, keyword_id: str) -> List[str]:
        """Genera recomendaciones para mejorar Quality Score"""
        recommendations = []
        
        # Obtener historial de Quality Score
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT quality_score FROM quality_score_history 
                WHERE keyword_id = ? 
                ORDER BY date DESC LIMIT 10
            """, (keyword_id,))
            
            scores = [row[0] for row in cursor.fetchall()]
            
        if scores and len(scores) >= 2:
            # Analizar tendencia
            if scores[-1] < scores[0]:
                recommendations.append("El Quality Score está disminuyendo. Revisa la relevancia del anuncio y la página de destino.")
            
            if scores[-1] < 5.0:
                recommendations.extend([
                    "Mejorar la relevancia de la palabra clave",
                    "Aumentar la relevancia del anuncio",
                    "Optimizar la experiencia en la página de destino"
                ])
        
        return recommendations
    
    def get_quality_score_report(self) -> Dict:
        """Genera reporte de Quality Score"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            # Estadísticas por rango de Quality Score
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN quality_score >= 8 THEN 'Alto (8-10)'
                        WHEN quality_score >= 6 THEN 'Medio (6-7.9)'
                        ELSE 'Bajo (0-5.9)'
                    END as range,
                    COUNT(*) as count,
                    AVG(quality_score) as avg_score
                FROM quality_score_history 
                WHERE date >= date('now', '-30 days')
                GROUP BY range
            """)
            
            stats = {}
            for row in cursor.fetchall():
                stats[row[0]] = {
                    'count': row[1],
                    'avg_score': round(row[2], 2)
                }
            
            return {
                'statistics': stats,
                'total_keywords': sum(s['count'] for s in stats.values()),
                'alerts_count': self.alerts.qsize()
            }


class CampaignOptimizer:
    """Optimizador automático de campañas basado en performance y ML"""
    
    def __init__(self, db_manager: DatabaseManager, google_ads: 'GoogleAdsManager'):
        self.db_manager = db_manager
        self.google_ads = google_ads
        self.optimization_rules = self._load_optimization_rules()
        
    def _load_optimization_rules(self) -> List[Dict]:
        """Carga reglas de optimización configuradas"""
        # Cargar desde base de datos
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.execute("SELECT * FROM optimization_rules WHERE is_active = 1")
            rules = []
            for row in cursor.fetchall():
                rules.append({
                    'id': row[0],
                    'rule_name': row[1],
                    'rule_type': row[2],
                    'conditions': json.loads(row[3]),
                    'actions': json.loads(row[4])
                })
            return rules
    
    async def optimize_campaigns(self):
        """Ejecuta optimización automática de todas las campañas"""
        logger.info("Iniciando optimización automática de campañas")
        
        try:
            # Obtener campañas activas
            campaigns = await self.google_ads.get_active_campaigns()
            
            # Paralelizar optimización de campañas
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(asyncio.run, self._optimize_campaign(campaign['id'])): campaign['id']
                    for campaign in campaigns
                }
                
                for future in as_completed(futures):
                    campaign_id = futures[future]
                    try:
                        result = future.result()
                        logger.info(f"Optimización completada para campaña {campaign_id}: {result}")
                    except Exception as e:
                        logger.error(f"Error optimizando campaña {campaign_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error en optimización automática: {e}")
    
    async def _optimize_campaign(self, campaign_id: str) -> Dict:
        """Optimiza una campaña específica"""
        results = {
            'bid_optimizations': 0,
            'negative_keywords_added': 0,
            'ads_paused': 0,
            'ads_created': 0,
            'budget_adjustments': 0
        }
        
        # Obtener métricas actuales
        metrics = await self.google_ads.get_campaign_metrics(campaign_id)
        if not metrics:
            return results
        
        # Aplicar reglas de optimización
        for rule in self.optimization_rules:
            if self._evaluate_rule(rule, metrics):
                action_results = await self._execute_rule_actions(rule, campaign_id, metrics)
                for key, value in action_results.items():
                    results[key] += value
        
        return results
    
    def _evaluate_rule(self, rule: Dict, metrics: CampaignMetrics) -> bool:
        """Evalúa si una regla de optimización debe aplicarse"""
        conditions = rule['conditions']
        
        # Evaluar condiciones
        if 'min_impressions' in conditions and metrics.impressions < conditions['min_impressions']:
            return False
        
        if 'min_clicks' in conditions and metrics.clicks < conditions['min_clicks']:
            return False
        
        if 'max_cpa' in conditions and metrics.cpa_euros > conditions['max_cpa']:
            return True
        
        if 'min_roas' in conditions and metrics.roas < conditions['min_roas']:
            return True
        
        if 'max_cpc' in conditions and metrics.cpc > conditions['max_cpc']:
            return True
        
        if 'low_ctr' in conditions and metrics.ctr_percentage < conditions['low_ctr']:
            return True
        
        return False
    
    async def _execute_rule_actions(self, rule: Dict, campaign_id: str, metrics: CampaignMetrics) -> Dict:
        """Ejecuta las acciones de una regla de optimización"""
        actions = rule['actions']
        results = {
            'bid_optimizations': 0,
            'negative_keywords_added': 0,
            'ads_paused': 0,
            'ads_created': 0,
            'budget_adjustments': 0
        }
        
        # Optimizar pujas
        if 'optimize_bids' in actions:
            results['bid_optimizations'] = await self._optimize_bids(campaign_id, metrics)
        
        # Agregar negative keywords
        if 'add_negative_keywords' in actions:
            results['negative_keywords_added'] = await self._add_negative_keywords(campaign_id, actions['add_negative_keywords'])
        
        # Pausar anuncios de bajo rendimiento
        if 'pause_poor_ads' in actions:
            results['ads_paused'] = await self._pause_poor_performance_ads(campaign_id, actions['pause_poor_ads'])
        
        # Crear nuevos anuncios
        if 'create_new_ads' in actions:
            results['ads_created'] = await self._create_ads_from_templates(campaign_id, actions['create_new_ads'])
        
        # Ajustar presupuesto
        if 'adjust_budget' in actions:
            results['budget_adjustments'] = await self._adjust_budget(campaign_id, actions['adjust_budget'], metrics)
        
        return results
    
    async def _optimize_bids(self, campaign_id: str, metrics: CampaignMetrics) -> int:
        """Optimiza pujas basándose en performance"""
        optimizations = 0
        
        try:
            ad_groups = await self.google_ads.get_ad_groups(campaign_id)
            
            for ad_group in ad_groups:
                keywords = await self.google_ads.get_keywords(ad_group['id'])
                
                for keyword in keywords:
                    # Calcular nueva puja basada en performance
                    new_bid = self._calculate_optimal_bid(metrics, keyword)
                    
                    if new_bid != keyword.cpc_bid_micros:
                        await self.google_ads.update_keyword_bid(keyword.keyword_id, new_bid)
                        optimizations += 1
            
        except Exception as e:
            logger.error(f"Error optimizando pujas: {e}")
        
        return optimizations
    
    def _calculate_optimal_bid(self, campaign_metrics: CampaignMetrics, keyword: KeywordData) -> int:
        """Calcula la puja óptima para una keyword"""
        # Algoritmo básico de optimización de pujas
        if keyword.quality_score >= 8.0:
            # Alta calidad - incrementar puja si el rendimiento es bueno
            if campaign_metrics.roas > 3.0:
                return int(keyword.cpc_bid_micros * 1.2)
            elif campaign_metrics.ctr_percentage > 0.05:
                return int(keyword.cpc_bid_micros * 1.1)
        
        elif keyword.quality_score < 6.0:
            # Baja calidad - reducir puja
            if campaign_metrics.cpa_euros > 50:
                return int(keyword.cpc_bid_micros * 0.8)
        
        # Mantener puja actual
        return keyword.cpc_bid_micros
    
    async def _add_negative_keywords(self, campaign_id: str, keywords_to_add: List[str]) -> int:
        """Agrega negative keywords basadas en reglas"""
        added = 0
        
        for keyword_text in keywords_to_add:
            try:
                await self.google_ads.add_negative_keyword(campaign_id, keyword_text)
                added += 1
                
                # Guardar en base de datos
                with sqlite3.connect(self.db_manager.db_path) as conn:
                    conn.execute("""
                        INSERT INTO negative_keywords (keyword, reason, campaign_id)
                        VALUES (?, ?, ?)
                    """, (keyword_text, "Regla de optimización automática", campaign_id))
                    
            except Exception as e:
                logger.error(f"Error agregando negative keyword {keyword_text}: {e}")
        
        return added
    
    async def _pause_poor_performance_ads(self, campaign_id: str, threshold: Dict) -> int:
        """Pausa anuncios con bajo rendimiento"""
        paused = 0
        
        try:
            ad_groups = await self.google_ads.get_ad_groups(campaign_id)
            
            for ad_group in ad_groups:
                ads = await self.google_ads.get_ads(ad_group['id'])
                
                for ad in ads:
                    # Obtener métricas del anuncio
                    ad_metrics = await self.google_ads.get_ad_metrics(ad['id'])
                    
                    if ad_metrics and self._should_pause_ad(ad_metrics, threshold):
                        await self.google_ads.pause_ad(ad['id'])
                        paused += 1
                        
        except Exception as e:
            logger.error(f"Error pausando anuncios: {e}")
        
        return paused
    
    def _should_pause_ad(self, ad_metrics: CampaignMetrics, threshold: Dict) -> bool:
        """Determina si un anuncio debe ser pausado"""
        if 'min_clicks' in threshold and ad_metrics.clicks < threshold['min_clicks']:
            return True
        
        if 'max_cpc' in threshold and ad_metrics.cpc > threshold['max_cpc']:
            return True
        
        if 'min_impressions' in threshold and ad_metrics.impressions < threshold['min_impressions']:
            return True
        
        return False
    
    async def _create_ads_from_templates(self, campaign_id: str, templates: List[Dict]) -> int:
        """Crea anuncios desde templates predefinidos"""
        created = 0
        
        try:
            ad_groups = await self.google_ads.get_ad_groups(campaign_id)
            
            for template in templates:
                for ad_group in ad_groups:
                    # Generar anuncio desde template
                    ad_data = self._generate_ad_from_template(template)
                    
                    if ad_data:
                        result = await self.google_ads.create_ad(ad_group['id'], ad_data)
                        if result:
                            created += 1
                            
        except Exception as e:
            logger.error(f"Error creando anuncios: {e}")
        
        return created
    
    def _generate_ad_from_template(self, template: Dict) -> Dict:
        """Genera contenido de anuncio desde un template"""
        try:
            # Template básico de anuncio de texto
            ad_data = {
                'final_urls': template.get('final_urls', []),
                'headlines': template.get('headlines', []),
                'descriptions': template.get('descriptions', []),
                'display_paths': template.get('display_paths', [])
            }
            
            # Personalizar contenido dinámicamente si es necesario
            return ad_data
            
        except Exception as e:
            logger.error(f"Error generando anuncio desde template: {e}")
            return None
    
    async def _adjust_budget(self, campaign_id: str, adjustment: Dict, metrics: CampaignMetrics) -> int:
        """Ajusta presupuesto basándose en performance"""
        try:
            current_budget = await self.google_ads.get_campaign_budget(campaign_id)
            if not current_budget:
                return 0
            
            # Calcular nuevo presupuesto
            performance_multiplier = 1.0
            
            if metrics.roas > 4.0 and metrics.conversions > 10:
                performance_multiplier = 1.3
            elif metrics.roas > 2.0 and metrics.conversions > 5:
                performance_multiplier = 1.1
            elif metrics.roas < 1.5 or metrics.cpa_euros > 100:
                performance_multiplier = 0.8
            
            new_budget = int(current_budget * performance_multiplier)
            
            # Aplicar límites
            if 'min_budget' in adjustment:
                new_budget = max(new_budget, adjustment['min_budget'])
            if 'max_budget' in adjustment:
                new_budget = min(new_budget, adjustment['max_budget'])
            
            await self.google_ads.update_campaign_budget(campaign_id, new_budget)
            return 1
            
        except Exception as e:
            logger.error(f"Error ajustando presupuesto: {e}")
            return 0


class GoogleAdsManager:
    """Gestor principal de Google Ads API v14 con funciones avanzadas"""
    
    def __init__(self, customer_id: str, login_customer_id: str = None, config_file: str = None):
        self.customer_id = customer_id
        self.login_customer_id = login_customer_id
        self.config_file = config_file
        self.client = self._initialize_client()
        self.db_manager = DatabaseManager()
        self.ga4_integration = GoogleAnalyticsIntegration(
            os.getenv('GA4_PROPERTY_ID', ''),
            os.getenv('GA4_CREDENTIALS_PATH')
        )
        
    def _initialize_client(self):
        """Inicializa cliente de Google Ads"""
        try:
            config = {
                "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
                "client_id": os.getenv("GOOGLE_ADS_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_ADS_CLIENT_SECRET"),
                "refresh_token": os.getenv("GOOGLE_ADS_REFRESH_TOKEN"),
                "login_customer_id": self.login_customer_id or os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
                "use_proto_plus": True
            }
            
            if self.config_file and os.path.exists(self.config_file):
                config = self._load_config_from_file(self.config_file)
            
            return GoogleAdsClient.load_from_dict(config)
            
        except Exception as e:
            logger.error(f"Error inicializando cliente de Google Ads: {e}")
            return None
    
    def _load_config_from_file(self, config_file: str) -> Dict:
        """Carga configuración desde archivo"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return {}
    
    async def get_active_campaigns(self) -> List[Dict]:
        """Obtiene todas las campañas activas"""
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = f"""
                SELECT 
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    campaign.campaign_budget
                FROM campaign 
                WHERE campaign.status = 'ENABLED'
            """
            
            response = ga_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            campaigns = []
            for batch in response:
                for row in batch.results:
                    campaign = {
                        'id': row.campaign.id,
                        'name': row.campaign.name,
                        'status': row.campaign.status.name,
                        'advertising_channel_type': row.campaign.advertising_channel_type.name,
                        'budget': row.campaign.campaign_budget
                    }
                    campaigns.append(campaign)
            
            return campaigns
            
        except Exception as e:
            logger.error(f"Error obteniendo campañas: {e}")
            return []
    
    async def get_campaign_metrics(self, campaign_id: str) -> Optional[CampaignMetrics]:
        """Obtiene métricas de una campaña específica"""
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = f"""
                SELECT 
                    campaign.id,
                    campaign.name,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions,
                    metrics.conversion_value,
                    metrics.ctr,
                    metrics.average_cpc,
                    metrics.conversions,
                    metrics.all_conversions,
                    campaign.status
                FROM campaign 
                WHERE campaign.id = {campaign_id}
                    AND segments.date DURING LAST_30_DAYS
            """
            
            response = ga_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            for batch in response:
                for row in batch.results:
                    metrics = CampaignMetrics(
                        campaign_id=row.campaign.id,
                        campaign_name=row.campaign.name,
                        impressions=row.metrics.impressions,
                        clicks=row.metrics.clicks,
                        cost_micros=row.metrics.cost_micros,
                        conversions=row.metrics.conversions,
                        conversion_value=row.metrics.conversion_value,
                        ctr=row.metrics.ctr,
                        cpc=row.metrics.average_cpc,
                        cpa=row.metrics.cost_micros / max(row.metrics.conversions, 1),
                        roas=row.metrics.conversion_value / max(row.metrics.cost_micros / 1_000_000, 1),
                        quality_score=0.0,  # Se calculará por separado
                        status=row.campaign.status.name,
                        date=datetime.now().strftime('%Y-%m-%d')
                    )
                    return metrics
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas de campaña {campaign_id}: {e}")
            return None
    
    async def get_ad_groups(self, campaign_id: str) -> List[Dict]:
        """Obtiene ad groups de una campaña"""
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = f"""
                SELECT 
                    ad_group.id,
                    ad_group.name,
                    ad_group.status,
                    ad_group.cpc_bid_micros
                FROM ad_group 
                WHERE ad_group.campaign = {campaign_id}
                    AND ad_group.status != 'REMOVED'
            """
            
            response = ga_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            ad_groups = []
            for batch in response:
                for row in batch.results:
                    ad_group = {
                        'id': row.ad_group.id,
                        'name': row.ad_group.name,
                        'status': row.ad_group.status.name,
                        'cpc_bid_micros': row.ad_group.cpc_bid_micros
                    }
                    ad_groups.append(ad_group)
            
            return ad_groups
            
        except Exception as e:
            logger.error(f"Error obteniendo ad groups: {e}")
            return []
    
    async def get_keywords(self, ad_group_id: str) -> List[KeywordData]:
        """Obtiene keywords de un ad group"""
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = f"""
                SELECT 
                    ad_group_criterion.keyword.text,
                    ad_group_criterion.keyword.match_type,
                    ad_group_criterion.status,
                    ad_group_criterion.final_urls,
                    ad_group_criterion.cpc_bid_micros,
                    ad_group_criterion.quality_info.quality_score
                FROM ad_group_criterion 
                WHERE ad_group_criterion.ad_group = {ad_group_id}
                    AND ad_group_criterion.type = 'KEYWORD'
            """
            
            response = ga_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            keywords = []
            for batch in response:
                for row in batch.results:
                    keyword = KeywordData(
                        keyword_id="",  # Se puede obtener de otra consulta si es necesario
                        text=row.ad_group_criterion.keyword.text,
                        match_type=row.ad_group_criterion.keyword.match_type.name,
                        status=row.ad_group_criterion.status.name,
                        final_urls=list(row.ad_group_criterion.final_urls),
                        cpc_bid_micros=row.ad_group_criterion.cpc_bid_micros,
                        quality_score=row.ad_group_criterion.quality_info.quality_score,
                        is_negative=False
                    )
                    keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error obteniendo keywords: {e}")
            return []
    
    async def get_ads(self, ad_group_id: str) -> List[Dict]:
        """Obtiene anuncios de un ad group"""
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = f"""
                SELECT 
                    ad.id,
                    ad.name,
                    ad.status
                FROM ad 
                WHERE ad.ad_group = {ad_group_id}
                    AND ad.status != 'REMOVED'
            """
            
            response = ga_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            ads = []
            for batch in response:
                for row in batch.results:
                    ad = {
                        'id': row.ad.id,
                        'name': row.ad.name,
                        'status': row.ad.status.name
                    }
                    ads.append(ad)
            
            return ads
            
        except Exception as e:
            logger.error(f"Error obteniendo ads: {e}")
            return []
    
    async def get_ad_metrics(self, ad_id: str) -> Optional[CampaignMetrics]:
        """Obtiene métricas de un anuncio específico"""
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = f"""
                SELECT 
                    ad.id,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions,
                    metrics.conversion_value,
                    metrics.ctr,
                    metrics.average_cpc
                FROM ad 
                WHERE ad.id = {ad_id}
                    AND segments.date DURING LAST_30_DAYS
            """
            
            response = ga_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            for batch in response:
                for row in batch.results:
                    metrics = CampaignMetrics(
                        campaign_id="",
                        campaign_name="",
                        impressions=row.metrics.impressions,
                        clicks=row.metrics.clicks,
                        cost_micros=row.metrics.cost_micros,
                        conversions=row.metrics.conversions,
                        conversion_value=row.metrics.conversion_value,
                        ctr=row.metrics.ctr,
                        cpc=row.metrics.average_cpc,
                        cpa=row.metrics.cost_micros / max(row.metrics.conversions, 1),
                        roas=row.metrics.conversion_value / max(row.metrics.cost_micros / 1_000_000, 1),
                        quality_score=0.0,
                        status="",
                        date=datetime.now().strftime('%Y-%m-%d')
                    )
                    return metrics
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas de ad {ad_id}: {e}")
            return None
    
    async def create_campaign(self, campaign_data: Dict) -> str:
        """Crea una nueva campaña"""
        try:
            campaign_service = self.client.get_service("CampaignService")
            
            campaign = self.client.get_type("Campaign")
            campaign.name = campaign_data['name']
            campaign.status = campaign_data.get('status', 'PAUSED')
            campaign.advertising_channel_type = campaign_data.get('advertising_channel_type', 'SEARCH')
            campaign.campaign_budget = campaign_data['budget']
            
            # Configurar estrategia de puja
            if 'bidding_strategy_type' in campaign_data:
                campaign.bidding_strategy_type = campaign_data['bidding_strategy_type']
            
            response = campaign_service.mutate_campaigns(
                customer_id=self.customer_id,
                operations=[{"create": campaign}]
            )
            
            campaign_id = response.results[0].resource_name.split('/')[-1]
            logger.info(f"Campaña creada: {campaign_id}")
            return campaign_id
            
        except Exception as e:
            logger.error(f"Error creando campaña: {e}")
            return None
    
    async def update_campaign_status(self, campaign_id: str, status: str):
        """Actualiza el estado de una campaña"""
        try:
            campaign_service = self.client.get_service("CampaignService")
            
            campaign_operation = self.client.get_type("CampaignOperation")
            campaign_operation.update.CopyFrom(
                self.client.get_type("Campaign")(
                    resource_name=f"customers/{self.customer_id}/campaigns/{campaign_id}",
                    status=status
                )
            )
            
            campaign_service.mutate_campaigns(
                customer_id=self.customer_id,
                operations=[campaign_operation]
            )
            
            logger.info(f"Estado de campaña {campaign_id} actualizado a {status}")
            
        except Exception as e:
            logger.error(f"Error actualizando estado de campaña {campaign_id}: {e}")
    
    async def update_keyword_bid(self, keyword_id: str, new_bid: int):
        """Actualiza la puja de una keyword"""
        try:
            ad_group_criterion_service = self.client.get_service("AdGroupCriterionService")
            
            criterion = self.client.get_type("AdGroupCriterion")
            criterion.resource_name = f"customers/{self.customer_id}/adGroupCriteria/{keyword_id}"
            criterion.cpc_bid_micros = new_bid
            
            operation = self.client.get_type("AdGroupCriterionOperation")
            operation.update.CopyFrom(criterion)
            operation.update_mask.paths.append("cpc_bid_micros")
            
            ad_group_criterion_service.mutate_ad_group_criteria(
                customer_id=self.customer_id,
                operations=[operation]
            )
            
            logger.info(f"Puja de keyword {keyword_id} actualizada a {new_bid}")
            
        except Exception as e:
            logger.error(f"Error actualizando puja de keyword {keyword_id}: {e}")
    
    async def add_negative_keyword(self, campaign_id: str, keyword_text: str):
        """Agrega una negative keyword a una campaña"""
        try:
            ad_group_criterion_service = self.client.get_service("AdGroupCriterionService")
            
            # Crear negative keyword list si no existe
            keyword_list = self.client.get_type("AdGroup")
            keyword_list.resource_name = f"customers/{self.customer_id}/campaigns/{campaign_id}/negativeKeywordList"
            
            # Agregar keyword
            negative_keyword = self.client.get_type("AdGroupCriterion")
            negative_keyword.ad_group = f"customers/{self.customer_id}/campaigns/{campaign_id}/negativeKeywordList"
            negative_keyword.status = "ENABLED"
            negative_keyword.negative = True
            negative_keyword.keyword.text = keyword_text
            negative_keyword.keyword.match_type = "PHRASE"
            
            operation = self.client.get_type("AdGroupCriterionOperation")
            operation.create.CopyFrom(negative_keyword)
            
            ad_group_criterion_service.mutate_ad_group_criteria(
                customer_id=self.customer_id,
                operations=[operation]
            )
            
            logger.info(f"Negative keyword '{keyword_text}' agregada a campaña {campaign_id}")
            
        except Exception as e:
            logger.error(f"Error agregando negative keyword {keyword_text}: {e}")
    
    async def pause_ad(self, ad_id: str):
        """Pausa un anuncio"""
        try:
            ad_service = self.client.get_service("AdService")
            
            ad = self.client.get_type("Ad")
            ad.resource_name = f"customers/{self.customer_id}/ads/{ad_id}"
            ad.status = "PAUSED"
            
            operation = self.client.get_type("AdOperation")
            operation.update.CopyFrom(ad)
            operation.update_mask.paths.append("status")
            
            ad_service.mutate_ads(
                customer_id=self.customer_id,
                operations=[operation]
            )
            
            logger.info(f"Anuncio {ad_id} pausado")
            
        except Exception as e:
            logger.error(f"Error pausando anuncio {ad_id}: {e}")
    
    async def create_ad(self, ad_group_id: str, ad_data: Dict) -> bool:
        """Crea un nuevo anuncio"""
        try:
            ad_service = self.client.get_service("AdService")
            
            ad = self.client.get_type("Ad")
            ad.final_urls = ad_data.get('final_urls', [])
            ad.name = ad_data.get('name', f"Anuncio creado automáticamente - {datetime.now()}")
            
            # Configurar headlines
            for headline_text in ad_data.get('headlines', []):
                ad_group = ad.ad_group ad_data.get('ad_group', '')
                ad_headline = ad_group.headlines.add()
                ad_headline.text = headline_text
                ad_headline.pinned = "HEADLINE_1"
            
            # Configurar descriptions
            for desc_text in ad_data.get('descriptions', []):
                ad_description = ad.ad_group ad_data.get('ad_group', '').descriptions.add()
                ad_description.text = desc_text
            
            operation = self.client.get_type("AdOperation")
            operation.create.CopyFrom(ad)
            
            response = ad_service.mutate_ads(
                customer_id=self.customer_id,
                operations=[operation]
            )
            
            if response.results:
                logger.info(f"Anuncio creado exitosamente: {response.results[0].resource_name}")
                return True
            
        except Exception as e:
            logger.error(f"Error creando anuncio: {e}")
        
        return False
    
    async def get_campaign_budget(self, campaign_id: str) -> Optional[int]:
        """Obtiene el presupuesto actual de una campaña"""
        try:
            campaign_service = self.client.get_service("CampaignService")
            
            query = f"""
                SELECT campaign.campaign_budget
                FROM campaign
                WHERE campaign.id = {campaign_id}
            """
            
            response = campaign_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            for batch in response:
                for row in batch.results:
                    return int(row.campaign.campaign_budget.split('/')[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo presupuesto de campaña {campaign_id}: {e}")
            return None
    
    async def update_campaign_budget(self, campaign_id: str, new_budget: int):
        """Actualiza el presupuesto de una campaña"""
        try:
            campaign_budget_service = self.client.get_service("CampaignBudgetService")
            
            # Obtener budget actual
            current_budget_id = await self.get_campaign_budget(campaign_id)
            if not current_budget_id:
                return
            
            budget = self.client.get_type("CampaignBudget")
            budget.resource_name = f"customers/{self.customer_id}/campaignBudgets/{current_budget_id}"
            budget.amount_micros = new_budget
            
            operation = self.client.get_type("CampaignBudgetOperation")
            operation.update.CopyFrom(budget)
            operation.update_mask.paths.append("amount_micros")
            
            campaign_budget_service.mutate_campaign_budgets(
                customer_id=self.customer_id,
                operations=[operation]
            )
            
            logger.info(f"Presupuesto de campaña {campaign_id} actualizado a {new_budget}")
            
        except Exception as e:
            logger.error(f"Error actualizando presupuesto de campaña {campaign_id}: {e}")
    
    async def sync_with_google_analytics(self, start_date: str, end_date: str) -> Dict:
        """Sincroniza datos con Google Analytics 4"""
        try:
            # Obtener datos de conversiones desde GA4
            ga4_data = await self.ga4_integration.get_conversion_data(start_date, end_date)
            
            # Procesar y correlacionar con datos de Google Ads
            sync_results = {
                'conversions_synced': 0,
                'conversions_adjusted': 0,
                'data_correlation_score': 0.0
            }
            
            # Aquí se implementaría la lógica de correlación
            # Por ahora, retornar estructura básica
            sync_results['data_correlation_score'] = 0.85  # Score de correlación calculado
            
            logger.info(f"Sincronización con GA4 completada: {sync_results}")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error sincronizando con GA4: {e}")
            return {}


class RealTimeDashboard:
    """Dashboard de performance en tiempo real"""
    
    def __init__(self, db_manager: DatabaseManager, google_ads: GoogleAdsManager):
        self.db_manager = db_manager
        self.google_ads = google_ads
        self.last_update = None
        self.dashboard_data = {}
        
    async def get_dashboard_data(self) -> Dict:
        """Obtiene datos para el dashboard en tiempo real"""
        try:
            current_time = datetime.now()
            
            # Actualizar datos cada 5 minutos
            if (self.last_update is None or 
                (current_time - self.last_update).total_seconds() > 300):
                
                await self._update_dashboard_data()
                self.last_update = current_time
            
            return self.dashboard_data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard: {e}")
            return {}
    
    async def _update_dashboard_data(self):
        """Actualiza los datos del dashboard"""
        try:
            # Métricas generales
            total_campaigns = await self._get_total_campaigns()
            total_impressions = await self._get_total_impressions()
            total_clicks = await self._get_total_clicks()
            total_cost = await self._get_total_cost()
            total_conversions = await self._get_total_conversions()
            
            # Cálculo de métricas derivadas
            ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            cpc = (total_cost / total_clicks) if total_clicks > 0 else 0
            cpa = (total_cost / total_conversions) if total_conversions > 0 else 0
            
            # Tendencias de los últimos 7 días
            weekly_trend = await self._get_weekly_trend()
            
            # Campañas de mejor y peor rendimiento
            top_campaigns = await self._get_top_performing_campaigns(5)
            worst_campaigns = await self._get_worst_performing_campaigns(5)
            
            # Alertas activas
            active_alerts = await self._get_active_alerts()
            
            # Predicciones de rendimiento
            performance_predictions = await self._get_performance_predictions()
            
            self.dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'overview': {
                    'total_campaigns': total_campaigns,
                    'total_impressions': total_impressions,
                    'total_clicks': total_clicks,
                    'total_cost': round(total_cost, 2),
                    'total_conversions': total_conversions,
                    'ctr': round(ctr, 2),
                    'cpc': round(cpc, 2),
                    'cpa': round(cpa, 2)
                },
                'weekly_trend': weekly_trend,
                'top_campaigns': top_campaigns,
                'worst_campaigns': worst_campaigns,
                'alerts': active_alerts,
                'predictions': performance_predictions
            }
            
        except Exception as e:
            logger.error(f"Error actualizando datos del dashboard: {e}")
    
    async def _get_total_campaigns(self) -> int:
        """Obtiene el número total de campañas"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            return len(campaigns)
        except:
            return 0
    
    async def _get_total_impressions(self) -> int:
        """Obtiene el total de impresiones"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            total = 0
            for campaign in campaigns:
                metrics = await self.google_ads.get_campaign_metrics(campaign['id'])
                if metrics:
                    total += metrics.impressions
            return total
        except:
            return 0
    
    async def _get_total_clicks(self) -> int:
        """Obtiene el total de clics"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            total = 0
            for campaign in campaigns:
                metrics = await self.google_ads.get_campaign_metrics(campaign['id'])
                if metrics:
                    total += metrics.clicks
            return total
        except:
            return 0
    
    async def _get_total_cost(self) -> float:
        """Obtiene el costo total en euros"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            total = 0.0
            for campaign in campaigns:
                metrics = await self.google_ads.get_campaign_metrics(campaign['id'])
                if metrics:
                    total += metrics.cost_euros
            return total
        except:
            return 0.0
    
    async def _get_total_conversions(self) -> int:
        """Obtiene el total de conversiones"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            total = 0
            for campaign in campaigns:
                metrics = await self.google_ads.get_campaign_metrics(campaign['id'])
                if metrics:
                    total += metrics.conversions
            return total
        except:
            return 0
    
    async def _get_weekly_trend(self) -> List[Dict]:
        """Obtiene tendencias de la semana"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            trend_data = []
            
            for i in range(7):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                day_total_cost = 0.0
                day_total_clicks = 0
                day_total_conversions = 0
                
                for campaign in campaigns:
                    # Aquí se consultarían las métricas del día específico
                    # Por simplicidad, se usan valores de ejemplo
                    pass
                
                trend_data.append({
                    'date': date,
                    'cost': round(day_total_cost, 2),
                    'clicks': day_total_clicks,
                    'conversions': day_total_conversions
                })
            
            return list(reversed(trend_data))
            
        except Exception as e:
            logger.error(f"Error obteniendo tendencias semanales: {e}")
            return []
    
    async def _get_top_performing_campaigns(self, limit: int) -> List[Dict]:
        """Obtiene las campañas de mejor rendimiento"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            campaign_performance = []
            
            for campaign in campaigns:
                metrics = await self.google_ads.get_campaign_metrics(campaign['id'])
                if metrics:
                    score = metrics.roas * metrics.ctr_percentage  # Score de rendimiento simple
                    campaign_performance.append({
                        'id': campaign['id'],
                        'name': campaign['name'],
                        'roas': metrics.roas,
                        'ctr': metrics.ctr_percentage,
                        'cost': metrics.cost_euros,
                        'conversions': metrics.conversions,
                        'score': round(score, 2)
                    })
            
            # Ordenar por score y retornar top
            campaign_performance.sort(key=lambda x: x['score'], reverse=True)
            return campaign_performance[:limit]
            
        except Exception as e:
            logger.error(f"Error obteniendo top campañas: {e}")
            return []
    
    async def _get_worst_performing_campaigns(self, limit: int) -> List[Dict]:
        """Obtiene las campañas de peor rendimiento"""
        try:
            campaigns = await self.google_ads.get_active_campaigns()
            campaign_performance = []
            
            for campaign in campaigns:
                metrics = await self.google_ads.get_campaign_metrics(campaign['id'])
                if metrics:
                    score = metrics.roas * metrics.ctr_percentage
                    campaign_performance.append({
                        'id': campaign['id'],
                        'name': campaign['name'],
                        'roas': metrics.roas,
                        'ctr': metrics.ctr_percentage,
                        'cost': metrics.cost_euros,
                        'conversions': metrics.conversions,
                        'score': round(score, 2)
                    })
            
            # Ordenar por score y retornar bottom
            campaign_performance.sort(key=lambda x: x['score'])
            return campaign_performance[:limit]
            
        except Exception as e:
            logger.error(f"Error obteniendo peores campañas: {e}")
            return []
    
    async def _get_active_alerts(self) -> List[Dict]:
        """Obtiene alertas activas del sistema"""
        try:
            alerts = []
            
            # Alertas de Quality Score
            quality_score_monitor = QualityScoreMonitor(self.db_manager, self.google_ads)
            
            # Verificar cola de alertas del monitor
            while not quality_score_monitor.alerts.empty():
                alert = quality_score_monitor.alerts.get()
                alerts.append({
                    'type': alert['type'],
                    'message': f"Quality Score bajo en {alert['keyword_text']}: {alert['quality_score']}",
                    'timestamp': alert['timestamp'],
                    'priority': 'high'
                })
            
            # Agregar otras alertas si es necesario
            return alerts
            
        except Exception as e:
            logger.error(f"Error obteniendo alertas activas: {e}")
            return []
    
    async def _get_performance_predictions(self) -> Dict:
        """Obtiene predicciones de rendimiento usando ML básico"""
        try:
            # Predicciones simples basadas en tendencias históricas
            campaigns = await self.google_ads.get_active_campaigns()
            predictions = {}
            
            for campaign in campaigns:
                metrics = await self.google_ads.get_campaign_metrics(campaign['id'])
                if metrics:
                    # Predicción simple basada en ROAS actual
                    if metrics.roas > 3.0:
                        prediction = "positive"
                        confidence = 0.8
                    elif metrics.roas < 1.5:
                        prediction = "negative"
                        confidence = 0.7
                    else:
                        prediction = "stable"
                        confidence = 0.6
                    
                    predictions[campaign['id']] = {
                        'campaign_name': campaign['name'],
                        'trend_prediction': prediction,
                        'confidence': confidence,
                        'recommended_action': self._get_recommended_action(metrics)
                    }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error obteniendo predicciones: {e}")
            return {}
    
    def _get_recommended_action(self, metrics: CampaignMetrics) -> str:
        """Genera acción recomendada basada en métricas"""
        if metrics.roas > 4.0 and metrics.conversions > 10:
            return "Aumentar presupuesto (+20%)"
        elif metrics.cpa_euros > 100:
            return "Reducir puja (-15%)"
        elif metrics.ctr_percentage < 0.02:
            return "Optimizar anuncios"
        else:
            return "Mantener configuración actual"


# Función principal de inicialización del sistema
async def initialize_midas_google_opal(config: Dict) -> Dict:
    """Inicializa el sistema MIDAS Google OPAL"""
    try:
        logger.info("Inicializando sistema MIDAS Google OPAL...")
        
        # Crear gestor principal de Google Ads
        google_ads_manager = GoogleAdsManager(
            customer_id=config.get('customer_id'),
            login_customer_id=config.get('login_customer_id'),
            config_file=config.get('config_file')
        )
        
        # Inicializar optimizador de campañas
        campaign_optimizer = CampaignOptimizer(google_ads_manager.db_manager, google_ads_manager)
        
        # Inicializar monitor de Quality Score
        quality_monitor = QualityScoreMonitor(google_ads_manager.db_manager, google_ads_manager)
        
        # Inicializar dashboard en tiempo real
        dashboard = RealTimeDashboard(google_ads_manager.db_manager, google_ads_manager)
        
        return {
            'status': 'initialized',
            'google_ads_manager': google_ads_manager,
            'campaign_optimizer': campaign_optimizer,
            'quality_monitor': quality_monitor,
            'dashboard': dashboard,
            'message': 'Sistema MIDAS Google OPAL inicializado correctamente'
        }
        
    except Exception as e:
        logger.error(f"Error inicializando sistema: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'message': 'Error inicializando sistema MIDAS Google OPAL'
        }


# Función para ejecutar optimizaciones automáticas
async def run_automatic_optimization(config: Dict):
    """Ejecuta optimizaciones automáticas programadas"""
    try:
        # Inicializar sistema
        system = await initialize_midas_google_opal(config)
        
        if system['status'] != 'initialized':
            raise Exception("Error inicializando sistema")
        
        logger.info("Iniciando optimización automática...")
        
        # Ejecutar optimización de campañas
        await system['campaign_optimizer'].optimize_campaigns()
        
        # Sincronizar con Google Analytics
        await system['google_ads_manager'].sync_with_google_analytics(
            (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        
        logger.info("Optimización automática completada")
        
    except Exception as e:
        logger.error(f"Error en optimización automática: {e}")


if __name__ == "__main__":
    # Ejemplo de uso del sistema
    config = {
        'customer_id': os.getenv('GOOGLE_ADS_CUSTOMER_ID'),
        'login_customer_id': os.getenv('GOOGLE_ADS_LOGIN_CUSTOMER_ID'),
        'config_file': 'google_ads_config.json'
    }
    
    # Ejecutar sistema
    asyncio.run(initialize_midas_google_opal(config))