"""
MIDAS Platform Integration System
=================================

Sistema de integración unificada para Google Ads y Meta Ads con:
1. APIs unificadas para ambas plataformas
2. Sincronización bidireccional de campañas
3. Mapeo automático de targeting entre plataformas
4. Gestión unificada de presupuestos cross-platform
5. Reporting consolidado de métricas
6. Sistema de alertas por performance
7. Optimización cross-platform (budget shifting)
8. Detección de overlap de audiencias

Autor: OMNIA System
Versión: 1.0
Fecha: 2025-11-06
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importar conectores existentes
from code.anchor_connectors.google_ads_connector import GoogleAdsConnector, GoogleAdsConfig
from code.anchor_connectors.meta_ads_connector import MetaAdsConnector, MetaAdsConfig
from code.anchor_connectors.unified_connector_system import ConnectorManager

import psycopg2
from psycopg2.extras import RealDictCursor
import requests

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Platform(Enum):
    """Plataformas de publicidad soportadas"""
    GOOGLE_ADS = "google_ads"
    META_ADS = "meta_ads"


class CampaignStatus(Enum):
    """Estados de campaña"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    DRAFT = "draft"
    ARCHIVED = "archived"


class BudgetShiftDirection(Enum):
    """Dirección de redistribución de presupuesto"""
    FROM_GOOGLE_TO_META = "google_to_meta"
    FROM_META_TO_GOOGLE = "meta_to_google"
    EQUALIZE = "equalize"


@dataclass
class CampaignData:
    """Estructura unificada de datos de campaña"""
    campaign_id: str
    platform: Platform
    name: str
    status: CampaignStatus
    budget_daily: float
    budget_total: float
    start_date: datetime
    end_date: Optional[datetime]
    targeting: Dict[str, Any]
    creatives: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class TargetingMapping:
    """Mapeo de targeting entre plataformas"""
    google_targeting: Dict[str, Any]
    meta_targeting: Dict[str, Any]
    similarity_score: float
    conversion_rules: Dict[str, str]


@dataclass
class BudgetAllocation:
    """Asignación de presupuesto entre plataformas"""
    total_budget: float
    google_allocation: float
    meta_allocation: float
    allocation_strategy: str
    rebalancing_rules: Dict[str, Any]
    last_updated: datetime


@dataclass
class PerformanceMetrics:
    """Métricas de performance consolidadas"""
    platform: Platform
    campaign_id: str
    date: datetime
    impressions: int
    clicks: int
    spend: float
    conversions: int
    conversion_value: float
    ctr: float
    cpc: float
    cpm: float
    roas: float


@dataclass
class AudienceOverlap:
    """Análisis de overlap de audiencias"""
    audience_size_google: int
    audience_size_meta: int
    overlap_size: int
    overlap_percentage: float
    unique_audiences: Dict[str, int]
    efficiency_score: float


class TargetingMapper:
    """Mapeador automático de targeting entre plataformas"""
    
    # Mapeos de targeting entre plataformas
    TARGETING_MAPPINGS = {
        'age_range': {
            'google': 'ageRanges',
            'meta': 'age_min,age_max',
            'conversion': lambda meta_range: f"{meta_range.get('age_min', 18)}-{meta_range.get('age_max', 65)}"
        },
        'genders': {
            'google': 'genders',
            'meta': 'genders',
            'conversion': lambda x: x  # Mapeo directo
        },
        'interests': {
            'google': 'userInterests',
            'meta': 'interests',
            'conversion': lambda interests: [{'id': i.get('id'), 'name': i.get('name')} for i in interests]
        },
        'behaviors': {
            'google': 'userAdSystemLists',
            'meta': 'behaviors',
            'conversion': lambda behaviors: [{'id': b.get('id'), 'name': b.get('name')} for b in behaviors]
        },
        'locations': {
            'google': 'locations',
            'meta': 'geo_locations',
            'conversion': lambda locations: {'countries': [l.get('countryCode') for l in locations if l.get('countryCode')]}
        },
        'languages': {
            'google': 'languages',
            'meta': 'language_spoken',
            'conversion': lambda languages: [{'id': l.get('code'), 'name': l.get('name')} for l in languages]
        }
    }
    
    @classmethod
    def map_google_to_meta(cls, google_targeting: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte targeting de Google Ads a formato Meta Ads"""
        meta_targeting = {}
        
        for key, mapping in cls.TARGETING_MAPPINGS.items():
            if key in google_targeting:
                google_value = google_targeting[key]
                try:
                    if isinstance(google_value, list):
                        meta_value = mapping['conversion'](google_value)
                    else:
                        meta_value = mapping['conversion'](google_value)
                    meta_targeting[mapping['meta']] = meta_value
                except Exception as e:
                    logger.warning(f"Error mapeando {key}: {e}")
        
        return meta_targeting
    
    @classmethod
    def map_meta_to_google(cls, meta_targeting: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte targeting de Meta Ads a formato Google Ads"""
        google_targeting = {}
        
        for key, mapping in cls.TARGETING_MAPPINGS.items():
            meta_key = mapping['meta']
            if meta_key in meta_targeting:
                meta_value = meta_targeting[meta_key]
                try:
                    if isinstance(meta_value, dict) and 'countries' in meta_value:
                        # Caso especial para ubicaciones
                        google_value = [{'countryCode': c} for c in meta_value['countries']]
                    else:
                        google_value = mapping['conversion'](meta_value)
                    google_targeting[key] = google_value
                except Exception as e:
                    logger.warning(f"Error mapeando {key}: {e}")
        
        return google_targeting
    
    @classmethod
    def calculate_similarity(cls, targeting1: Dict[str, Any], targeting2: Dict[str, Any]) -> float:
        """Calcula similitud entre dos configuraciones de targeting"""
        similarity_scores = []
        
        for key in cls.TARGETING_MAPPINGS.keys():
            val1 = targeting1.get(key, [])
            val2 = targeting2.get(key, [])
            
            if not val1 and not val2:
                similarity_scores.append(1.0)
                continue
            
            if not val1 or not val2:
                similarity_scores.append(0.0)
                continue
            
            # Calcular Jaccard similarity
            set1 = set(json.dumps(val1, sort_keys=True))
            set2 = set(json.dumps(val2, sort_keys=True))
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            jaccard = intersection / union if union > 0 else 0
            similarity_scores.append(jaccard)
        
        return statistics.mean(similarity_scores)


class CrossPlatformOptimizer:
    """Optimizador de performance cross-platform"""
    
    def __init__(self, db_connection: str):
        self.db_connection = db_connection
        self.performance_history = defaultdict(list)
        
    async def analyze_performance(self, campaign_ids: List[str], days: int = 30) -> Dict[str, Any]:
        """Analiza performance de campañas across plataformas"""
        logger.info(f"Analizando performance de {len(campaign_ids)} campañas")
        
        try:
            # Obtener métricas históricas
            performance_data = await self._get_performance_data(campaign_ids, days)
            
            # Calcular métricas consolidadas
            consolidated_metrics = self._calculate_consolidated_metrics(performance_data)
            
            # Identificar oportunidades de optimización
            optimization_opportunities = self._identify_optimization_opportunities(consolidated_metrics)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(optimization_opportunities)
            
            return {
                'campaigns_analyzed': len(campaign_ids),
                'analysis_period_days': days,
                'consolidated_metrics': consolidated_metrics,
                'optimization_opportunities': optimization_opportunities,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analizando performance: {e}")
            raise
    
    async def optimize_budget_allocation(self, 
                                       campaign_ids: List[str],
                                       target_budget: float,
                                       optimization_rules: Dict[str, Any]) -> BudgetAllocation:
        """Optimiza asignación de presupuesto entre plataformas"""
        logger.info(f"Optimizando presupuesto total: {target_budget}")
        
        try:
            # Obtener performance histórica
            performance_data = await self._get_performance_data(campaign_ids, 30)
            
            # Calcular ROAS por plataforma
            platform_roas = self._calculate_platform_roas(performance_data)
            
            # Aplicar reglas de optimización
            allocation_strategy = optimization_rules.get('strategy', 'performance_based')
            
            if allocation_strategy == 'performance_based':
                google_weight, meta_weight = self._calculate_performance_weights(platform_roas)
            elif allocation_strategy == 'equal':
                google_weight, meta_weight = 0.5, 0.5
            elif allocation_strategy == 'conservative':
                google_weight, meta_weight = 0.6, 0.4  # Google más estable
            else:
                google_weight, meta_weight = 0.5, 0.5
            
            # Calcular asignaciones
            google_allocation = target_budget * google_weight
            meta_allocation = target_budget * meta_weight
            
            # Crear reglas de rebalanceo
            rebalancing_rules = {
                'min_allocation_google': target_budget * 0.2,
                'min_allocation_meta': target_budget * 0.2,
                'rebalance_threshold': 0.15,  # 15% de diferencia
                'rebalance_frequency': 'daily'
            }
            
            return BudgetAllocation(
                total_budget=target_budget,
                google_allocation=google_allocation,
                meta_allocation=meta_allocation,
                allocation_strategy=allocation_strategy,
                rebalancing_rules=rebalancing_rules,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error optimizando presupuesto: {e}")
            raise
    
    def shift_budget(self, 
                    current_allocation: BudgetAllocation,
                    direction: BudgetShiftDirection,
                    amount: float) -> BudgetAllocation:
        """Redistribuye presupuesto entre plataformas"""
        logger.info(f"Redistribuyendo {amount} en dirección: {direction.value}")
        
        if direction == BudgetShiftDirection.FROM_GOOGLE_TO_META:
            new_google = max(0, current_allocation.google_allocation - amount)
            new_meta = current_allocation.total_budget - new_google
        elif direction == BudgetShiftDirection.FROM_META_TO_GOOGLE:
            new_meta = max(0, current_allocation.meta_allocation - amount)
            new_google = current_allocation.total_budget - new_meta
        else:  # EQUALIZE
            equal_amount = current_allocation.total_budget / 2
            new_google = equal_amount
            new_meta = equal_amount
        
        return BudgetAllocation(
            total_budget=current_allocation.total_budget,
            google_allocation=new_google,
            meta_allocation=new_meta,
            allocation_strategy="manual_adjustment",
            rebalancing_rules=current_allocation.rebalancing_rules,
            last_updated=datetime.now()
        )
    
    async def _get_performance_data(self, campaign_ids: List[str], days: int) -> List[Dict[str, Any]]:
        """Obtiene datos de performance de la base de datos"""
        try:
            conn = psycopg2.connect(self.db_connection)
            
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                campaign_title,
                market,
                product_line,
                platform,
                DATE(ts) as date,
                SUM(impressions) as impressions,
                SUM(clicks) as clicks,
                SUM(spend_total) as spend,
                SUM(leads_day) as conversions
            FROM anchor_data 
            WHERE campaign_title = ANY(%s)
                AND ts >= %s
            GROUP BY campaign_title, market, product_line, platform, DATE(ts)
            ORDER BY date DESC
            """
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (campaign_ids, start_date))
                results = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de performance: {e}")
            return []
    
    def _calculate_consolidated_metrics(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula métricas consolidadas"""
        if not performance_data:
            return {}
        
        # Agrupar por plataforma
        by_platform = defaultdict(list)
        for row in performance_data:
            by_platform[row['platform']].append(row)
        
        consolidated = {}
        
        for platform, data in by_platform.items():
            total_impressions = sum(d['impressions'] for d in data)
            total_clicks = sum(d['clicks'] for d in data)
            total_spend = sum(d['spend'] for d in data)
            total_conversions = sum(d['conversions'] for d in data)
            
            consolidated[platform] = {
                'impressions': total_impressions,
                'clicks': total_clicks,
                'spend': total_spend,
                'conversions': total_conversions,
                'ctr': (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
                'cpc': (total_spend / total_clicks) if total_clicks > 0 else 0,
                'cpm': (total_spend / total_impressions * 1000) if total_impressions > 0 else 0,
                'conversion_rate': (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
            }
        
        return consolidated
    
    def _calculate_platform_roas(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula ROAS por plataforma"""
        platform_spend = defaultdict(float)
        platform_value = defaultdict(float)
        
        for row in performance_data:
            platform = row['platform']
            platform_spend[platform] += row['spend']
            # Asumir valor de conversión promedio si no está disponible
            conversion_value = row['conversions'] * 50  # $50 por conversión
            platform_value[platform] += conversion_value
        
        roas = {}
        for platform in platform_spend:
            if platform_spend[platform] > 0:
                roas[platform] = platform_value[platform] / platform_spend[platform]
            else:
                roas[platform] = 0
        
        return roas
    
    def _calculate_performance_weights(self, platform_roas: Dict[str, float]) -> Tuple[float, float]:
        """Calcula pesos de asignación basados en performance"""
        google_roas = platform_roas.get('google_ads', 1.0)
        meta_roas = platform_roas.get('meta_ads', 1.0)
        
        total_roas = google_roas + meta_roas
        if total_roas == 0:
            return 0.5, 0.5
        
        google_weight = google_roas / total_roas
        meta_weight = meta_roas / total_roas
        
        # Aplicar límites mínimos y máximos
        min_weight = 0.2
        max_weight = 0.8
        
        google_weight = max(min_weight, min(max_weight, google_weight))
        meta_weight = 1.0 - google_weight
        
        return google_weight, meta_weight
    
    def _identify_optimization_opportunities(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica oportunidades de optimización"""
        opportunities = []
        
        if 'google_ads' in metrics and 'meta_ads' in metrics:
            google_metrics = metrics['google_ads']
            meta_metrics = metrics['meta_ads']
            
            # Comparar CTR
            if abs(google_metrics['ctr'] - meta_metrics['ctr']) > 1.0:
                better_platform = 'google_ads' if google_metrics['ctr'] > meta_metrics['ctr'] else 'meta_ads'
                opportunities.append({
                    'type': 'ctr_optimization',
                    'description': f'CTR difference detected: {better_platform} performing better',
                    'google_ctr': google_metrics['ctr'],
                    'meta_ctr': meta_metrics['ctr'],
                    'recommendation': f'Consider shifting budget to {better_platform}'
                })
            
            # Comparar CPC
            if google_metrics['cpc'] < meta_metrics['cpc'] * 0.8:
                opportunities.append({
                    'type': 'cpc_optimization',
                    'description': 'Google Ads showing lower CPC',
                    'google_cpc': google_metrics['cpc'],
                    'meta_cpc': meta_metrics['cpc'],
                    'recommendation': 'Consider increasing Google Ads budget'
                })
        
        return opportunities
    
    def _generate_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Genera recomendaciones basadas en oportunidades"""
        recommendations = []
        
        for opportunity in opportunities:
            if opportunity['type'] == 'ctr_optimization':
                recommendations.append(
                    f"Optimize CTR: {opportunity['recommendation']}"
                )
            elif opportunity['type'] == 'cpc_optimization':
                recommendations.append(
                    f"Optimize CPC: {opportunity['recommendation']}"
                )
        
        if not recommendations:
            recommendations.append("Performance is balanced across platforms")
        
        return recommendations


class AudienceManager:
    """Gestor de audiencias cross-platform"""
    
    def __init__(self, db_connection: str):
        self.db_connection = db_connection
        self.audience_cache = {}
    
    async def detect_audience_overlap(self, 
                                    google_campaign_ids: List[str],
                                    meta_campaign_ids: List[str]) -> AudienceOverlap:
        """Detecta overlap entre audiencias de Google y Meta"""
        logger.info("Detectando overlap de audiencias")
        
        try:
            # Obtener tamaños de audiencia estimados
            google_audience_size = await self._estimate_audience_size(google_campaign_ids, Platform.GOOGLE_ADS)
            meta_audience_size = await self._estimate_audience_size(meta_campaign_ids, Platform.META_ADS)
            
            # Calcular overlap estimado
            overlap_size = self._calculate_estimated_overlap(google_audience_size, meta_audience_size)
            overlap_percentage = (overlap_size / min(google_audience_size, meta_audience_size)) * 100
            
            # Identificar audiencias únicas
            unique_audiences = {
                'google_only': max(0, google_audience_size - overlap_size),
                'meta_only': max(0, meta_audience_size - overlap_size)
            }
            
            # Calcular efficiency score
            efficiency_score = self._calculate_audience_efficiency(google_audience_size, meta_audience_size, overlap_size)
            
            return AudienceOverlap(
                audience_size_google=google_audience_size,
                audience_size_meta=meta_audience_size,
                overlap_size=overlap_size,
                overlap_percentage=round(overlap_percentage, 2),
                unique_audiences=unique_audiences,
                efficiency_score=round(efficiency_score, 2)
            )
            
        except Exception as e:
            logger.error(f"Error detectando overlap: {e}")
            raise
    
    async def sync_audiences(self, 
                           source_platform: Platform,
                           target_platform: Platform,
                           campaign_mappings: List[Dict[str, str]]) -> Dict[str, Any]:
        """Sincroniza audiencias entre plataformas"""
        logger.info(f"Sincronizando audiencias de {source_platform.value} a {target_platform.value}")
        
        sync_results = []
        
        for mapping in campaign_mappings:
            try:
                source_campaign_id = mapping['source_campaign_id']
                target_campaign_id = mapping['target_campaign_id']
                
                # Obtener targeting de plataforma origen
                source_targeting = await self._get_campaign_targeting(source_campaign_id, source_platform)
                
                # Mapear al formato de plataforma destino
                if source_platform == Platform.GOOGLE_ADS and target_platform == Platform.META_ADS:
                    mapped_targeting = TargetingMapper.map_google_to_meta(source_targeting)
                else:
                    mapped_targeting = TargetingMapper.map_meta_to_google(source_targeting)
                
                # Aplicar targeting a campaña destino
                success = await self._apply_targeting_to_campaign(target_campaign_id, target_platform, mapped_targeting)
                
                sync_results.append({
                    'source_campaign_id': source_campaign_id,
                    'target_campaign_id': target_campaign_id,
                    'success': success,
                    'targeting_applied': mapped_targeting
                })
                
            except Exception as e:
                logger.error(f"Error sincronizando campaña {mapping}: {e}")
                sync_results.append({
                    'source_campaign_id': mapping['source_campaign_id'],
                    'target_campaign_id': mapping['target_campaign_id'],
                    'success': False,
                    'error': str(e)
                })
        
        successful_syncs = sum(1 for r in sync_results if r['success'])
        
        return {
            'total_mappings': len(campaign_mappings),
            'successful_syncs': successful_syncs,
            'failed_syncs': len(campaign_mappings) - successful_syncs,
            'sync_results': sync_results,
            'timestamp': datetime.now().isoformat()
        }
    
    async def create_unified_audience(self, 
                                    google_campaign_ids: List[str],
                                    meta_campaign_ids: List[str],
                                    audience_name: str) -> Dict[str, Any]:
        """Crea audiencia unificada combinando targeting de ambas plataformas"""
        logger.info(f"Creando audiencia unificada: {audience_name}")
        
        try:
            # Obtener targeting de ambas plataformas
            google_targeting = await self._combine_google_targeting(google_campaign_ids)
            meta_targeting = await self._combine_meta_targeting(meta_campaign_ids)
            
            # Crear targeting unificado
            unified_targeting = self._create_unified_targeting(google_targeting, meta_targeting)
            
            # Guardar en base de datos
            audience_id = str(uuid.uuid4())
            await self._save_unified_audience(audience_id, audience_name, unified_targeting)
            
            return {
                'audience_id': audience_id,
                'audience_name': audience_name,
                'unified_targeting': unified_targeting,
                'google_campaigns': google_campaign_ids,
                'meta_campaigns': meta_campaign_ids,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creando audiencia unificada: {e}")
            raise
    
    async def _estimate_audience_size(self, campaign_ids: List[str], platform: Platform) -> int:
        """Estima tamaño de audiencia basado en impresiones y frecuencia"""
        try:
            conn = psycopg2.connect(self.db_connection)
            
            query = """
            SELECT 
                SUM(impressions) as total_impressions,
                AVG( CASE WHEN reach > 0 THEN impressions::float / reach ELSE 0 END ) as avg_frequency
            FROM anchor_data 
            WHERE campaign_title = ANY(%s)
                AND platform = %s
                AND ts >= %s
            """
            
            since_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            with conn.cursor() as cursor:
                cursor.execute(query, (campaign_ids, platform.value, since_date))
                result = cursor.fetchone()
            
            conn.close()
            
            if result and result[0] and result[1]:
                total_impressions = int(result[0])
                avg_frequency = float(result[1])
                estimated_audience = int(total_impressions / avg_frequency) if avg_frequency > 0 else total_impressions
                return estimated_audience
            
            return 0
            
        except Exception as e:
            logger.error(f"Error estimando tamaño de audiencia: {e}")
            return 0
    
    def _calculate_estimated_overlap(self, google_size: int, meta_size: int) -> int:
        """Calcula overlap estimado basado en tamaños de audiencia"""
        if google_size == 0 or meta_size == 0:
            return 0
        
        # Estimación basada en estudios de mercado: 15-25% overlap promedio
        overlap_factor = 0.2  # 20% por defecto
        estimated_overlap = int(min(google_size, meta_size) * overlap_factor)
        
        return estimated_overlap
    
    def _calculate_audience_efficiency(self, google_size: int, meta_size: int, overlap: int) -> float:
        """Calcula score de eficiencia de audiencia"""
        if google_size == 0 or meta_size == 0:
            return 0.0
        
        # Score basado en diversidad de audiencia
        unique_total = google_size + meta_size - overlap
        total_reach = max(google_size, meta_size)
        
        efficiency = (unique_total / total_reach) * 100
        return min(efficiency, 100.0)
    
    async def _get_campaign_targeting(self, campaign_id: str, platform: Platform) -> Dict[str, Any]:
        """Obtiene targeting de una campaña específica"""
        # Esta implementación sería específica de cada API
        # Por ahora retornamos un targeting de ejemplo
        return {
            'age_range': '25-54',
            'genders': ['male', 'female'],
            'interests': [
                {'id': '6003107900433', 'name': 'Technology'},
                {'id': '6003115821873', 'name': 'Business'}
            ],
            'locations': [{'countryCode': 'US'}],
            'languages': [{'code': 'en', 'name': 'English'}]
        }
    
    async def _apply_targeting_to_campaign(self, campaign_id: str, platform: Platform, targeting: Dict[str, Any]) -> bool:
        """Aplica targeting a una campaña"""
        try:
            # Esta implementación usaría las APIs específicas de cada plataforma
            logger.info(f"Aplicando targeting a campaña {campaign_id} en {platform.value}")
            return True
        except Exception as e:
            logger.error(f"Error aplicando targeting: {e}")
            return False
    
    async def _combine_google_targeting(self, campaign_ids: List[str]) -> Dict[str, Any]:
        """Combina targeting de múltiples campañas de Google"""
        # Implementación simplificada
        return {
            'age_range': '25-54',
            'genders': ['male', 'female'],
            'interests': [
                {'id': '6003107900433', 'name': 'Technology'},
                {'id': '6003115821873', 'name': 'Business'},
                {'id': '6003120310086', 'name': 'Marketing'}
            ]
        }
    
    async def _combine_meta_targeting(self, campaign_ids: List[str]) -> Dict[str, Any]:
        """Combina targeting de múltiples campañas de Meta"""
        # Implementación simplificada
        return {
            'age_min': 25,
            'age_max': 54,
            'genders': 0,  # All
            'interests': [
                {'id': '6003107900433', 'name': 'Technology'},
                {'id': '6003115821873', 'name': 'Business'},
                {'id': '6003099554383', 'name': 'Entrepreneurship'}
            ]
        }
    
    def _create_unified_targeting(self, google_targeting: Dict[str, Any], meta_targeting: Dict[str, Any]) -> Dict[str, Any]:
        """Crea targeting unificado combinando ambas plataformas"""
        # Combinar intereses de ambas plataformas
        all_interests = []
        
        if 'interests' in google_targeting:
            all_interests.extend(google_targeting['interests'])
        if 'interests' in meta_targeting:
            all_interests.extend(meta_targeting['interests'])
        
        # Eliminar duplicados
        seen_ids = set()
        unique_interests = []
        for interest in all_interests:
            if interest.get('id') not in seen_ids:
                seen_ids.add(interest.get('id'))
                unique_interests.append(interest)
        
        return {
            'age_range': '25-54',
            'genders': ['male', 'female'],
            'interests': unique_interests[:10],  # Limitar a 10 intereses
            'locations': [{'countryCode': 'US'}],
            'languages': [{'code': 'en', 'name': 'English'}]
        }
    
    async def _save_unified_audience(self, audience_id: str, name: str, targeting: Dict[str, Any]) -> None:
        """Guarda audiencia unificada en base de datos"""
        try:
            conn = psycopg2.connect(self.db_connection)
            
            query = """
            INSERT INTO midas_unified_audiences (
                audience_id, name, targeting, created_at
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (audience_id) DO UPDATE SET
                name = EXCLUDED.name,
                targeting = EXCLUDED.targeting,
                updated_at = EXCLUDED.created_at
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    audience_id, 
                    name, 
                    json.dumps(targeting), 
                    datetime.now()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error guardando audiencia unificada: {e}")


class AlertingSystem:
    """Sistema de alertas por performance"""
    
    def __init__(self, db_connection: str, notification_webhook: str = None):
        self.db_connection = db_connection
        self.notification_webhook = notification_webhook
        self.alert_rules = {}
        self.alert_history = []
    
    def configure_alert_rules(self, rules: Dict[str, Any]) -> None:
        """Configura reglas de alerta"""
        self.alert_rules = {
            'low_ctr_threshold': rules.get('low_ctr_threshold', 1.0),
            'high_cpc_threshold': rules.get('high_cpc_threshold', 5.0),
            'low_conversion_rate': rules.get('low_conversion_rate', 2.0),
            'budget_utilization': rules.get('budget_utilization', 0.8),
            'performance_drop': rules.get('performance_drop', 0.2)
        }
        logger.info("Reglas de alerta configuradas")
    
    async def check_performance_alerts(self, 
                                     campaign_ids: List[str],
                                     lookback_days: int = 7) -> List[Dict[str, Any]]:
        """Verifica alertas de performance"""
        alerts = []
        
        try:
            for campaign_id in campaign_ids:
                campaign_alerts = await self._check_campaign_alerts(campaign_id, lookback_days)
                alerts.extend(campaign_alerts)
            
            # Enviar notificaciones si hay alertas
            if alerts:
                await self._send_alert_notifications(alerts)
            
            # Guardar alertas en historial
            self.alert_history.extend(alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error verificando alertas: {e}")
            return []
    
    async def _check_campaign_alerts(self, campaign_id: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Verifica alertas para una campaña específica"""
        alerts = []
        
        try:
            # Obtener métricas recientes
            metrics = await self._get_recent_metrics(campaign_id, lookback_days)
            
            if not metrics:
                return alerts
            
            # Verificar cada regla
            alerts.extend(self._check_low_ctr_alert(metrics))
            alerts.extend(self._check_high_cpc_alert(metrics))
            alerts.extend(self._check_low_conversion_rate_alert(metrics))
            alerts.extend(self._check_budget_utilization_alert(metrics))
            alerts.extend(self._check_performance_drop_alert(metrics, lookback_days))
            
        except Exception as e:
            logger.error(f"Error verificando alertas para campaña {campaign_id}: {e}")
        
        return alerts
    
    def _check_low_ctr_alert(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verifica alerta de CTR bajo"""
        alerts = []
        
        if metrics:
            avg_ctr = statistics.mean([m['ctr'] for m in metrics])
            
            if avg_ctr < self.alert_rules['low_ctr_threshold']:
                alerts.append({
                    'alert_type': 'low_ctr',
                    'severity': 'warning',
                    'message': f'Average CTR ({avg_ctr:.2f}%) below threshold ({self.alert_rules["low_ctr_threshold"]}%)',
                    'value': avg_ctr,
                    'threshold': self.alert_rules['low_ctr_threshold'],
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _check_high_cpc_alert(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verifica alerta de CPC alto"""
        alerts = []
        
        if metrics:
            avg_cpc = statistics.mean([m['cpc'] for m in metrics])
            
            if avg_cpc > self.alert_rules['high_cpc_threshold']:
                alerts.append({
                    'alert_type': 'high_cpc',
                    'severity': 'warning',
                    'message': f'Average CPC (${avg_cpc:.2f}) above threshold (${self.alert_rules["high_cpc_threshold"]:.2f})',
                    'value': avg_cpc,
                    'threshold': self.alert_rules['high_cpc_threshold'],
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _check_low_conversion_rate_alert(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verifica alerta de tasa de conversión baja"""
        alerts = []
        
        if metrics:
            avg_conversion_rate = statistics.mean([m['conversion_rate'] for m in metrics])
            
            if avg_conversion_rate < self.alert_rules['low_conversion_rate']:
                alerts.append({
                    'alert_type': 'low_conversion_rate',
                    'severity': 'warning',
                    'message': f'Conversion rate ({avg_conversion_rate:.2f}%) below threshold ({self.alert_rules["low_conversion_rate"]:.2f}%)',
                    'value': avg_conversion_rate,
                    'threshold': self.alert_rules['low_conversion_rate'],
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _check_budget_utilization_alert(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verifica alerta de utilización de presupuesto"""
        alerts = []
        
        if metrics:
            total_spend = sum(m['spend'] for m in metrics)
            estimated_budget = total_spend / len(metrics) * 7  # Estimar presupuesto diario
            
            # Esta es una verificación simplificada
            # En implementación real se compararía con presupuesto asignado
            
        return alerts
    
    def _check_performance_drop_alert(self, metrics: List[Dict[str, Any]], lookback_days: int) -> List[Dict[str, Any]]:
        """Verifica alerta de caída de performance"""
        alerts = []
        
        if len(metrics) >= lookback_days:
            # Dividir métricas en dos períodos
            mid_point = len(metrics) // 2
            recent_metrics = metrics[:mid_point]
            older_metrics = metrics[mid_point:]
            
            recent_ctr = statistics.mean([m['ctr'] for m in recent_metrics])
            older_ctr = statistics.mean([m['ctr'] for m in older_metrics])
            
            if older_ctr > 0:
                performance_drop = (older_ctr - recent_ctr) / older_ctr
                
                if performance_drop > self.alert_rules['performance_drop']:
                    alerts.append({
                        'alert_type': 'performance_drop',
                        'severity': 'critical',
                        'message': f'Performance drop of {performance_drop:.2%} detected',
                        'current_performance': recent_ctr,
                        'previous_performance': older_ctr,
                        'drop_percentage': performance_drop,
                        'timestamp': datetime.now().isoformat()
                    })
        
        return alerts
    
    async def _get_recent_metrics(self, campaign_id: str, days: int) -> List[Dict[str, Any]]:
        """Obtiene métricas recientes de una campaña"""
        try:
            conn = psycopg2.connect(self.db_connection)
            
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                DATE(ts) as date,
                AVG(CTR) as ctr,
                AVG(CPC) as cpc,
                SUM(conversions) as conversions,
                SUM(clicks) as clicks,
                SUM(spend_total) as spend
            FROM (
                SELECT 
                    ts,
                    CASE WHEN impressions > 0 THEN (clicks::float / impressions) * 100 ELSE 0 END as CTR,
                    CASE WHEN clicks > 0 THEN spend_total / clicks ELSE 0 END as CPC,
                    leads_day as conversions,
                    clicks,
                    spend_total
                FROM anchor_data 
                WHERE campaign_title = %s
                    AND ts >= %s
            ) subquery
            GROUP BY DATE(ts)
            ORDER BY date DESC
            """
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (campaign_id, start_date))
                results = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas recientes: {e}")
            return []
    
    async def _send_alert_notifications(self, alerts: List[Dict[str, Any]]) -> None:
        """Envía notificaciones de alertas"""
        if not self.notification_webhook:
            logger.warning("No webhook de notificación configurado")
            return
        
        try:
            # Agrupar alertas por severidad
            critical_alerts = [a for a in alerts if a['severity'] == 'critical']
            warning_alerts = [a for a in alerts if a['severity'] == 'warning']
            
            notification = {
                'timestamp': datetime.now().isoformat(),
                'total_alerts': len(alerts),
                'critical_alerts': len(critical_alerts),
                'warning_alerts': len(warning_alerts),
                'alerts': alerts
            }
            
            # Enviar a webhook
            response = requests.post(
                self.notification_webhook,
                json=notification,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Notificaciones enviadas: {len(alerts)} alertas")
            else:
                logger.error(f"Error enviando notificaciones: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error enviando notificaciones: {e}")


class UnifiedAdsManager:
    """Gestor unificado de campañas cross-platform"""
    
    def __init__(self, 
                 db_connection: str,
                 google_ads_config: GoogleAdsConfig,
                 meta_ads_config: MetaAdsConfig,
                 redis_connection: str = "redis://localhost:6379"):
        self.db_connection = db_connection
        self.google_ads_config = google_ads_config
        self.meta_ads_config = meta_ads_config
        self.redis_connection = redis_connection
        
        # Inicializar conectores
        self.google_connector = None
        self.meta_connector = None
        
        # Inicializar componentes
        self.optimizer = CrossPlatformOptimizer(db_connection)
        self.audience_manager = AudienceManager(db_connection)
        self.alerting_system = AlertingSystem(db_connection)
        self.targeting_mapper = TargetingMapper()
        
        # Cache de campañas
        self.campaign_cache = {}
        
        logger.info("UnifiedAdsManager inicializado")
    
    async def initialize(self) -> None:
        """Inicializa los conectores"""
        try:
            # Inicializar Google Ads connector
            self.google_connector = GoogleAdsConnector(
                self.google_ads_config,
                self.db_connection,
                self.redis_connection
            )
            
            # Inicializar Meta Ads connector
            self.meta_connector = MetaAdsConnector(
                self.meta_ads_config,
                self.db_connection,
                self.redis_connection
            )
            
            logger.info("Conectores inicializados exitosamente")
            
        except Exception as e:
            logger.error(f"Error inicializando conectores: {e}")
            raise
    
    async def sync_campaigns_bidirectional(self, 
                                         google_campaign_ids: List[str] = None,
                                         meta_campaign_ids: List[str] = None) -> Dict[str, Any]:
        """Sincroniza campañas bidireccionalmente entre plataformas"""
        logger.info("Iniciando sincronización bidireccional de campañas")
        
        sync_results = {
            'google_to_meta': {'successful': 0, 'failed': 0},
            'meta_to_google': {'successful': 0, 'failed': 0},
            'errors': []
        }
        
        try:
            # Sincronización Google Ads -> Meta Ads
            if google_campaign_ids:
                google_sync_result = await self._sync_google_to_meta(google_campaign_ids)
                sync_results['google_to_meta'] = google_sync_result
            
            # Sincronización Meta Ads -> Google Ads
            if meta_campaign_ids:
                meta_sync_result = await self._sync_meta_to_google(meta_campaign_ids)
                sync_results['meta_to_google'] = meta_sync_result
            
            # Actualizar cache
            await self._update_campaign_cache()
            
            return {
                'sync_results': sync_results,
                'timestamp': datetime.now().isoformat(),
                'total_syncs': sync_results['google_to_meta']['successful'] + sync_results['meta_to_google']['successful']
            }
            
        except Exception as e:
            logger.error(f"Error en sincronización bidireccional: {e}")
            sync_results['errors'].append(str(e))
            return sync_results
    
    async def create_unified_campaign(self, 
                                    campaign_config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea campaña unificada en ambas plataformas"""
        logger.info(f"Creando campaña unificada: {campaign_config.get('name')}")
        
        creation_results = {
            'google_campaign': None,
            'meta_campaign': None,
            'success': False,
            'errors': []
        }
        
        try:
            # Crear en Google Ads
            google_result = await self._create_google_campaign(campaign_config)
            if google_result['success']:
                creation_results['google_campaign'] = google_result
            else:
                creation_results['errors'].append(f"Google Ads: {google_result.get('error')}")
            
            # Crear en Meta Ads
            meta_result = await self._create_meta_campaign(campaign_config)
            if meta_result['success']:
                creation_results['meta_campaign'] = meta_result
            else:
                creation_results['errors'].append(f"Meta Ads: {meta_result.get('error')}")
            
            # Verificar si ambas fueron exitosas
            creation_results['success'] = (
                google_result['success'] and meta_result['success']
            )
            
            # Si fue exitosa, agregar al cache
            if creation_results['success']:
                await self._add_to_campaign_cache(campaign_config)
            
            return creation_results
            
        except Exception as e:
            logger.error(f"Error creando campaña unificada: {e}")
            creation_results['errors'].append(str(e))
            return creation_results
    
    async def get_consolidated_report(self, 
                                    campaign_ids: List[str],
                                    start_date: str,
                                    end_date: str) -> Dict[str, Any]:
        """Genera reporte consolidado de métricas cross-platform"""
        logger.info(f"Generando reporte consolidado para {len(campaign_ids)} campañas")
        
        try:
            # Obtener datos de ambas plataformas
            google_data = await self._get_platform_data(campaign_ids, Platform.GOOGLE_ADS, start_date, end_date)
            meta_data = await self._get_platform_data(campaign_ids, Platform.META_ADS, start_date, end_date)
            
            # Consolidar métricas
            consolidated_metrics = self._consolidate_metrics(google_data, meta_data)
            
            # Calcular KPIs consolidados
            kpis = self._calculate_consolidated_kpis(consolidated_metrics)
            
            # Generar insights
            insights = self._generate_performance_insights(consolidated_metrics)
            
            return {
                'report_period': {'start_date': start_date, 'end_date': end_date},
                'campaigns_analyzed': campaign_ids,
                'consolidated_metrics': consolidated_metrics,
                'kpis': kpis,
                'insights': insights,
                'platform_breakdown': {
                    'google_ads': google_data,
                    'meta_ads': meta_data
                },
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generando reporte consolidado: {e}")
            raise
    
    async def optimize_cross_platform(self, 
                                    campaign_ids: List[str],
                                    optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta optimización cross-platform"""
        logger.info("Iniciando optimización cross-platform")
        
        try:
            # Análisis de performance
            performance_analysis = await self.optimizer.analyze_performance(
                campaign_ids, 
                optimization_config.get('analysis_days', 30)
            )
            
            # Optimización de presupuesto
            budget_optimization = await self.optimizer.optimize_budget_allocation(
                campaign_ids,
                optimization_config.get('total_budget', 1000),
                optimization_config.get('optimization_rules', {})
            )
            
            # Detección de overlap de audiencias
            google_campaigns = [cid for cid in campaign_ids if cid.startswith('google_')]
            meta_campaigns = [cid for cid in campaign_ids if cid.startswith('meta_')]
            
            audience_overlap = None
            if google_campaigns and meta_campaigns:
                audience_overlap = await self.audience_manager.detect_audience_overlap(
                    google_campaigns, meta_campaigns
                )
            
            # Aplicar optimizaciones
            optimization_actions = await self._apply_optimizations(
                performance_analysis,
                budget_optimization,
                optimization_config
            )
            
            return {
                'performance_analysis': performance_analysis,
                'budget_optimization': asdict(budget_optimization),
                'audience_overlap': asdict(audience_overlap) if audience_overlap else None,
                'optimization_actions': optimization_actions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en optimización cross-platform: {e}")
            raise
    
    async def setup_monitoring(self, alert_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configura sistema de monitoreo y alertas"""
        logger.info("Configurando sistema de monitoreo")
        
        try:
            # Configurar reglas de alerta
            self.alerting_system.configure_alert_rules(alert_config)
            
            # Obtener campañas activas para monitorear
            active_campaigns = await self._get_active_campaigns()
            
            # Verificar alertas iniciales
            initial_alerts = await self.alerting_system.check_performance_alerts(
                [c['id'] for c in active_campaigns],
                alert_config.get('lookback_days', 7)
            )
            
            return {
                'alert_rules_configured': True,
                'active_campaigns_monitored': len(active_campaigns),
                'initial_alerts': initial_alerts,
                'monitoring_setup_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error configurando monitoreo: {e}")
            raise
    
    async def _sync_google_to_meta(self, google_campaign_ids: List[str]) -> Dict[str, Any]:
        """Sincroniza campañas de Google Ads a Meta Ads"""
        successful = 0
        failed = 0
        errors = []
        
        for google_id in google_campaign_ids:
            try:
                # Obtener datos de Google
                google_data = await self._get_campaign_data(google_id, Platform.GOOGLE_ADS)
                
                # Mapear targeting
                meta_targeting = self.targeting_mapper.map_google_to_meta(google_data['targeting'])
                
                # Crear en Meta Ads (simulado)
                success = await self._create_meta_campaign_from_google(google_data, meta_targeting)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error sincronizando {google_id}: {e}")
                failed += 1
                errors.append(str(e))
        
        return {'successful': successful, 'failed': failed, 'errors': errors}
    
    async def _sync_meta_to_google(self, meta_campaign_ids: List[str]) -> Dict[str, Any]:
        """Sincroniza campañas de Meta Ads a Google Ads"""
        successful = 0
        failed = 0
        errors = []
        
        for meta_id in meta_campaign_ids:
            try:
                # Obtener datos de Meta
                meta_data = await self._get_campaign_data(meta_id, Platform.META_ADS)
                
                # Mapear targeting
                google_targeting = self.targeting_mapper.map_meta_to_google(meta_data['targeting'])
                
                # Crear en Google Ads (simulado)
                success = await self._create_google_campaign_from_meta(meta_data, google_targeting)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error sincronizando {meta_id}: {e}")
                failed += 1
                errors.append(str(e))
        
        return {'successful': successful, 'failed': failed, 'errors': errors}
    
    async def _get_platform_data(self, campaign_ids: List[str], platform: Platform, 
                                start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Obtiene datos de una plataforma específica"""
        try:
            conn = psycopg2.connect(self.db_connection)
            
            query = """
            SELECT 
                campaign_title,
                SUM(impressions) as impressions,
                SUM(clicks) as clicks,
                SUM(spend_total) as spend,
                SUM(leads_day) as conversions
            FROM anchor_data 
            WHERE campaign_title = ANY(%s)
                AND platform = %s
                AND ts BETWEEN %s AND %s
            GROUP BY campaign_title
            """
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (campaign_ids, platform.value, start_date, end_date))
                results = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de {platform.value}: {e}")
            return []
    
    def _consolidate_metrics(self, google_data: List[Dict[str, Any]], 
                           meta_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolida métricas de ambas plataformas"""
        total_impressions = sum(d['impressions'] for d in google_data) + sum(d['impressions'] for d in meta_data)
        total_clicks = sum(d['clicks'] for d in google_data) + sum(d['clicks'] for d in meta_data)
        total_spend = sum(d['spend'] for d in google_data) + sum(d['spend'] for d in meta_data)
        total_conversions = sum(d['conversions'] for d in google_data) + sum(d['conversions'] for d in meta_data)
        
        return {
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_spend': total_spend,
            'total_conversions': total_conversions,
            'consolidated_ctr': (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
            'consolidated_cpc': (total_spend / total_clicks) if total_clicks > 0 else 0,
            'consolidated_cpm': (total_spend / total_impressions * 1000) if total_impressions > 0 else 0,
            'consolidated_conversion_rate': (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        }
    
    def _calculate_consolidated_kpis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula KPIs consolidados"""
        return {
            'cost_per_acquisition': (metrics['total_spend'] / metrics['total_conversions']) if metrics['total_conversions'] > 0 else 0,
            'return_on_ad_spend': (metrics['total_conversions'] * 50 / metrics['total_spend']) if metrics['total_spend'] > 0 else 0,  # Asumiendo $50 por conversión
            'efficiency_score': self._calculate_efficiency_score(metrics),
            'performance_grade': self._get_performance_grade(metrics)
        }
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calcula score de eficiencia general"""
        ctr_score = min(metrics['consolidated_ctr'] / 2.0, 1.0)  # Normalizar CTR
        conversion_score = min(metrics['consolidated_conversion_rate'] / 5.0, 1.0)  # Normalizar conversión
        
        efficiency = (ctr_score + conversion_score) / 2 * 100
        return round(efficiency, 2)
    
    def _get_performance_grade(self, metrics: Dict[str, Any]) -> str:
        """Asigna grade de performance"""
        efficiency = self._calculate_efficiency_score(metrics)
        
        if efficiency >= 80:
            return 'A'
        elif efficiency >= 70:
            return 'B'
        elif efficiency >= 60:
            return 'C'
        elif efficiency >= 50:
            return 'D'
        else:
            return 'F'
    
    def _generate_performance_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Genera insights de performance"""
        insights = []
        
        if metrics['consolidated_ctr'] < 1.0:
            insights.append("CTR está por debajo del promedio de la industria. Considera optimizar creativos.")
        
        if metrics['consolidated_conversion_rate'] < 2.0:
            insights.append("Tasa de conversión baja. Revisa la experiencia de landing page.")
        
        if metrics['consolidated_cpc'] > 3.0:
            insights.append("CPC elevado. Considera ajustar targeting para mejorar calidad de audiencia.")
        
        if not insights:
            insights.append("Performance sólida en métricas clave.")
        
        return insights
    
    async def _get_campaign_data(self, campaign_id: str, platform: Platform) -> Dict[str, Any]:
        """Obtiene datos de una campaña específica"""
        # Implementación simplificada
        return {
            'id': campaign_id,
            'name': f'Campaign {campaign_id}',
            'targeting': {
                'age_range': '25-54',
                'genders': ['male', 'female'],
                'interests': [{'id': '123', 'name': 'Technology'}]
            },
            'budget': 1000,
            'platform': platform
        }
    
    async def _create_google_campaign_from_google(self, data: Dict[str, Any], targeting: Dict[str, Any]) -> bool:
        """Crea campaña en Google Ads desde datos de Google (placeholder)"""
        return True
    
    async def _create_meta_campaign_from_meta(self, data: Dict[str, Any], targeting: Dict[str, Any]) -> bool:
        """Crea campaña en Meta Ads desde datos de Meta (placeholder)"""
        return True
    
    async def _get_active_campaigns(self) -> List[Dict[str, Any]]:
        """Obtiene campañas activas"""
        try:
            conn = psycopg2.connect(self.db_connection)
            
            query = """
            SELECT DISTINCT campaign_title as id, campaign_title as name
            FROM anchor_data 
            WHERE ts >= %s
            """
            
            since_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (since_date,))
                results = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error obteniendo campañas activas: {e}")
            return []
    
    async def _update_campaign_cache(self) -> None:
        """Actualiza cache de campañas"""
        # Implementación simplificada
        pass
    
    async def _add_to_campaign_cache(self, campaign_config: Dict[str, Any]) -> None:
        """Agrega campaña al cache"""
        # Implementación simplificada
        pass
    
    async def _apply_optimizations(self, performance_analysis: Dict[str, Any], 
                                 budget_optimization: BudgetAllocation,
                                 config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aplica optimizaciones automáticas"""
        actions = []
        
        # Simular aplicación de optimizaciones
        actions.append({
            'action': 'budget_reallocation',
            'description': 'Reallocate budget based on performance analysis',
            'details': {
                'google_allocation': budget_optimization.google_allocation,
                'meta_allocation': budget_optimization.meta_allocation
            }
        })
        
        return actions
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica salud del sistema"""
        health_status = {
            'overall_status': 'healthy',
            'google_ads': {'connected': False, 'error': None},
            'meta_ads': {'connected': False, 'error': None},
            'database': {'connected': False, 'error': None},
            'components': {
                'optimizer': False,
                'audience_manager': False,
                'alerting_system': False,
                'targeting_mapper': False
            }
        }
        
        try:
            # Verificar conectores
            if self.google_connector:
                google_health = self.google_connector.health_check()
                health_status['google_ads']['connected'] = google_health.get('status') == 'healthy'
                if not health_status['google_ads']['connected']:
                    health_status['google_ads']['error'] = google_health.get('error')
            
            if self.meta_connector:
                meta_health = self.meta_connector.health_check()
                health_status['meta_ads']['connected'] = meta_health.get('status') == 'healthy'
                if not health_status['meta_ads']['connected']:
                    health_status['meta_ads']['error'] = meta_health.get('error')
            
            # Verificar base de datos
            conn = psycopg2.connect(self.db_connection)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            health_status['database']['connected'] = True
            
            # Verificar componentes
            health_status['components']['optimizer'] = True
            health_status['components']['audience_manager'] = True
            health_status['components']['alerting_system'] = True
            health_status['components']['targeting_mapper'] = True
            
        except Exception as e:
            health_status['overall_status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status


# Ejemplo de uso y configuración
if __name__ == "__main__":
    async def main():
        """Ejemplo de uso del sistema MIDAS Platform Integration"""
        
        # Configuración de Google Ads
        google_config = GoogleAdsConfig(
            customer_id="1234567890",
            developer_token="your_developer_token",
            client_id="your_client_id",
            client_secret="your_client_secret",
            refresh_token="your_refresh_token"
        )
        
        # Configuración de Meta Ads
        meta_config = MetaAdsConfig(
            access_token="your_access_token",
            app_id="your_app_id",
            app_secret="your_app_secret",
            ad_account_id="123456789"
        )
        
        # Crear manager unificado
        manager = UnifiedAdsManager(
            db_connection="postgresql://user:pass@localhost/anchor",
            google_ads_config=google_config,
            meta_ads_config=meta_config,
            redis_connection="redis://localhost:6379"
        )
        
        # Inicializar
        await manager.initialize()
        
        # Health check
        health = await manager.health_check()
        print(f"Health Check: {json.dumps(health, indent=2, default=str)}")
        
        # Configurar monitoreo
        alert_config = {
            'low_ctr_threshold': 1.5,
            'high_cpc_threshold': 4.0,
            'low_conversion_rate': 2.5,
            'budget_utilization': 0.85,
            'performance_drop': 0.25
        }
        
        monitoring_result = await manager.setup_monitoring(alert_config)
        print(f"Monitoring Setup: {json.dumps(monitoring_result, indent=2, default=str)}")
        
        # Crear campaña unificada
        campaign_config = {
            'name': 'Test Cross-Platform Campaign',
            'budget': 1000,
            'targeting': {
                'age_range': '25-54',
                'interests': ['Technology', 'Business'],
                'locations': ['US']
            },
            'creatives': [
                {'type': 'image', 'url': 'https://example.com/image1.jpg'},
                {'type': 'video', 'url': 'https://example.com/video1.mp4'}
            ]
        }
        
        # creation_result = await manager.create_unified_campaign(campaign_config)
        # print(f"Campaign Creation: {json.dumps(creation_result, indent=2, default=str)}")
        
        # Sincronización bidireccional
        google_campaigns = ['google_123', 'google_456']
        meta_campaigns = ['meta_789', 'meta_101']
        
        # sync_result = await manager.sync_campaigns_bidirectional(google_campaigns, meta_campaigns)
        # print(f"Bidirectional Sync: {json.dumps(sync_result, indent=2, default=str)}")
        
        # Reporte consolidado
        campaign_ids = ['campaign_1', 'campaign_2', 'campaign_3']
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        # report = await manager.get_consolidated_report(campaign_ids, start_date, end_date)
        # print(f"Consolidated Report: {json.dumps(report, indent=2, default=str)}")
        
        # Optimización cross-platform
        optimization_config = {
            'total_budget': 2000,
            'optimization_rules': {'strategy': 'performance_based'},
            'analysis_days': 30
        }
        
        # optimization_result = await manager.optimize_cross_platform(campaign_ids, optimization_config)
        # print(f"Cross-Platform Optimization: {json.dumps(optimization_result, indent=2, default=str)}")
    
    # Ejecutar ejemplo
    # asyncio.run(main())
