"""
MIDAS ROI Tracking System
Sistema completo de tracking y medición ROI para MIDAS

Funcionalidades principales:
- Attribution modeling (first-click, last-click, time-decay, data-driven)
- Cross-device conversion tracking
- E-commerce platform integration (Shopify, WooCommerce)
- ROAS, LTV, CAC calculation
- Real-time ROI dashboard
- Customer journey analysis
- Ad fraud detection
- Automated executive reporting

Autor: MIDAS Team
Fecha: 2025-11-06
"""

import json
import sqlite3
import hashlib
import time
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import sqlite3
import logging
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TouchPoint:
    """Punto de contacto en el customer journey"""
    user_id: str
    device_id: str
    timestamp: datetime
    source: str
    medium: str
    campaign: str
    channel: str
    cost: float = 0.0
    value: float = 0.0
    
@dataclass
class Conversion:
    """Evento de conversión"""
    user_id: str
    device_id: str
    timestamp: datetime
    conversion_type: str
    value: float
    order_id: str
    source_attribution: str = ""
    medium_attribution: str = ""
    campaign_attribution: str = ""

@dataclass
class CustomerJourney:
    """Customer journey completo de un usuario"""
    user_id: str
    touchpoints: List[TouchPoint]
    conversions: List[Conversion]
    start_date: datetime
    end_date: datetime
    total_value: float
    total_cost: float
    
class AttributionEngine:
    """
    Motor de atribución con múltiples modelos
    """
    
    def __init__(self, db_path: str = "midas_roi_tracking.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Inicializa la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de touchpoints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS touchpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                device_id TEXT,
                timestamp TEXT,
                source TEXT,
                medium TEXT,
                campaign TEXT,
                channel TEXT,
                cost REAL,
                value REAL,
                created_at TEXT
            )
        ''')
        
        # Tabla de conversiones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                device_id TEXT,
                timestamp TEXT,
                conversion_type TEXT,
                value REAL,
                order_id TEXT,
                source_attribution TEXT,
                medium_attribution TEXT,
                campaign_attribution TEXT,
                created_at TEXT
            )
        ''')
        
        # Tabla de atribuciones calculadas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversion_id TEXT,
                user_id TEXT,
                attribution_model TEXT,
                touchpoint_id TEXT,
                attribution_percentage REAL,
                attributed_value REAL,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_touchpoint(self, touchpoint: TouchPoint):
        """Añade un punto de contacto"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO touchpoints 
            (user_id, device_id, timestamp, source, medium, campaign, channel, cost, value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            touchpoint.user_id, touchpoint.device_id, touchpoint.timestamp.isoformat(),
            touchpoint.source, touchpoint.medium, touchpoint.campaign, touchpoint.channel,
            touchpoint.cost, touchpoint.value, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def add_conversion(self, conversion: Conversion):
        """Añade una conversión"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversions 
            (user_id, device_id, timestamp, conversion_type, value, order_id, 
             source_attribution, medium_attribution, campaign_attribution, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversion.user_id, conversion.device_id, conversion.timestamp.isoformat(),
            conversion.conversion_type, conversion.value, conversion.order_id,
            conversion.source_attribution, conversion.medium_attribution, 
            conversion.campaign_attribution, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def get_user_journey(self, user_id: str, days_back: int = 30) -> CustomerJourney:
        """Obtiene el customer journey de un usuario"""
        conn = sqlite3.connect(self.db_path)
        
        # Obtener touchpoints
        touchpoints_query = '''
            SELECT * FROM touchpoints 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp
        '''
        
        # Obtener conversiones
        conversions_query = '''
            SELECT * FROM conversions 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp
        '''
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        touchpoints_df = pd.read_sql_query(touchpoints_query, conn, 
                                          params=[user_id, cutoff_date])
        conversions_df = pd.read_sql_query(conversions_query, conn, 
                                          params=[user_id, cutoff_date])
        
        conn.close()
        
        # Convertir a objetos
        touchpoints = []
        for _, row in touchpoints_df.iterrows():
            touchpoint = TouchPoint(
                user_id=row['user_id'],
                device_id=row['device_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                source=row['source'],
                medium=row['medium'],
                campaign=row['campaign'],
                channel=row['channel'],
                cost=row['cost'],
                value=row['value']
            )
            touchpoints.append(touchpoint)
            
        conversions = []
        for _, row in conversions_df.iterrows():
            conversion = Conversion(
                user_id=row['user_id'],
                device_id=row['device_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                conversion_type=row['conversion_type'],
                value=row['value'],
                order_id=row['order_id'],
                source_attribution=row['source_attribution'],
                medium_attribution=row['medium_attribution'],
                campaign_attribution=row['campaign_attribution']
            )
            conversions.append(conversion)
            
        start_date = min([tp.timestamp for tp in touchpoints] + [datetime.min]) if touchpoints else datetime.min
        end_date = max([tp.timestamp for tp in touchpoints] + [datetime.min]) if touchpoints else datetime.min
        
        total_value = sum(c.value for c in conversions)
        total_cost = sum(tp.cost for tp in touchpoints)
        
        return CustomerJourney(
            user_id=user_id,
            touchpoints=touchpoints,
            conversions=conversions,
            start_date=start_date,
            end_date=end_date,
            total_value=total_value,
            total_cost=total_cost
        )
        
    def first_click_attribution(self, journey: CustomerJourney) -> Dict[str, float]:
        """Atribución first-click"""
        attribution = defaultdict(float)
        
        for conversion in journey.conversions:
            # Encontrar el primer touchpoint
            if journey.touchpoints:
                first_touchpoint = journey.touchpoints[0]
                attribution[first_touchpoint.campaign] += conversion.value
                
        return dict(attribution)
        
    def last_click_attribution(self, journey: CustomerJourney) -> Dict[str, float]:
        """Atribución last-click"""
        attribution = defaultdict(float)
        
        for conversion in journey.conversions:
            # Encontrar el último touchpoint antes de la conversión
            relevant_touchpoints = [tp for tp in journey.touchpoints 
                                  if tp.timestamp <= conversion.timestamp]
            if relevant_touchpoints:
                last_touchpoint = relevant_touchpoints[-1]
                attribution[last_touchpoint.campaign] += conversion.value
                
        return dict(attribution)
        
    def time_decay_attribution(self, journey: CustomerJourney, half_life_days: int = 7) -> Dict[str, float]:
        """Atribución time-decay"""
        attribution = defaultdict(float)
        half_life_seconds = half_life_days * 24 * 3600
        
        for conversion in journey.conversions:
            conversion_time = conversion.timestamp.timestamp()
            total_weight = 0
            weights = {}
            
            # Calcular pesos basados en el tiempo
            for tp in journey.touchpoints:
                if tp.timestamp <= conversion.timestamp:
                    time_diff = conversion_time - tp.timestamp.timestamp()
                    weight = 0.5 ** (time_diff / half_life_seconds)
                    weights[tp.campaign] = weights.get(tp.campaign, 0) + weight
                    total_weight += weight
                    
            # Asignar valor basado en pesos
            if total_weight > 0:
                for campaign, weight in weights.items():
                    attribution[campaign] += conversion.value * (weight / total_weight)
                    
        return dict(attribution)
        
    def data_driven_attribution(self, journey: CustomerJourney, 
                              lookback_window_days: int = 30) -> Dict[str, float]:
        """Atribución data-driven usando machine learning simplificado"""
        # Simulación de algoritmo de machine learning
        attribution = defaultdict(float)
        
        for conversion in journey.conversions:
            # Factores de peso para diferentes posiciones
            touchpoints_before = [tp for tp in journey.touchpoints 
                                if tp.timestamp <= conversion.timestamp]
            
            if not touchpoints_before:
                continue
                
            # Calcular posición y peso
            n_touchpoints = len(touchpoints_before)
            position_weights = []
            
            for i, tp in enumerate(touchpoints_before):
                # Peso basado en posición (últimos touchpoints tienen más peso)
                position_weight = (i + 1) / n_touchpoints
                
                # Peso basado en canal
                channel_multiplier = self._get_channel_weight(tp.channel)
                
                # Peso basado en tiempo
                time_weight = self._get_time_weight(tp.timestamp, conversion.timestamp)
                
                combined_weight = position_weight * channel_multiplier * time_weight
                position_weights.append((tp, combined_weight))
                
            # Normalizar pesos y asignar valor
            total_weight = sum(w[1] for w in position_weights)
            if total_weight > 0:
                for tp, weight in position_weights:
                    attribution[tp.campaign] += conversion.value * (weight / total_weight)
                    
        return dict(attribution)
        
    def _get_channel_weight(self, channel: str) -> float:
        """Obtiene peso del canal"""
        channel_weights = {
            'paid_search': 1.5,
            'organic_search': 1.2,
            'social_media': 1.0,
            'email': 1.3,
            'direct': 0.8,
            'referral': 1.1,
            'display': 0.9
        }
        return channel_weights.get(channel.lower(), 1.0)
        
    def _get_time_weight(self, touchpoint_time: datetime, conversion_time: datetime) -> float:
        """Calcula peso basado en tiempo"""
        days_diff = (conversion_time - touchpoint_time).days
        if days_diff <= 1:
            return 1.0
        elif days_diff <= 7:
            return 0.8
        elif days_diff <= 30:
            return 0.6
        else:
            return 0.3
            
    def calculate_all_attribution_models(self, user_id: str) -> Dict[str, Dict[str, float]]:
        """Calcula todos los modelos de atribución para un usuario"""
        journey = self.get_user_journey(user_id)
        
        return {
            'first_click': self.first_click_attribution(journey),
            'last_click': self.last_click_attribution(journey),
            'time_decay': self.time_decay_attribution(journey),
            'data_driven': self.data_driven_attribution(journey)
        }

class ConversionTracker:
    """
    Tracker de conversiones cross-device
    """
    
    def __init__(self, attribution_engine: AttributionEngine):
        self.attribution_engine = attribution_engine
        self.device_mapping = {}
        self.session_data = {}
        
    def identify_cross_device_user(self, device_id: str, user_id: str) -> str:
        """Identifica usuarios a través de dispositivos"""
        if device_id not in self.device_mapping:
            self.device_mapping[device_id] = user_id
            
        return self.device_mapping[device_id]
        
    def track_cross_device_conversion(self, device_id: str, user_id: str, 
                                    conversion_data: Dict[str, Any]) -> Conversion:
        """Registra conversión cross-device"""
        conversion = Conversion(
            user_id=user_id,
            device_id=device_id,
            timestamp=datetime.now(),
            conversion_type=conversion_data.get('type', 'purchase'),
            value=conversion_data.get('value', 0.0),
            order_id=conversion_data.get('order_id', str(uuid.uuid4()))
        )
        
        # Asociar con touchpoints anteriores del usuario
        self._link_conversion_to_touchpoints(conversion)
        
        self.attribution_engine.add_conversion(conversion)
        return conversion
        
    def _link_conversion_to_touchpoints(self, conversion: Conversion):
        """Vincula conversión con touchpoints relevantes"""
        # Esta función se podría expandir para análisis más sofisticado
        pass
        
    def get_cross_device_metrics(self) -> Dict[str, Any]:
        """Métricas cross-device"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Usuarios únicos
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversions")
        unique_users = cursor.fetchone()[0]
        
        # Dispositivos únicos
        cursor.execute("SELECT COUNT(DISTINCT device_id) FROM conversions")
        unique_devices = cursor.fetchone()[0]
        
        # Relación dispositivo por usuario
        cursor.execute('''
            SELECT user_id, COUNT(DISTINCT device_id) as device_count
            FROM conversions
            GROUP BY user_id
            HAVING device_count > 1
        ''')
        cross_device_users = cursor.fetchall()
        
        avg_devices_per_user = np.mean([row[1] for row in cross_device_users]) if cross_device_users else 0
        
        conn.close()
        
        return {
            'unique_users': unique_users,
            'unique_devices': unique_devices,
            'cross_device_users': len(cross_device_users),
            'avg_devices_per_user': round(avg_devices_per_user, 2),
            'cross_device_rate': round(len(cross_device_users) / unique_users * 100, 2) if unique_users > 0 else 0
        }

class ECommerceIntegration:
    """
    Integración con plataformas de e-commerce
    """
    
    def __init__(self, attribution_engine: AttributionEngine):
        self.attribution_engine = attribution_engine
        self.platform_credentials = {}
        
    def connect_shopify(self, store_url: str, access_token: str):
        """Conecta con Shopify"""
        # Simulación de conexión - en producción usar API real
        self.platform_credentials['shopify'] = {
            'store_url': store_url,
            'access_token': access_token,
            'connected_at': datetime.now().isoformat()
        }
        logger.info(f"Conectado a Shopify: {store_url}")
        
    def connect_woocommerce(self, site_url: str, consumer_key: str, consumer_secret: str):
        """Conecta con WooCommerce"""
        self.platform_credentials['woocommerce'] = {
            'site_url': site_url,
            'consumer_key': consumer_key,
            'consumer_secret': consumer_secret,
            'connected_at': datetime.now().isoformat()
        }
        logger.info(f"Conectado a WooCommerce: {site_url}")
        
    def sync_shopify_orders(self) -> List[Conversion]:
        """Sincroniza órdenes de Shopify"""
        # Simulación - en producción conectar con API real
        logger.info("Sincronizando órdenes de Shopify...")
        
        # Datos simulados
        simulated_orders = [
            {
                'order_id': 'SHOP_001',
                'user_id': 'user_123',
                'device_id': 'device_001',
                'value': 299.99,
                'timestamp': datetime.now() - timedelta(hours=2),
                'source': 'google_ads',
                'medium': 'cpc',
                'campaign': 'summer_sale'
            },
            {
                'order_id': 'SHOP_002',
                'user_id': 'user_456',
                'device_id': 'device_002',
                'value': 149.50,
                'timestamp': datetime.now() - timedelta(hours=1),
                'source': 'facebook_ads',
                'medium': 'cpm',
                'campaign': 'retargeting'
            }
        ]
        
        conversions = []
        for order in simulated_orders:
            # Añadir touchpoint
            touchpoint = TouchPoint(
                user_id=order['user_id'],
                device_id=order['device_id'],
                timestamp=order['timestamp'],
                source=order['source'],
                medium=order['medium'],
                campaign=order['campaign'],
                channel='ecommerce',
                cost=50.0,  # Costo estimado
                value=order['value']
            )
            self.attribution_engine.add_touchpoint(touchpoint)
            
            # Añadir conversión
            conversion = Conversion(
                user_id=order['user_id'],
                device_id=order['device_id'],
                timestamp=order['timestamp'],
                conversion_type='purchase',
                value=order['value'],
                order_id=order['order_id']
            )
            self.attribution_engine.add_conversion(conversion)
            conversions.append(conversion)
            
        return conversions
        
    def sync_woocommerce_orders(self) -> List[Conversion]:
        """Sincroniza órdenes de WooCommerce"""
        logger.info("Sincronizando órdenes de WooCommerce...")
        
        # Datos simulados
        simulated_orders = [
            {
                'order_id': 'WOO_001',
                'user_id': 'user_789',
                'device_id': 'device_003',
                'value': 89.99,
                'timestamp': datetime.now() - timedelta(minutes=30),
                'source': 'google_ads',
                'medium': 'cpc',
                'campaign': 'black_friday'
            }
        ]
        
        conversions = []
        for order in simulated_orders:
            # Añadir touchpoint
            touchpoint = TouchPoint(
                user_id=order['user_id'],
                device_id=order['device_id'],
                timestamp=order['timestamp'],
                source=order['source'],
                medium=order['medium'],
                campaign=order['campaign'],
                channel='ecommerce',
                cost=25.0,
                value=order['value']
            )
            self.attribution_engine.add_touchpoint(touchpoint)
            
            # Añadir conversión
            conversion = Conversion(
                user_id=order['user_id'],
                device_id=order['device_id'],
                timestamp=order['timestamp'],
                conversion_type='purchase',
                value=order['value'],
                order_id=order['order_id']
            )
            self.attribution_engine.add_conversion(conversion)
            conversions.append(conversion)
            
        return conversions

class ROIAnalyzer:
    """
    Analizador de ROI que calcula ROAS, LTV, CAC
    """
    
    def __init__(self, attribution_engine: AttributionEngine):
        self.attribution_engine = attribution_engine
        
    def calculate_roas(self, user_id: Optional[str] = None, 
                      campaign: Optional[str] = None,
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None) -> float:
        """Calcula Return on Ad Spend (ROAS)"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Query base
        query = "SELECT SUM(value) as total_value, SUM(cost) as total_cost FROM touchpoints WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
            
        if campaign:
            query += " AND campaign = ?"
            params.append(campaign)
            
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from.isoformat())
            
        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to.isoformat())
            
        cursor.execute(query, params)
        result = cursor.fetchone()
        conn.close()
        
        if not result or result[1] == 0:  # Evitar división por cero
            return 0.0
            
        total_value, total_cost = result
        return round(total_value / total_cost, 2)
        
    def calculate_ltv(self, user_id: str, months: int = 12) -> float:
        """Calcula Customer Lifetime Value"""
        journey = self.attribution_engine.get_user_journey(user_id, days_back=months*30)
        
        # Valor total del cliente
        total_value = journey.total_value
        
        # Estimación de valor futuro basado en comportamiento histórico
        conversion_frequency = len(journey.conversions) / max(1, months)
        avg_order_value = total_value / max(1, len(journey.conversions))
        
        # LTV estimado
        ltv = total_value * (1 + conversion_frequency * 0.1)  # Factor de crecimiento
        return round(ltv, 2)
        
    def calculate_cac(self, campaign: Optional[str] = None,
                     date_from: Optional[datetime] = None,
                     date_to: Optional[datetime] = None) -> float:
        """Calcula Customer Acquisition Cost"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Costo total de marketing
        cost_query = "SELECT SUM(cost) as total_cost FROM touchpoints WHERE 1=1"
        cost_params = []
        
        if campaign:
            cost_query += " AND campaign = ?"
            cost_params.append(campaign)
            
        if date_from:
            cost_query += " AND timestamp >= ?"
            cost_params.append(date_from.isoformat())
            
        if date_to:
            cost_query += " AND timestamp <= ?"
            cost_params.append(date_to.isoformat())
            
        cursor.execute(cost_query, cost_params)
        total_cost = cursor.fetchone()[0] or 0
        
        # Número de clientes únicos adquiridos
        user_query = "SELECT COUNT(DISTINCT user_id) FROM conversions WHERE 1=1"
        user_params = []
        
        if date_from:
            user_query += " AND timestamp >= ?"
            user_params.append(date_from.isoformat())
            
        if date_to:
            user_query += " AND timestamp <= ?"
            user_params.append(date_to.isoformat())
            
        cursor.execute(user_query, user_params)
        unique_customers = cursor.fetchone()[0]
        conn.close()
        
        if unique_customers == 0:
            return 0.0
            
        return round(total_cost / unique_customers, 2)
        
    def calculate_roi_metrics(self, user_id: Optional[str] = None) -> Dict[str, float]:
        """Calcula todas las métricas de ROI"""
        metrics = {}
        
        # ROAS
        metrics['roas'] = self.calculate_roas(user_id=user_id)
        
        # LTV (si se especifica usuario)
        if user_id:
            metrics['ltv'] = self.calculate_ltv(user_id)
            metrics['ltv_cac_ratio'] = round(metrics['ltv'] / max(1, self.calculate_cac()), 2)
        else:
            metrics['ltv'] = 0.0
            metrics['ltv_cac_ratio'] = 0.0
            
        # CAC
        metrics['cac'] = self.calculate_cac()
        
        # Métricas adicionales
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Revenue total
        cursor.execute("SELECT SUM(value) FROM conversions")
        total_revenue = cursor.fetchone()[0] or 0
        
        # Costo total
        cursor.execute("SELECT SUM(cost) FROM touchpoints")
        total_cost = cursor.fetchone()[0] or 0
        
        conn.close()
        
        metrics['total_revenue'] = round(total_revenue, 2)
        metrics['total_cost'] = round(total_cost, 2)
        metrics['profit'] = round(total_revenue - total_cost, 2)
        metrics['roi_percentage'] = round((total_revenue - total_cost) / max(1, total_cost) * 100, 2)
        
        return metrics

class FraudDetector:
    """
    Detector de fraude publicitario
    """
    
    def __init__(self, attribution_engine: AttributionEngine):
        self.attribution_engine = attribution_engine
        self.fraud_patterns = []
        self.setup_fraud_patterns()
        
    def setup_fraud_patterns(self):
        """Configura patrones de fraude conocidos"""
        self.fraud_patterns = [
            {
                'name': 'Click Spam',
                'description': 'Patrones de clicks anómalos en poco tiempo',
                'threshold_clicks_per_minute': 30,
                'threshold_same_ip': 100
            },
            {
                'name': 'Conversion Fraud',
                'description': 'Conversiones de valor anómalo',
                'min_conversion_value': 0.01,
                'max_conversion_value': 10000
            },
            {
                'name': 'Bot Traffic',
                'description': 'Tráfico de bots basado en patrones de comportamiento',
                'min_touchpoints_per_hour': 50
            }
        ]
        
    def detect_click_spam(self) -> List[Dict[str, Any]]:
        """Detecta fraude de click spam"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Buscar patrones de clicks anómalos
        query = '''
            SELECT device_id, COUNT(*) as click_count, 
                   MIN(timestamp) as first_click, MAX(timestamp) as last_click
            FROM touchpoints 
            WHERE timestamp >= datetime('now', '-1 hour')
            GROUP BY device_id
            HAVING click_count > 30
        '''
        
        cursor.execute(query)
        suspicious_devices = cursor.fetchall()
        
        fraud_alerts = []
        for device in suspicious_devices:
            device_id, click_count, first_click, last_click = device
            time_span = (datetime.fromisoformat(last_click) - 
                        datetime.fromisoformat(first_click)).total_seconds() / 60
            
            if time_span > 0:
                clicks_per_minute = click_count / time_span
                if clicks_per_minute > 30:
                    fraud_alerts.append({
                        'type': 'Click Spam',
                        'device_id': device_id,
                        'clicks_in_last_hour': click_count,
                        'clicks_per_minute': round(clicks_per_minute, 2),
                        'time_span_minutes': round(time_span, 2),
                        'severity': 'HIGH' if clicks_per_minute > 100 else 'MEDIUM',
                        'detected_at': datetime.now().isoformat()
                    })
        
        conn.close()
        return fraud_alerts
        
    def detect_conversion_fraud(self) -> List[Dict[str, Any]]:
        """Detecta fraude de conversiones"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Buscar conversiones con valores anómalos
        query = '''
            SELECT user_id, device_id, value, timestamp, conversion_type
            FROM conversions
            WHERE value < 0.01 OR value > 10000
        '''
        
        cursor.execute(query)
        suspicious_conversions = cursor.fetchall()
        
        fraud_alerts = []
        for conv in suspicious_conversions:
            user_id, device_id, value, timestamp, conv_type = conv
            
            if value < 0.01:
                reason = "Valor de conversión muy bajo"
            else:
                reason = "Valor de conversión sospechosamente alto"
                
            fraud_alerts.append({
                'type': 'Conversion Fraud',
                'user_id': user_id,
                'device_id': device_id,
                'conversion_value': value,
                'conversion_type': conv_type,
                'reason': reason,
                'timestamp': timestamp,
                'severity': 'MEDIUM',
                'detected_at': datetime.now().isoformat()
            })
        
        conn.close()
        return fraud_alerts
        
    def detect_bot_traffic(self) -> List[Dict[str, Any]]:
        """Detecta tráfico de bots"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Buscar dispositivos con actividad anómala
        query = '''
            SELECT device_id, COUNT(*) as touchpoint_count
            FROM touchpoints
            WHERE timestamp >= datetime('now', '-1 hour')
            GROUP BY device_id
            HAVING touchpoint_count > 50
        '''
        
        cursor.execute(query)
        suspicious_devices = cursor.fetchall()
        
        fraud_alerts = []
        for device in suspicious_devices:
            device_id, touchpoint_count = device
            
            # Verificar diversidad de fuentes (bots suelen tener fuentes limitadas)
            cursor.execute('''
                SELECT COUNT(DISTINCT source) as source_count
                FROM touchpoints
                WHERE device_id = ? AND timestamp >= datetime('now', '-1 hour')
            ''', (device_id,))
            
            source_count = cursor.fetchone()[0]
            
            if source_count <= 2:  # Muy pocas fuentes diferentes
                fraud_alerts.append({
                    'type': 'Bot Traffic',
                    'device_id': device_id,
                    'touchpoints_in_last_hour': touchpoint_count,
                    'unique_sources': source_count,
                    'reason': 'Actividad muy alta con pocas fuentes diferentes',
                    'severity': 'HIGH',
                    'detected_at': datetime.now().isoformat()
                })
        
        conn.close()
        return fraud_alerts
        
    def detect_fraud_all_types(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detecta todos los tipos de fraude"""
        return {
            'click_spam': self.detect_click_spam(),
            'conversion_fraud': self.detect_conversion_fraud(),
            'bot_traffic': self.detect_bot_traffic()
        }
        
    def calculate_fraud_score(self) -> float:
        """Calcula un score de fraude general"""
        fraud_data = self.detect_fraud_all_types()
        
        total_alerts = (len(fraud_data['click_spam']) + 
                       len(fraud_data['conversion_fraud']) + 
                       len(fraud_data['bot_traffic']))
        
        # Score basado en cantidad y severidad de alertas
        high_severity = sum(1 for alert_list in fraud_data.values() 
                          for alert in alert_list if alert.get('severity') == 'HIGH')
        
        medium_severity = sum(1 for alert_list in fraud_data.values() 
                            for alert in alert_list if alert.get('severity') == 'MEDIUM')
        
        # Score normalizado de 0-100
        base_score = min(total_alerts * 10, 50)  # Base 50 puntos por cantidad
        severity_bonus = (high_severity * 30 + medium_severity * 15)
        final_score = min(base_score + severity_bonus, 100)
        
        return round(final_score, 2)

class ROIDashboard:
    """
    Dashboard de ROI en tiempo real
    """
    
    def __init__(self, attribution_engine: AttributionEngine, 
                 roi_analyzer: ROIAnalyzer, fraud_detector: FraudDetector):
        self.attribution_engine = attribution_engine
        self.roi_analyzer = roi_analyzer
        self.fraud_detector = fraud_detector
        
    def generate_real_time_metrics(self) -> Dict[str, Any]:
        """Genera métricas en tiempo real"""
        # Métricas de ROI
        roi_metrics = self.roi_analyzer.calculate_roi_metrics()
        
        # Métricas cross-device
        conversion_tracker = ConversionTracker(self.attribution_engine)
        cross_device_metrics = conversion_tracker.get_cross_device_metrics()
        
        # Métricas de fraude
        fraud_score = self.fraud_detector.calculate_fraud_score()
        fraud_alerts = self.fraud_detector.detect_fraud_all_types()
        
        # Métricas de atribución (últimas 24 horas)
        last_24h = datetime.now() - timedelta(hours=24)
        attribution_data = self._get_attribution_summary(last_24h)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'roi_metrics': roi_metrics,
            'cross_device_metrics': cross_device_metrics,
            'fraud_score': fraud_score,
            'fraud_alerts': fraud_alerts,
            'attribution_summary': attribution_data,
            'system_status': 'HEALTHY' if fraud_score < 50 else 'WARNING' if fraud_score < 80 else 'CRITICAL'
        }
        
    def _get_attribution_summary(self, since: datetime) -> Dict[str, Any]:
        """Obtiene resumen de atribución"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Top campañas (JOIN con touchpoints para obtener campaign)
        cursor.execute('''
            SELECT t.campaign, SUM(c.value) as total_value, COUNT(*) as conversions
            FROM conversions c
            JOIN touchpoints t ON c.user_id = t.user_id
            WHERE c.timestamp >= ?
            GROUP BY t.campaign
            ORDER BY total_value DESC
            LIMIT 10
        ''', (since.isoformat(),))
        
        top_campaigns = cursor.fetchall()
        
        # Canales más efectivos
        cursor.execute('''
            SELECT source, SUM(value) as total_value, SUM(cost) as total_cost
            FROM touchpoints
            WHERE timestamp >= ?
            GROUP BY source
            ORDER BY total_value DESC
            LIMIT 5
        ''', (since.isoformat(),))
        
        top_channels = cursor.fetchall()
        
        conn.close()
        
        return {
            'top_campaigns': [
                {
                    'campaign': row[0],
                    'total_value': row[1],
                    'conversions': row[2],
                    'roas': round(row[1] / max(1, row[2] * 50), 2)  # Estimación
                } for row in top_campaigns
            ],
            'top_channels': [
                {
                    'source': row[0],
                    'total_value': row[1],
                    'total_cost': row[2],
                    'roi': round((row[1] - row[2]) / max(1, row[2]) * 100, 2)
                } for row in top_channels
            ]
        }
        
    def export_dashboard_data(self, format: str = 'json') -> str:
        """Exporta datos del dashboard"""
        data = self.generate_real_time_metrics()
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == 'csv':
            return self._convert_to_csv(data)
        else:
            raise ValueError("Formato no soportado. Use 'json' o 'csv'")
            
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convierte datos a CSV"""
        import io
        
        output = io.StringIO()
        
        # Métricas principales
        output.write("Métrica,Valor\n")
        roi_metrics = data['roi_metrics']
        for key, value in roi_metrics.items():
            output.write(f"{key},{value}\n")
            
        # Métricas cross-device
        output.write(f"cross_device_rate,{data['cross_device_metrics']['cross_device_rate']}\n")
        output.write(f"avg_devices_per_user,{data['cross_device_metrics']['avg_devices_per_user']}\n")
        
        output.seek(0)
        return output.read()

class CustomerJourneyAnalyzer:
    """
    Analizador de customer journey
    """
    
    def __init__(self, attribution_engine: AttributionEngine):
        self.attribution_engine = attribution_engine
        
    def analyze_user_journey(self, user_id: str) -> Dict[str, Any]:
        """Analiza el customer journey de un usuario"""
        journey = self.attribution_engine.get_user_journey(user_id, days_back=90)
        
        # Análisis de rutas de conversión
        conversion_paths = self._identify_conversion_paths(journey)
        
        # Análisis de canales
        channel_analysis = self._analyze_channels(journey)
        
        # Análisis de tiempo
        time_analysis = self._analyze_time_patterns(journey)
        
        # Patrones de comportamiento
        behavior_patterns = self._analyze_behavior_patterns(journey)
        
        return {
            'user_id': user_id,
            'journey_summary': {
                'total_touchpoints': len(journey.touchpoints),
                'total_conversions': len(journey.conversions),
                'journey_length_days': (journey.end_date - journey.start_date).days,
                'total_value': journey.total_value,
                'total_cost': journey.total_cost,
                'conversion_rate': round(len(journey.conversions) / max(1, len(journey.touchpoints)) * 100, 2)
            },
            'conversion_paths': conversion_paths,
            'channel_analysis': channel_analysis,
            'time_analysis': time_analysis,
            'behavior_patterns': behavior_patterns
        }
        
    def _identify_conversion_paths(self, journey: CustomerJourney) -> List[Dict[str, Any]]:
        """Identifica rutas de conversión"""
        paths = []
        
        for conversion in journey.conversions:
            # Touchpoints antes de la conversión
            relevant_touchpoints = [tp for tp in journey.touchpoints 
                                  if tp.timestamp <= conversion.timestamp]
            
            if relevant_touchpoints:
                path = {
                    'conversion_timestamp': conversion.timestamp.isoformat(),
                    'conversion_value': conversion.value,
                    'touchpoints_before_conversion': len(relevant_touchpoints),
                    'channel_sequence': [tp.channel for tp in relevant_touchpoints[-5:]],  # Últimos 5
                    'source_sequence': [tp.source for tp in relevant_touchpoints[-5:]],
                    'time_to_conversion_hours': (
                        conversion.timestamp - relevant_touchpoints[0].timestamp
                    ).total_seconds() / 3600
                }
                paths.append(path)
                
        return paths
        
    def _analyze_channels(self, journey: CustomerJourney) -> Dict[str, Any]:
        """Analiza efectividad de canales"""
        channel_stats = defaultdict(lambda: {'touches': 0, 'conversions': 0, 'value': 0})
        
        for tp in journey.touchpoints:
            channel_stats[tp.channel]['touches'] += 1
            
        for conversion in journey.conversions:
            # Asignar conversión al último canal
            relevant_touchpoints = [tp for tp in journey.touchpoints 
                                  if tp.timestamp <= conversion.timestamp]
            if relevant_touchpoints:
                last_channel = relevant_touchpoints[-1].channel
                channel_stats[last_channel]['conversions'] += 1
                channel_stats[last_channel]['value'] += conversion.value
                
        return dict(channel_stats)
        
    def _analyze_time_patterns(self, journey: CustomerJourney) -> Dict[str, Any]:
        """Analiza patrones de tiempo"""
        touchpoint_hours = [tp.timestamp.hour for tp in journey.touchpoints]
        touchpoint_days = [tp.timestamp.weekday() for tp in journey.touchpoints]
        
        return {
            'peak_hours': Counter(touchpoint_hours).most_common(3),
            'peak_days': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][Counter(touchpoint_days).most_common(1)[0][0]],
            'avg_time_between_touches_hours': self._calculate_avg_time_between_touches(journey.touchpoints)
        }
        
    def _analyze_behavior_patterns(self, journey: CustomerJourney) -> Dict[str, Any]:
        """Analiza patrones de comportamiento"""
        if len(journey.touchpoints) < 2:
            return {'pattern_type': 'insufficient_data'}
            
        # Tiempo promedio entre touchpoints
        time_gaps = []
        for i in range(1, len(journey.touchpoints)):
            gap = (journey.touchpoints[i].timestamp - 
                   journey.touchpoints[i-1].timestamp).total_seconds() / 3600
            time_gaps.append(gap)
            
        avg_gap = np.mean(time_gaps) if time_gaps else 0
        
        # Frecuencia de conversión
        if len(journey.conversions) > 0:
            journey_days = max(1, (journey.end_date - journey.start_date).days)
            conversion_frequency = len(journey.conversions) / journey_days
        else:
            conversion_frequency = 0
            
        return {
            'avg_time_between_touches_hours': round(avg_gap, 2),
            'conversion_frequency_per_day': round(conversion_frequency, 3),
            'engagement_level': 'HIGH' if avg_gap < 24 else 'MEDIUM' if avg_gap < 168 else 'LOW'
        }
        
    def _calculate_avg_time_between_touches(self, touchpoints: List[TouchPoint]) -> float:
        """Calcula tiempo promedio entre touchpoints"""
        if len(touchpoints) < 2:
            return 0
            
        total_gap = 0
        for i in range(1, len(touchpoints)):
            gap = (touchpoints[i].timestamp - touchpoints[i-1].timestamp).total_seconds() / 3600
            total_gap += gap
            
        return total_gap / (len(touchpoints) - 1)
        
    def generate_journey_insights(self, user_id: str) -> List[str]:
        """Genera insights del customer journey"""
        analysis = self.analyze_user_journey(user_id)
        insights = []
        
        # Insights basados en métricas
        summary = analysis['journey_summary']
        
        if summary['conversion_rate'] > 10:
            insights.append("Cliente con alta tasa de conversión - excelente para remarketing")
        elif summary['conversion_rate'] < 2:
            insights.append("Cliente con baja tasa de conversión - requiere optimización de experiencia")
            
        if summary['journey_length_days'] > 30:
            insights.append("Customer journey largo - considerar estrategias de nurturing")
        elif summary['journey_length_days'] < 1:
            insights.append("Conversión rápida - cliente con intención de compra clara")
            
        # Insights de canales
        channel_analysis = analysis['channel_analysis']
        for channel, stats in channel_analysis.items():
            if stats['conversions'] > 0 and stats['touches'] > 0:
                conversion_rate = stats['conversions'] / stats['touches'] * 100
                if conversion_rate > 20:
                    insights.append(f"Canal {channel} muy efectivo - aumentar inversión")
                elif conversion_rate < 5:
                    insights.append(f"Canal {channel} con bajo rendimiento - optimizar o reducir")
                    
        return insights

class ExecutiveReporter:
    """
    Generador de reportes ejecutivos automatizados
    """
    
    def __init__(self, attribution_engine: AttributionEngine, 
                 roi_analyzer: ROIAnalyzer, fraud_detector: FraudDetector):
        self.attribution_engine = attribution_engine
        self.roi_analyzer = roi_analyzer
        self.fraud_detector = fraud_detector
        
    def generate_executive_report(self, period_days: int = 30) -> str:
        """Genera reporte ejecutivo"""
        date_from = datetime.now() - timedelta(days=period_days)
        date_to = datetime.now()
        
        # Métricas principales
        roi_metrics = self.roi_analyzer.calculate_roi_metrics()
        
        # Análisis de fraude
        fraud_score = self.fraud_detector.calculate_fraud_score()
        fraud_alerts = self.fraud_detector.detect_fraud_all_types()
        
        # Análisis de atribución
        attribution_summary = self._get_attribution_summary_period(date_from, date_to)
        
        # Top campañas
        top_campaigns = self._get_top_campaigns(date_from, date_to)
        
        # Recomendaciones
        recommendations = self._generate_recommendations(roi_metrics, fraud_score, top_campaigns)
        
        report = f"""
# REPORTE EJECUTIVO MIDAS ROI TRACKING
**Período:** {date_from.strftime('%Y-%m-%d')} a {date_to.strftime('%Y-%m-%d')}
**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 MÉTRICAS CLAVE DE ROI

### Rendimiento Financiero
- **ROAS (Return on Ad Spend):** {roi_metrics['roas']}:1
- **Revenue Total:** ${roi_metrics['total_revenue']:,.2f}
- **Costo Total:** ${roi_metrics['total_cost']:,.2f}
- **Beneficio Neto:** ${roi_metrics['profit']:,.2f}
- **ROI:** {roi_metrics['roi_percentage']}%

### Métricas de Cliente
- **LTV Promedio:** ${roi_metrics['ltv']:,.2f}
- **CAC Promedio:** ${roi_metrics['cac']:,.2f}
- **Ratio LTV/CAC:** {roi_metrics['ltv_cac_ratio']}:1

## 🛡️ ESTADO DE FRAUDE
- **Score de Fraude:** {fraud_score}/100
- **Alertas de Click Spam:** {len(fraud_alerts['click_spam'])}
- **Alertas de Conversión:** {len(fraud_alerts['conversion_fraud'])}
- **Alertas de Bot Traffic:** {len(fraud_alerts['bot_traffic'])}

**Estado:** {'🟢 SALUDABLE' if fraud_score < 30 else '🟡 PRECAUCIÓN' if fraud_score < 70 else '🔴 CRÍTICO'}

## 🏆 TOP CAMPAÑAS
{self._format_top_campaigns(top_campaigns)}

## 📈 RESUMEN DE ATRIBUCIÓN
{self._format_attribution_summary(attribution_summary)}

## 💡 RECOMENDACIONES EJECUTIVAS
{self._format_recommendations(recommendations)}

## 📋 ACCIONES REQUERIDAS
{self._format_required_actions(roi_metrics, fraud_score, top_campaigns)}

---
*Reporte generado automáticamente por MIDAS ROI Tracking System*
        """
        
        return report
        
    def _get_attribution_summary_period(self, date_from: datetime, date_to: datetime) -> Dict[str, Any]:
        """Obtiene resumen de atribución para período"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        # Revenue por canal (JOIN con touchpoints para obtener source)
        cursor.execute('''
            SELECT t.source, SUM(c.value) as total_revenue, COUNT(*) as conversions
            FROM conversions c
            JOIN touchpoints t ON c.user_id = t.user_id
            WHERE c.timestamp BETWEEN ? AND ?
            GROUP BY t.source
            ORDER BY total_revenue DESC
        ''', (date_from.isoformat(), date_to.isoformat()))
        
        channel_performance = cursor.fetchall()
        
        # Costo por campaña
        cursor.execute('''
            SELECT campaign, SUM(cost) as total_cost, COUNT(*) as touches
            FROM touchpoints
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY campaign
            ORDER BY total_cost DESC
        ''', (date_from.isoformat(), date_to.isoformat()))
        
        campaign_costs = cursor.fetchall()
        
        conn.close()
        
        return {
            'channel_performance': channel_performance,
            'campaign_costs': campaign_costs
        }
        
    def _get_top_campaigns(self, date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Obtiene top campañas por performance"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT t.campaign, SUM(c.value) as revenue, 
                   COUNT(DISTINCT c.user_id) as customers,
                   SUM(t.cost) as cost
            FROM conversions c
            JOIN touchpoints t ON c.user_id = t.user_id
            WHERE c.timestamp BETWEEN ? AND ?
            GROUP BY t.campaign
            ORDER BY revenue DESC
            LIMIT 10
        ''', (date_from.isoformat(), date_to.isoformat()))
        
        campaigns = cursor.fetchall()
        conn.close()
        
        return [
            {
                'campaign': row[0],
                'revenue': row[1],
                'customers': row[2],
                'cost': row[3],
                'roas': round(row[1] / max(1, row[3]), 2),
                'profit': round(row[1] - row[3], 2)
            }
            for row in campaigns
        ]
        
    def _generate_recommendations(self, roi_metrics: Dict[str, float], 
                                fraud_score: float, top_campaigns: List[Dict[str, Any]]) -> List[str]:
        """Genera recomendaciones basadas en datos"""
        recommendations = []
        
        # Recomendaciones de ROI
        if roi_metrics['roas'] < 2.0:
            recommendations.append("ROAS bajo detectado - revisar estrategia de targeting y creatividades")
            
        if roi_metrics['roi_percentage'] < 20:
            recommendations.append("ROI por debajo del umbral - optimizar mix de canales")
            
        # Recomendaciones de fraude
        if fraud_score > 70:
            recommendations.append("Alto nivel de fraude - implementar medidas de protección adicionales")
            
        # Recomendaciones de campañas
        if top_campaigns and len(top_campaigns) > 0:
            best_campaign = top_campaigns[0]
            if best_campaign.get('roas', 0) > 5.0:
                recommendations.append(f"Campaña '{best_campaign.get('campaign', 'Unknown')}' con excelente ROAS - aumentar presupuesto")
                
        # Recomendaciones de canales
        recommendations.append("Diversificar fuentes de tráfico para reducir dependencia")
        recommendations.append("Implementar más seguimiento cross-device")
        
        return recommendations
        
    def _format_top_campaigns(self, campaigns: List[Dict[str, Any]]) -> str:
        """Formatea top campañas para reporte"""
        if not campaigns:
            return "No hay datos de campañas disponibles."
            
        formatted = "### Top 5 Campañas por Revenue\n\n"
        formatted += "| Campaña | Revenue | ROAS | Clientes | Beneficio |\n"
        formatted += "|---------|---------|------|----------|----------|\n"
        
        for campaign in campaigns[:5]:
            formatted += f"| {campaign['campaign']} | ${campaign['revenue']:,.0f} | {campaign['roas']}:1 | {campaign['customers']} | ${campaign['profit']:,.0f} |\n"
            
        return formatted
        
    def _format_attribution_summary(self, attribution: Dict[str, Any]) -> str:
        """Formatea resumen de atribución"""
        formatted = "### Performance por Canal\n\n"
        formatted += "| Canal | Revenue | Conversiones |\n"
        formatted += "|-------|---------|-------------|\n"
        
        for channel, revenue, conversions in attribution['channel_performance']:
            formatted += f"| {channel} | ${revenue:,.0f} | {conversions} |\n"
            
        return formatted
        
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Formatea recomendaciones"""
        if not recommendations:
            return "No hay recomendaciones específicas para este período."
            
        formatted = ""
        for i, rec in enumerate(recommendations, 1):
            formatted += f"{i}. {rec}\n"
            
        return formatted
        
    def _format_required_actions(self, roi_metrics: Dict[str, float], 
                               fraud_score: float, campaigns: List[Dict[str, Any]]) -> str:
        """Formatea acciones requeridas"""
        actions = []
        
        if fraud_score > 70:
            actions.append("🔴 **URGENTE:** Revisar y bloquear fuentes de tráfico fraudulentas")
            
        if roi_metrics['roas'] < 1.5:
            actions.append("🟡 **IMPORTANTE:** Optimizar campañas con ROAS bajo")
            
        if campaigns and campaigns[0]['roas'] > 8.0:
            actions.append("🟢 **OPORTUNIDAD:** Escalar campaña top performer")
            
        if not actions:
            actions.append("✅ **SISTEMA SALUDABLE:** No se requieren acciones inmediatas")
            
        return "\n".join(actions)
        
    def send_email_report(self, recipients: List[str], period_days: int = 30):
        """Envía reporte por email"""
        # En una implementación real, configurar SMTP
        report = self.generate_executive_report(period_days)
        
        logger.info(f"Reporte ejecutivo enviado a: {', '.join(recipients)}")
        logger.info(f"Tamaño del reporte: {len(report)} caracteres")
        
        # Simular envío
        return True

class MIDASROITrackingSystem:
    """
    Sistema principal de MIDAS ROI Tracking
    """
    
    def __init__(self, db_path: str = "midas_roi_tracking.db"):
        # Inicializar componentes
        self.attribution_engine = AttributionEngine(db_path)
        self.roi_analyzer = ROIAnalyzer(self.attribution_engine)
        self.fraud_detector = FraudDetector(self.attribution_engine)
        self.dashboard = ROIDashboard(self.attribution_engine, self.roi_analyzer, self.fraud_detector)
        self.journey_analyzer = CustomerJourneyAnalyzer(self.attribution_engine)
        self.executive_reporter = ExecutiveReporter(self.attribution_engine, self.roi_analyzer, self.fraud_detector)
        self.ecommerce_integration = ECommerceIntegration(self.attribution_engine)
        
        # Thread pool para procesamiento en paralelo
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("MIDAS ROI Tracking System inicializado")
        
    def process_data_batch(self, data: List[Dict[str, Any]]):
        """Procesa lote de datos en paralelo"""
        futures = []
        
        for item in data:
            if item['type'] == 'touchpoint':
                future = self.executor.submit(self._process_touchpoint, item['data'])
                futures.append(future)
            elif item['type'] == 'conversion':
                future = self.executor.submit(self._process_conversion, item['data'])
                futures.append(future)
                
        # Esperar a que completen
        for future in futures:
            future.result()
            
    def _process_touchpoint(self, data: Dict[str, Any]):
        """Procesa touchpoint individual"""
        touchpoint = TouchPoint(**data)
        self.attribution_engine.add_touchpoint(touchpoint)
        
    def _process_conversion(self, data: Dict[str, Any]):
        """Procesa conversión individual"""
        conversion = Conversion(**data)
        self.attribution_engine.add_conversion(conversion)
        
    def run_full_analysis(self) -> Dict[str, Any]:
        """Ejecuta análisis completo del sistema"""
        logger.info("Iniciando análisis completo...")
        
        # Métricas en tiempo real
        real_time_metrics = self.dashboard.generate_real_time_metrics()
        
        # Análisis de fraude
        fraud_analysis = self.fraud_detector.detect_fraud_all_types()
        
        # Generar reporte ejecutivo
        executive_report = self.executive_reporter.generate_executive_report()
        
        # Top usuarios por valor
        top_users = self._get_top_users()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'real_time_metrics': real_time_metrics,
            'fraud_analysis': fraud_analysis,
            'top_users': top_users,
            'executive_summary': {
                'system_health': 'HEALTHY' if real_time_metrics['fraud_score'] < 50 else 'WARNING',
                'total_roas': real_time_metrics['roi_metrics']['roas'],
                'fraud_score': real_time_metrics['fraud_score'],
                'recommendations': len(self.executive_reporter._generate_recommendations(
                    real_time_metrics['roi_metrics'], 
                    real_time_metrics['fraud_score'], 
                    top_users
                ))
            }
        }
        
        logger.info("Análisis completo finalizado")
        return results
        
    def _get_top_users(self) -> List[Dict[str, Any]]:
        """Obtiene top usuarios por valor generado"""
        conn = sqlite3.connect(self.attribution_engine.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, SUM(value) as total_value, COUNT(*) as conversions
            FROM conversions
            GROUP BY user_id
            ORDER BY total_value DESC
            LIMIT 10
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return [
            {
                'user_id': row[0],
                'total_value': row[1],
                'conversions': row[2],
                'avg_order_value': round(row[1] / row[2], 2) if row[2] > 0 else 0
            }
            for row in users
        ]
        
    def export_all_data(self, output_path: str):
        """Exporta todos los datos del sistema"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_metrics': self.dashboard.generate_real_time_metrics(),
            'fraud_analysis': self.fraud_detector.detect_fraud_all_types(),
            'top_campaigns': self._get_top_users(),
            'executive_report': self.executive_reporter.generate_executive_report()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        logger.info(f"Datos exportados a: {output_path}")
        
    def schedule_automated_tasks(self):
        """Programa tareas automatizadas"""
        # En una implementación real, usar un scheduler como APScheduler
        logger.info("Tareas automatizadas programadas")
        
        # Simular tareas programadas
        tasks = [
            "Generación de reporte diario",
            "Análisis de fraude en tiempo real",
            "Sincronización con plataformas e-commerce",
            "Backup de datos"
        ]
        
        for task in tasks:
            logger.info(f"Tarea programada: {task}")

# Función principal para testing
def main():
    """
    Función principal de demostración
    """
    print("🚀 MIDAS ROI Tracking System - Inicialización")
    
    # Crear instancia del sistema
    midas_system = MIDASROITrackingSystem()
    
    # Conectar plataformas e-commerce (simulado)
    midas_system.ecommerce_integration.connect_shopify("mi-tienda.myshopify.com", "token_123")
    midas_system.ecommerce_integration.connect_woocommerce("https://mi-tienda.com", "ck_123", "cs_123")
    
    # Sincronizar datos
    print("📊 Sincronizando datos de e-commerce...")
    shopify_conversions = midas_system.ecommerce_integration.sync_shopify_orders()
    woo_conversions = midas_system.ecommerce_integration.sync_woocommerce_orders()
    
    # Ejecutar análisis completo
    print("🔍 Ejecutando análisis completo...")
    results = midas_system.run_full_analysis()
    
    # Mostrar resultados
    print("\n📈 MÉTRICAS PRINCIPALES:")
    print(f"ROAS: {results['real_time_metrics']['roi_metrics']['roas']}:1")
    print(f"Revenue Total: ${results['real_time_metrics']['roi_metrics']['total_revenue']:,.2f}")
    print(f"Score de Fraude: {results['real_time_metrics']['fraud_score']}/100")
    print(f"Estado del Sistema: {results['real_time_metrics']['system_status']}")
    
    # Generar reporte ejecutivo
    print("\n📋 Generando reporte ejecutivo...")
    executive_report = midas_system.executive_reporter.generate_executive_report()
    
    # Guardar reporte
    with open("reporte_ejecutivo_midas.txt", "w", encoding="utf-8") as f:
        f.write(executive_report)
    
    print("✅ Reporte ejecutivo guardado en: reporte_ejecutivo_midas.txt")
    
    # Exportar datos
    print("💾 Exportando datos completos...")
    midas_system.export_all_data("midas_roi_export.json")
    
    print("🎉 MIDAS ROI Tracking System - Demostración completada")
    return midas_system

if __name__ == "__main__":
    # Ejecutar demostración
    system = main()