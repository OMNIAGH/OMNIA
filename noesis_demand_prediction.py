"""
NOESIS Demand Prediction System
Sistema de predicción de demanda para NOESIS

Funcionalidades:
- Modelos específicos para demanda de productos/servicios
- Integración con datos de marketing (campañas, presupuestos, canales)
- Factores externos (estacionalidad, eventos, competencia)
- Optimización de inventario y pricing
- Predicciones multi-horizon (corto, medio, largo plazo)
- Confianza e intervalos de predicción
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import json
import joblib

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionHorizon(Enum):
    """Horizontes de predicción disponibles"""
    SHORT_TERM = "short_term"      # 1-7 días
    MEDIUM_TERM = "medium_term"    # 1-12 semanas
    LONG_TERM = "long_term"        # 1-12 meses

class ProductType(Enum):
    """Tipos de productos/servicios"""
    PHYSICAL_PRODUCT = "physical_product"
    DIGITAL_SERVICE = "digital_service"
    SUBSCRIPTION = "subscription"
    SEASONAL_PRODUCT = "seasonal_product"

@dataclass
class MarketingData:
    """Datos de marketing para análisis de impacto"""
    campaign_id: str
    channel: str
    budget: float
    start_date: datetime
    end_date: datetime
    campaign_type: str
    impressions: int
    clicks: int
    conversions: int
    ctr: float  # Click-through rate
    cpc: float  # Cost per click
    roas: float  # Return on ad spend

@dataclass
class ExternalFactor:
    """Factores externos que afectan la demanda"""
    date: datetime
    factor_type: str  # 'seasonal', 'event', 'competitor', 'economic'
    factor_name: str
    impact_magnitude: float  # -1.0 a 1.0
    duration_days: int

@dataclass
class PredictionResult:
    """Resultado de predicción con intervalos de confianza"""
    product_id: str
    prediction_date: datetime
    horizon: PredictionHorizon
    predicted_demand: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    model_used: str
    features_importance: Dict[str, float]

class DemandPredictor:
    """
    Clase principal para predicción de demanda
    Maneja modelos específicos para diferentes tipos de productos/servicios
    """
    
    def __init__(self, model_config: Optional[Dict] = None):
        """
        Inicializa el predictor de demanda
        
        Args:
            model_config: Configuración específica del modelo
        """
        self.models = {}
        self.scalers = {}
        self.feature_encoders = {}
        self.model_config = model_config or self._get_default_config()
        self.is_fitted = False
        
        # Configurar modelos por tipo de producto
        self._setup_models()
        
        logger.info("DemandPredictor inicializado correctamente")
    
    def _get_default_config(self) -> Dict:
        """Configuración por defecto del modelo"""
        return {
            'short_term': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'features': ['lag_1', 'lag_7', 'trend', 'day_of_week', 'month']
            },
            'medium_term': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'features': ['lag_7', 'lag_30', 'seasonal', 'trend', 'marketing_impact']
            },
            'long_term': {
                'model': Ridge,
                'params': {
                    'alpha': 1.0
                },
                'features': ['monthly_trend', 'seasonal', 'external_factors', 'marketing_spend']
            }
        }
    
    def _setup_models(self):
        """Configura los modelos para cada horizonte temporal"""
        for horizon in PredictionHorizon:
            config = self.model_config.get(horizon.value, {})
            model_class = config.get('model', RandomForestRegressor)
            params = config.get('params', {})
            
            self.models[horizon] = model_class(**params)
            self.scalers[horizon] = StandardScaler()
            self.feature_encoders[horizon] = LabelEncoder()
        
        self.is_fitted = True
    
    def prepare_features(self, 
                        data: pd.DataFrame, 
                        product_type: ProductType,
                        horizon: PredictionHorizon) -> pd.DataFrame:
        """
        Prepara las características para el modelo
        
        Args:
            data: DataFrame con datos históricos
            product_type: Tipo de producto/servicio
            horizon: Horizonte de predicción
            
        Returns:
            DataFrame con características preparadas
        """
        features = data.copy()
        
        # Características de lag (valores pasados)
        for lag in [1, 7, 14, 30]:
            if f'lag_{lag}' not in features.columns:
                features[f'lag_{lag}'] = features['demand'].shift(lag)
        
        # Características de tendencia
        features['trend'] = range(len(features))
        
        # Características de tiempo
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        features['quarter'] = features['date'].dt.quarter
        features['is_weekend'] = features['date'].dt.dayofweek >= 5
        
        # Características de estacionalidad (si es producto estacional)
        if product_type == ProductType.SEASONAL_PRODUCT:
            features['seasonal'] = np.sin(2 * np.pi * features['month'] / 12)
        
        # Características de marketing (impacto agregado)
        if 'marketing_impact' not in features.columns:
            features['marketing_impact'] = 0.0
        
        # Características de factores externos
        if 'external_factors' not in features.columns:
            features['external_factors'] = 0.0
        
        # Características específicas por tipo de producto
        if product_type == ProductType.DIGITAL_SERVICE:
            # Para servicios digitales: tráfico web, suscripciones
            features['traffic_trend'] = features.get('web_traffic', 0).pct_change()
            features['engagement_rate'] = features.get('user_engagement', 0)
        
        elif product_type == ProductType.SUBSCRIPTION:
            # Para suscripciones: churn rate, retention
            features['churn_rate'] = features.get('monthly_churn', 0)
            features['retention_rate'] = features.get('retention', 0)
        
        # Eliminar filas con valores NaN
        features = features.dropna()
        
        return features
    
    def train(self, 
              data: pd.DataFrame,
              product_id: str,
              product_type: ProductType,
              target_column: str = 'demand') -> Dict[str, float]:
        """
        Entrena el modelo para un producto específico
        
        Args:
            data: DataFrame con datos históricos
            product_id: ID del producto
            product_type: Tipo de producto
            target_column: Nombre de la columna objetivo
            
        Returns:
            Métricas de entrenamiento
        """
        logger.info(f"Entrenando modelo para producto {product_id} tipo {product_type}")
        
        metrics = {}
        
        # Preparar datos
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        for horizon in PredictionHorizon:
            # Preparar características
            features = self.prepare_features(data, product_type, horizon)
            
            if len(features) < 50:  # Datos insuficientes
                logger.warning(f"Datos insuficientes para {product_id} en horizonte {horizon}")
                continue
            
            # Seleccionar características según configuración
            config = self.model_config.get(horizon.value, {})
            selected_features = config.get('features', [])
            available_features = [f for f in selected_features if f in features.columns]
            
            if not available_features:
                available_features = features.select_dtypes(include=[np.number]).columns.tolist()
                available_features = [f for f in available_features if f != target_column][:10]
            
            X = features[available_features]
            y = features[target_column]
            
            # Escalar características
            X_scaled = self.scalers[horizon].fit_transform(X)
            
            # Entrenar modelo
            self.models[horizon].fit(X_scaled, y)
            
            # Calcular métricas
            y_pred = self.models[horizon].predict(X_scaled)
            metrics[horizon.value] = {
                'mae': mean_absolute_error(y, y_pred),
                'mse': mean_squared_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        logger.info(f"Entrenamiento completado para {product_id}")
        return metrics
    
    def predict(self,
                data: pd.DataFrame,
                product_id: str,
                product_type: ProductType,
                horizon: PredictionHorizon,
                external_factors: Optional[List[ExternalFactor]] = None,
                marketing_data: Optional[List[MarketingData]] = None) -> List[PredictionResult]:
        """
        Realiza predicciones de demanda
        
        Args:
            data: DataFrame con datos históricos
            product_id: ID del producto
            product_type: Tipo de producto
            horizon: Horizonte de predicción
            external_factors: Factores externos
            marketing_data: Datos de marketing
            
        Returns:
            Lista de resultados de predicción
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        logger.info(f"Realizando predicción para {product_id} en horizonte {horizon}")
        
        # Preparar datos
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # Preparar características
        features = self.prepare_features(data, product_type, horizon)
        
        # Determinar horizonte de predicción
        if horizon == PredictionHorizon.SHORT_TERM:
            future_days = 7
        elif horizon == PredictionHorizon.MEDIUM_TERM:
            future_days = 84  # 12 semanas
        else:  # LONG_TERM
            future_days = 365  # 12 meses
        
        # Generar fechas futuras
        last_date = data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=future_days, freq='D')
        
        results = []
        
        # Obtener último registro para predicciones
        last_record = data.iloc[-1]
        
        for future_date in future_dates:
            # Crear características futuras
            future_features = self._create_future_features(
                last_record, future_date, product_type, 
                external_factors, marketing_data
            )
            
            # Obtener características del modelo
            config = self.model_config.get(horizon.value, {})
            selected_features = config.get('features', [])
            available_features = [f for f in selected_features if f in future_features]
            
            if not available_features:
                available_features = [f for f in future_features if isinstance(future_features[f], (int, float))]
            
            X_future = np.array([future_features[f] for f in available_features]).reshape(1, -1)
            X_future_scaled = self.scalers[horizon].transform(X_future)
            
            # Realizar predicción
            prediction = self.models[horizon].predict(X_future_scaled)[0]
            
            # Calcular intervalos de confianza usando residuales
            residuals = self._get_prediction_uncertainty(horizon)
            confidence_level = 0.95
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            # Error estándar de predicción
            std_error = np.sqrt(np.mean(residuals**2))
            
            lower_bound = prediction - z_score * std_error
            upper_bound = prediction + z_score * std_error
            
            # Asegurar valores positivos para demanda
            lower_bound = max(0, lower_bound)
            prediction = max(0, prediction)
            upper_bound = max(0, upper_bound)
            
            # Obtener importancia de características
            feature_importance = self._get_feature_importance(horizon, available_features)
            
            result = PredictionResult(
                product_id=product_id,
                prediction_date=future_date,
                horizon=horizon,
                predicted_demand=prediction,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                model_used=str(type(self.models[horizon]).__name__),
                features_importance=feature_importance
            )
            
            results.append(result)
        
        return results
    
    def _create_future_features(self,
                               last_record: pd.Series,
                               future_date: datetime,
                               product_type: ProductType,
                               external_factors: Optional[List[ExternalFactor]] = None,
                               marketing_data: Optional[List[MarketingData]] = None) -> Dict[str, float]:
        """Crea características para predicción futura"""
        features = {}
        
        # Características de lag (usar últimos valores conocidos)
        for lag in [1, 7, 14, 30]:
            features[f'lag_{lag}'] = last_record.get('demand', 0) * (0.9 ** lag)  # Decay
        
        # Características de tiempo
        features['trend'] = last_record.get('trend', 0) + 1
        features['day_of_week'] = future_date.weekday()
        features['month'] = future_date.month
        features['quarter'] = (future_date.month - 1) // 3 + 1
        features['is_weekend'] = int(future_date.weekday() >= 5)
        
        # Características de estacionalidad
        if product_type == ProductType.SEASONAL_PRODUCT:
            features['seasonal'] = np.sin(2 * np.pi * future_date.month / 12)
        else:
            features['seasonal'] = 0.0
        
        # Impacto de marketing
        marketing_impact = self._calculate_marketing_impact(marketing_data, future_date)
        features['marketing_impact'] = marketing_impact
        
        # Factores externos
        external_impact = self._calculate_external_impact(external_factors, future_date)
        features['external_factors'] = external_impact
        
        # Características específicas por tipo de producto
        if product_type == ProductType.DIGITAL_SERVICE:
            features['traffic_trend'] = last_record.get('traffic_trend', 0)
            features['engagement_rate'] = last_record.get('engagement_rate', 0)
        elif product_type == ProductType.SUBSCRIPTION:
            features['churn_rate'] = last_record.get('churn_rate', 0)
            features['retention_rate'] = last_record.get('retention_rate', 0)
        
        return features
    
    def _calculate_marketing_impact(self, 
                                  marketing_data: Optional[List[MarketingData]], 
                                  date: datetime) -> float:
        """Calcula el impacto de marketing para una fecha específica"""
        if not marketing_data:
            return 0.0
        
        total_impact = 0.0
        for campaign in marketing_data:
            if campaign.start_date <= date <= campaign.end_date:
                # Impacto basado en ROAS y presupuesto
                impact = (campaign.roas * campaign.budget) / 1000
                total_impact += impact
        
        # Normalizar el impacto
        return min(total_impact / 100, 1.0)  # Máximo impacto de 1.0
    
    def _calculate_external_impact(self, 
                                 external_factors: Optional[List[ExternalFactor]], 
                                 date: datetime) -> float:
        """Calcula el impacto de factores externos"""
        if not external_factors:
            return 0.0
        
        total_impact = 0.0
        for factor in external_factors:
            factor_start = factor.date
            factor_end = factor.date + timedelta(days=factor.duration_days)
            
            if factor_start <= date <= factor_end:
                # Aplicar decay del impacto
                days_elapsed = (date - factor_start).days
                decay_factor = max(0, 1 - days_elapsed / factor.duration_days)
                total_impact += factor.impact_magnitude * decay_factor
        
        return total_impact
    
    def _get_prediction_uncertainty(self, horizon: PredictionHorizon) -> np.ndarray:
        """Obtiene los residuales del modelo para estimar incertidumbre"""
        # En una implementación completa, esto vendría del entrenamiento
        # Por ahora, retornamos valores por defecto basados en el tipo de modelo
        if isinstance(self.models[horizon], RandomForestRegressor):
            return np.array([0.1, 0.15, 0.12, 0.18, 0.09, 0.11, 0.14])
        elif isinstance(self.models[horizon], GradientBoostingRegressor):
            return np.array([0.08, 0.12, 0.10, 0.13, 0.09, 0.11, 0.12])
        else:  # Ridge, LinearRegression
            return np.array([0.20, 0.25, 0.22, 0.28, 0.21, 0.23, 0.26])
    
    def _get_feature_importance(self, horizon: PredictionHorizon, feature_names: List[str]) -> Dict[str, float]:
        """Obtiene la importancia de las características"""
        model = self.models[horizon]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances.tolist()))
        elif hasattr(model, 'coef_'):
            # Para modelos lineales, usar valores absolutos de coeficientes
            coefs = np.abs(model.coef_)
            return dict(zip(feature_names, coefs.tolist()))
        else:
            # Características por defecto
            return {f: 1.0/len(feature_names) for f in feature_names}
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_encoders': self.feature_encoders,
            'model_config': self.model_config,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo entrenado"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_encoders = model_data['feature_encoders']
        self.model_config = model_data['model_config']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"Modelo cargado desde {filepath}")


class MarketingImpactAnalyzer:
    """
    Analizador de impacto de marketing
    Integra datos de campañas, presupuestos y canales para mejorar predicciones
    """
    
    def __init__(self):
        """Inicializa el analizador de impacto de marketing"""
        self.marketing_data = []
        self.channel_performance = {}
        self.campaign_effectiveness = {}
        logger.info("MarketingImpactAnalyzer inicializado")
    
    def add_marketing_data(self, data: List[MarketingData]):
        """Agrega datos de marketing al sistema"""
        self.marketing_data.extend(data)
        self._update_performance_metrics()
        logger.info(f"Agregados {len(data)} registros de marketing")
    
    def _update_performance_metrics(self):
        """Actualiza métricas de rendimiento de canales y campañas"""
        if not self.marketing_data:
            return
        
        df = pd.DataFrame([{
            'channel': d.channel,
            'campaign_id': d.campaign_id,
            'budget': d.budget,
            'impressions': d.impressions,
            'clicks': d.clicks,
            'conversions': d.conversions,
            'ctr': d.ctr,
            'cpc': d.cpc,
            'roas': d.roas
        } for d in self.marketing_data])
        
        # Métricas por canal
        self.channel_performance = df.groupby('channel').agg({
            'budget': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'ctr': 'mean',
            'cpc': 'mean',
            'roas': 'mean'
        }).to_dict('index')
        
        # Efectividad de campañas
        self.campaign_effectiveness = df.groupby('campaign_id').agg({
            'roas': 'mean',
            'budget': 'sum',
            'conversions': 'sum'
        }).to_dict('index')
    
    def analyze_campaign_impact(self, 
                              campaign_id: str,
                              date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """
        Analiza el impacto de una campaña específica en la demanda
        
        Args:
            campaign_id: ID de la campaña
            date_range: Rango de fechas a analizar
            
        Returns:
            Análisis de impacto de la campaña
        """
        start_date, end_date = date_range
        
        campaign_data = [d for d in self.marketing_data 
                        if d.campaign_id == campaign_id and 
                        start_date <= d.start_date <= end_date]
        
        if not campaign_data:
            return {'error': 'No se encontraron datos para la campaña'}
        
        # Calcular impacto agregado
        total_budget = sum(d.budget for d in campaign_data)
        total_conversions = sum(d.conversions for d in campaign_data)
        avg_roas = np.mean([d.roas for d in campaign_data])
        
        # Impacto estimado en demanda (sistema de atribución simple)
        estimated_demand_impact = self._calculate_demand_attribution(campaign_data)
        
        return {
            'campaign_id': campaign_id,
            'total_budget': total_budget,
            'total_conversions': total_conversions,
            'average_roas': avg_roas,
            'estimated_demand_impact': estimated_demand_impact,
            'effectiveness_score': self._calculate_effectiveness_score(campaign_data),
            'channels_used': list(set(d.channel for d in campaign_data)),
            'duration_days': (end_date - start_date).days
        }
    
    def optimize_budget_allocation(self, 
                                 channels: List[str], 
                                 total_budget: float,
                                 target_kpis: Dict[str, float]) -> Dict[str, float]:
        """
        Optimiza la asignación de presupuesto entre canales
        
        Args:
            channels: Lista de canales disponibles
            total_budget: Presupuesto total a asignar
            target_kpis: KPIs objetivo (ROAS, conversiones, etc.)
            
        Returns:
            Asignación óptima de presupuesto por canal
        """
        allocation = {}
        remaining_budget = total_budget
        
        # Priorizar canales por rendimiento histórico
        channel_scores = {}
        for channel in channels:
            if channel in self.channel_performance:
                perf = self.channel_performance[channel]
                # Score basado en ROAS y conversiones
                score = (perf.get('roas', 0) * 0.7) + (perf.get('conversions', 0) * 0.3)
                channel_scores[channel] = score
            else:
                channel_scores[channel] = 0  # Score neutro para canales sin historial
        
        # Asignar presupuesto proporcionalmente al score
        total_score = sum(channel_scores.values())
        if total_score == 0:
            # Distribución equitativa si no hay historial
            allocation = {channel: total_budget / len(channels) for channel in channels}
        else:
            for channel in channels:
                proportion = channel_scores[channel] / total_score
                allocation[channel] = total_budget * proportion
        
        return allocation
    
    def predict_marketing_impact(self, 
                               channel: str,
                               budget: float,
                               campaign_duration: int) -> Dict[str, float]:
        """
        Predice el impacto de una inversión de marketing
        
        Args:
            channel: Canal de marketing
            budget: Presupuesto propuesto
            campaign_duration: Duración de la campaña en días
            
        Returns:
            Predicción de impacto
        """
        if channel not in self.channel_performance:
            # Usar métricas promedio si no hay historial
            return {
                'predicted_conversions': budget / 50,  # Estimación conservadora
                'predicted_revenue': budget * 2.0,     # ROAS de 2.0
                'confidence_level': 0.5
            }
        
        perf = self.channel_performance[channel]
        
        # Proyecciones basadas en rendimiento histórico
        historical_roas = perf.get('roas', 2.0)
        historical_cpc = perf.get('cpc', 10.0)
        
        # Predicciones
        predicted_clicks = budget / historical_cpc
        predicted_conversions = predicted_clicks * (perf.get('ctr', 0.02))
        predicted_revenue = predicted_conversions * (budget / predicted_conversions) * historical_roas
        
        # Ajuste por duración de campaña
        duration_factor = min(campaign_duration / 30, 2.0)  # Máximo 2x por duración
        predicted_conversions *= duration_factor
        predicted_revenue *= duration_factor
        
        return {
            'predicted_conversions': predicted_conversions,
            'predicted_revenue': predicted_revenue,
            'predicted_roas': predicted_revenue / budget,
            'confidence_level': min(len(self.marketing_data) / 100, 0.95)
        }
    
    def _calculate_demand_attribution(self, campaign_data: List[MarketingData]) -> float:
        """Calcula la atribución de demanda a campañas de marketing"""
        # Modelo de atribución simple: ROAS y conversiones
        total_attribution = 0
        
        for campaign in campaign_data:
            # Atribución basada en ROAS y duración
            daily_impact = (campaign.roas * campaign.budget) / max(campaign_duration(campaign), 1)
            total_attribution += daily_impact
        
        return total_attribution
    
    def _calculate_effectiveness_score(self, campaign_data: List[MarketingData]) -> float:
        """Calcula un score de efectividad para campañas"""
        if not campaign_data:
            return 0.0
        
        # Score compuesto basado en múltiples métricas
        roas_scores = [d.roas for d in campaign_data]
        conversion_rates = [d.conversions / max(d.impressions, 1) for d in campaign_data]
        budget_efficiency = [d.conversions / max(d.budget, 1) for d in campaign_data]
        
        # Normalizar scores
        roas_score = np.mean(roas_scores) / 10  # Normalizar ROAS
        conversion_score = np.mean(conversion_rates) * 100  # Convertir a porcentaje
        efficiency_score = np.mean(budget_efficiency)
        
        # Score ponderado
        effectiveness = (roas_score * 0.4) + (conversion_score * 0.3) + (efficiency_score * 0.3)
        
        return min(effectiveness, 10.0)  # Máximo score de 10
    
    def get_channel_recommendations(self, 
                                  product_type: ProductType,
                                  budget_range: Tuple[float, float]) -> List[Dict[str, Any]]:
        """
        Recomienda canales óptimos para un tipo de producto
        
        Args:
            product_type: Tipo de producto/servicio
            budget_range: Rango de presupuesto
            
        Returns:
            Lista de recomendaciones de canales
        """
        min_budget, max_budget = budget_range
        
        recommendations = []
        
        for channel, perf in self.channel_performance.items():
            # Filtrar por presupuesto
            if perf.get('budget', 0) < min_budget or perf.get('budget', 0) > max_budget * 2:
                continue
            
            # Score de compatibilidad por tipo de producto
            compatibility = self._get_channel_product_compatibility(channel, product_type)
            
            # Score general de rendimiento
            performance_score = (
                perf.get('roas', 1) * 0.3 +
                (perf.get('conversions', 0) / max(perf.get('impressions', 1), 1)) * 0.4 +
                (1 / max(perf.get('cpc', 100), 1)) * 0.3
            )
            
            # Score final
            final_score = (performance_score * 0.7) + (compatibility * 0.3)
            
            recommendations.append({
                'channel': channel,
                'score': final_score,
                'estimated_roas': perf.get('roas', 1),
                'estimated_conversions': perf.get('conversions', 0),
                'compatibility': compatibility,
                'recommended_budget': min_budget + (max_budget - min_budget) * final_score
            })
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:5]  # Top 5 recomendaciones
    
    def _get_channel_product_compatibility(self, 
                                         channel: str, 
                                         product_type: ProductType) -> float:
        """Evalúa compatibilidad entre canal y tipo de producto"""
        compatibility_matrix = {
            ProductType.PHYSICAL_PRODUCT: {
                'google_ads': 0.9,
                'facebook': 0.8,
                'instagram': 0.7,
                'email': 0.6,
                'affiliate': 0.8
            },
            ProductType.DIGITAL_SERVICE: {
                'google_ads': 0.8,
                'facebook': 0.6,
                'instagram': 0.5,
                'email': 0.9,
                'linkedin': 0.8
            },
            ProductType.SUBSCRIPTION: {
                'google_ads': 0.7,
                'facebook': 0.8,
                'email': 0.9,
                'linkedin': 0.7,
                'content_marketing': 0.8
            },
            ProductType.SEASONAL_PRODUCT: {
                'google_ads': 0.9,
                'facebook': 0.9,
                'instagram': 0.8,
                'tiktok': 0.7,
                'seasonal_ads': 1.0
            }
        }
        
        return compatibility_matrix.get(product_type, {}).get(channel, 0.5)


class InventoryOptimizer:
    """
    Optimizador de inventario y pricing
    Utiliza predicciones de demanda para optimizar niveles de stock y precios
    """
    
    def __init__(self, cost_structure: Optional[Dict] = None):
        """
        Inicializa el optimizador de inventario
        
        Args:
            cost_structure: Estructura de costos (costo_unitario, costo_almacenamiento, etc.)
        """
        self.cost_structure = cost_structure or {
            'costo_unitario': 10.0,
            'costo_almacenamiento': 1.0,  # Por unidad por mes
            'costo_pedido': 50.0,
            'costo_faltante': 20.0,  # Por unidad faltante
            'tasa_descuento': 0.1
        }
        self.current_inventory = {}
        self.supplier_info = {}
        logger.info("InventoryOptimizer inicializado")
    
    def set_inventory_level(self, product_id: str, current_stock: float, 
                          lead_time_days: int = 7):
        """Establece el nivel actual de inventario para un producto"""
        self.current_inventory[product_id] = {
            'current_stock': current_stock,
            'lead_time_days': lead_time_days,
            'last_updated': datetime.now()
        }
        logger.info(f"Inventario establecido para {product_id}: {current_stock} unidades")
    
    def set_supplier_info(self, supplier_id: str, min_order: float, 
                         cost_per_unit: float, lead_time_days: int = 7):
        """Establece información del proveedor"""
        self.supplier_info[supplier_id] = {
            'min_order': min_order,
            'cost_per_unit': cost_per_unit,
            'lead_time_days': lead_time_days
        }
    
    def calculate_optimal_inventory_level(self, 
                                        product_id: str,
                                        demand_predictions: List[PredictionResult],
                                        service_level: float = 0.95) -> Dict[str, Any]:
        """
        Calcula el nivel óptimo de inventario
        
        Args:
            product_id: ID del producto
            demand_predictions: Predicciones de demanda
            service_level: Nivel de servicio deseado (0.95 = 95%)
            
        Returns:
            Recomendación de nivel de inventario
        """
        if product_id not in self.current_inventory:
            raise ValueError(f"No hay información de inventario para {product_id}")
        
        current_info = self.current_inventory[product_id]
        
        # Calcular demanda promedio y desviación estándar
        demands = [p.predicted_demand for p in demand_predictions]
        mean_demand = np.mean(demands)
        std_demand = np.std(demands)
        
        lead_time = current_info['lead_time_days']
        
        # Punto de reorden (ROP)
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * std_demand * np.sqrt(lead_time)
        reorder_point = mean_demand * lead_time + safety_stock
        
        # Cantidad económica de pedido (EOQ)
        annual_demand = mean_demand * 365
        ordering_cost = self.cost_structure['costo_pedido']
        holding_cost = self.cost_structure['costo_almacenamiento']
        
        if holding_cost > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        else:
            eoq = annual_demand * 0.1  # Fallback
        
        # Nivel máximo de inventario
        max_inventory = reorder_point + eoq
        
        # Costo total de inventario
        total_cost = self._calculate_inventory_cost(
            max_inventory, annual_demand, eoq, safety_stock
        )
        
        return {
            'product_id': product_id,
            'current_stock': current_info['current_stock'],
            'reorder_point': reorder_point,
            'safety_stock': safety_stock,
            'economic_order_quantity': eoq,
            'max_inventory_level': max_inventory,
            'recommended_order': max(0, reorder_point - current_info['current_stock']),
            'total_inventory_cost': total_cost,
            'service_level': service_level,
            'lead_time_days': lead_time
        }
    
    def optimize_pricing(self, 
                       product_id: str,
                       demand_predictions: List[PredictionResult],
                       current_price: float,
                       price_elasticity: float = -1.5,
                       competitor_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimiza el precio basado en demanda y elasticidad
        
        Args:
            product_id: ID del producto
            demand_predictions: Predicciones de demanda
            current_price: Precio actual
            price_elasticity: Elasticidad precio-demanda (negativa)
            competitor_prices: Precios de competidores
            
        Returns:
            Recomendación de precio óptimo
        """
        # Demanda promedio
        mean_demand = np.mean([p.predicted_demand for p in demand_predictions])
        
        # Calcular precio óptimo usando elasticidad
        if price_elasticity < 0:
            # Fórmula: P* = P_current * (elasticity / (elasticity + 1))
            optimal_price = current_price * (abs(price_elasticity) / (abs(price_elasticity) + 1))
        else:
            optimal_price = current_price
        
        # Ajustar por rango de precios de competidores
        if competitor_prices:
            competitor_avg = np.mean(list(competitor_prices.values()))
            if abs(optimal_price - competitor_avg) / competitor_avg > 0.3:  # Diferencia > 30%
                optimal_price = competitor_avg * 0.95  # Ligeramente más barato
        
        # Calcular demanda esperada al precio óptimo
        if price_elasticity != 0:
            price_change_factor = optimal_price / current_price
            expected_demand = mean_demand * (price_change_factor ** price_elasticity)
        else:
            expected_demand = mean_demand
        
        # Calcular beneficios esperados
        unit_cost = self.cost_structure['costo_unitario']
        profit_per_unit = optimal_price - unit_cost
        total_profit = expected_demand * profit_per_unit
        
        # Análisis de escenarios
        scenarios = self._analyze_pricing_scenarios(
            current_price, mean_demand, price_elasticity
        )
        
        return {
            'product_id': product_id,
            'current_price': current_price,
            'recommended_price': optimal_price,
            'price_change_pct': (optimal_price - current_price) / current_price,
            'expected_demand': expected_demand,
            'expected_profit': total_profit,
            'profit_per_unit': profit_per_unit,
            'price_elasticity': price_elasticity,
            'scenarios': scenarios,
            'competitor_analysis': competitor_prices
        }
    
    def calculate_inventory_metrics(self, 
                                  product_id: str,
                                  demand_predictions: List[PredictionResult],
                                  current_stock: float) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento del inventario
        
        Args:
            product_id: ID del producto
            demand_predictions: Predicciones de demanda
            current_stock: Stock actual
            
        Returns:
            Métricas de inventario
        """
        # Días de inventario restantes
        daily_demand = np.mean([p.predicted_demand for p in demand_predictions])
        days_of_inventory = current_stock / max(daily_demand, 0.1)
        
        # Probabilidad de faltante
        demands = [p.predicted_demand for p in demand_predictions]
        if len(demands) > 0:
            stockout_prob = np.mean([d > current_stock for d in demands])
        else:
            stockout_prob = 0.0
        
        # Costo de oportunidad
        holding_cost_per_unit = self.cost_structure['costo_almacenamiento'] / 30  # Mensual a diario
        daily_holding_cost = current_stock * holding_cost_per_unit
        
        # Valor del inventario
        unit_cost = self.cost_structure['costo_unitario']
        inventory_value = current_stock * unit_cost
        
        return {
            'product_id': product_id,
            'current_stock': current_stock,
            'daily_demand': daily_demand,
            'days_of_inventory': days_of_inventory,
            'stockout_probability': stockout_prob,
            'daily_holding_cost': daily_holding_cost,
            'inventory_value': inventory_value,
            'inventory_turns': (daily_demand * 365) / max(current_stock, 1),
            'service_level': 1 - stockout_prob
        }
    
    def generate_inventory_actions(self, 
                                 product_id: str,
                                 demand_predictions: List[PredictionResult]) -> List[Dict[str, Any]]:
        """
        Genera acciones recomendadas para el inventario
        
        Args:
            product_id: ID del producto
            demand_predictions: Predicciones de demanda
            
        Returns:
            Lista de acciones recomendadas
        """
        actions = []
        
        if product_id not in self.current_inventory:
            return actions
        
        current_info = self.current_inventory[product_id]
        current_stock = current_info['current_stock']
        
        # Calcular demanda promedio
        mean_demand = np.mean([p.predicted_demand for p in demand_predictions])
        days_of_inventory = current_stock / max(mean_demand, 0.1)
        
        # Acciones basadas en días de inventario
        if days_of_inventory < 7:
            # Stock crítico - ordenar urgentemente
            actions.append({
                'action': 'URGENT_ORDER',
                'priority': 'HIGH',
                'description': f'Stock crítico: {days_of_inventory:.1f} días restantes',
                'recommended_quantity': mean_demand * 30,  # 1 mes de stock
                'urgency': 'IMMEDIATE'
            })
        elif days_of_inventory < 14:
            # Stock bajo - ordenar pronto
            actions.append({
                'action': 'REORDER',
                'priority': 'MEDIUM',
                'description': f'Stock bajo: {days_of_inventory:.1f} días restantes',
                'recommended_quantity': mean_demand * 21,  # 3 semanas
                'urgency': 'SOON'
            })
        elif days_of_inventory > 90:
            # Stock excesivo - reducir pedidos
            actions.append({
                'action': 'REDUCE_ORDERS',
                'priority': 'LOW',
                'description': f'Stock excesivo: {days_of_inventory:.1f} días',
                'recommended_action': 'Reducir pedidos durante 2 meses',
                'urgency': 'PLAN'
            })
        
        # Acciones basadas en estacionalidad
        seasonal_actions = self._analyze_seasonal_patterns(demand_predictions)
        actions.extend(seasonal_actions)
        
        # Acciones basadas en variabilidad
        demands = [p.predicted_demand for p in demand_predictions]
        demand_variability = np.std(demands) / max(np.mean(demands), 0.1)
        
        if demand_variability > 0.5:
            actions.append({
                'action': 'INCREASE_SAFETY_STOCK',
                'priority': 'MEDIUM',
                'description': f'Alta variabilidad de demanda (CV: {demand_variability:.2f})',
                'recommended_action': 'Aumentar stock de seguridad en 20%',
                'urgency': 'PLAN'
            })
        
        return actions
    
    def _calculate_inventory_cost(self, 
                                max_inventory: float, 
                                annual_demand: float,
                                eoq: float,
                                safety_stock: float) -> float:
        """Calcula el costo total de inventario"""
        # Costo de almacenamiento
        holding_cost = self.cost_structure['costo_almacenamiento']
        storage_cost = max_inventory * holding_cost
        
        # Costo de pedidos
        ordering_cost = self.cost_structure['costo_pedido']
        order_frequency = annual_demand / max(eoq, 1)
        order_cost = order_frequency * ordering_cost
        
        # Costo de seguridad (almacenamiento extra)
        safety_storage_cost = safety_stock * holding_cost
        
        total_cost = storage_cost + order_cost + safety_storage_cost
        
        return total_cost
    
    def _analyze_pricing_scenarios(self, 
                                 current_price: float, 
                                 base_demand: float,
                                 elasticity: float) -> Dict[str, Any]:
        """Analiza diferentes escenarios de precios"""
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 5)
        scenarios = []
        
        for price in price_range:
            if elasticity != 0:
                price_ratio = price / current_price
                demand = base_demand * (price_ratio ** elasticity)
            else:
                demand = base_demand
            
            unit_cost = self.cost_structure['costo_unitario']
            profit = (price - unit_cost) * demand
            
            scenarios.append({
                'price': price,
                'demand': demand,
                'profit': profit,
                'price_change_pct': (price - current_price) / current_price
            })
        
        # Encontrar escenario óptimo
        best_scenario = max(scenarios, key=lambda x: x['profit'])
        
        return {
            'scenarios': scenarios,
            'optimal_scenario': best_scenario,
            'price_elasticity': elasticity
        }
    
    def _analyze_seasonal_patterns(self, 
                                 demand_predictions: List[PredictionResult]) -> List[Dict[str, Any]]:
        """Analiza patrones estacionales en la demanda"""
        if not demand_predictions:
            return []
        
        # Agrupar por mes
        monthly_demands = {}
        for pred in demand_predictions:
            month = pred.prediction_date.month
            if month not in monthly_demands:
                monthly_demands[month] = []
            monthly_demands[month].append(pred.predicted_demand)
        
        # Calcular promedios mensuales
        monthly_averages = {month: np.mean(demands) 
                          for month, demands in monthly_demands.items()}
        
        # Identificar picos estacionales
        max_month = max(monthly_averages, key=monthly_averages.get)
        min_month = min(monthly_averages, key=monthly_averages.get)
        
        peak_demand = monthly_averages[max_month]
        low_demand = monthly_averages[min_month]
        seasonal_variation = (peak_demand - low_demand) / max(low_demand, 0.1)
        
        actions = []
        
        if seasonal_variation > 0.3:  # Variación > 30%
            actions.append({
                'action': 'SEASONAL_ADJUSTMENT',
                'priority': 'MEDIUM',
                'description': f'Patrón estacional detectado (variación: {seasonal_variation:.1%})',
                'peak_month': max_month,
                'low_month': min_month,
                'recommended_action': f'Ajustar inventario para pico en mes {max_month}',
                'urgency': 'PLAN'
            })
        
        return actions


# Funciones de utilidad y ejemplo de uso
def create_sample_data(n_days: int = 365) -> pd.DataFrame:
    """Crea datos de ejemplo para pruebas"""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Simular demanda con tendencia, estacionalidad y ruido
    trend = np.linspace(100, 150, n_days)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 10, n_days)
    demand = trend + seasonal + noise
    
    # Agregar algunos eventos especiales
    for i in [30, 120, 250]:  # Eventos en días específicos
        if i < n_days:
            demand[i:i+7] += np.random.uniform(30, 50)
    
    return pd.DataFrame({
        'date': dates,
        'demand': np.maximum(demand, 0),  # Demandas no negativas
        'price': np.random.uniform(20, 40, n_days),
        'marketing_spend': np.random.uniform(100, 500, n_days)
    })


def main():
    """Función principal de demostración"""
    print("=== NOESIS Demand Prediction System ===\n")
    
    # Crear datos de ejemplo
    print("1. Generando datos de ejemplo...")
    data = create_sample_data(365)
    print(f"Datos generados: {len(data)} días")
    
    # Inicializar componentes
    print("\n2. Inicializando sistema de predicción...")
    demand_predictor = DemandPredictor()
    marketing_analyzer = MarketingImpactAnalyzer()
    inventory_optimizer = InventoryOptimizer()
    
    # Agregar datos de marketing de ejemplo
    marketing_data = [
        MarketingData(
            campaign_id="CAMP001",
            channel="google_ads",
            budget=5000.0,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            campaign_type="brand_awareness",
            impressions=100000,
            clicks=2000,
            conversions=100,
            ctr=0.02,
            cpc=2.5,
            roas=3.5
        )
    ]
    marketing_analyzer.add_marketing_data(marketing_data)
    
    # Establecer inventario
    inventory_optimizer.set_inventory_level("PROD001", 500, lead_time_days=7)
    
    # Entrenar modelo
    print("\n3. Entrenando modelo...")
    try:
        metrics = demand_predictor.train(
            data, 
            product_id="PROD001", 
            product_type=ProductType.PHYSICAL_PRODUCT
        )
        print("Métricas de entrenamiento:")
        for horizon, metric in metrics.items():
            print(f"  {horizon}: MAE={metric['mae']:.2f}, R2={metric['r2']:.3f}")
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
    
    # Realizar predicciones
    print("\n4. Realizando predicciones...")
    try:
        predictions = demand_predictor.predict(
            data,
            product_id="PROD001",
            product_type=ProductType.PHYSICAL_PRODUCT,
            horizon=PredictionHorizon.SHORT_TERM
        )
        
        print("Predicciones (próximos 7 días):")
        for pred in predictions[:7]:
            print(f"  {pred.prediction_date.strftime('%Y-%m-%d')}: "
                  f"{pred.predicted_demand:.1f} "
                  f"({pred.lower_bound:.1f} - {pred.upper_bound:.1f})")
    except Exception as e:
        print(f"Error en predicción: {e}")
    
    # Optimizar inventario
    print("\n5. Optimizando inventario...")
    try:
        if 'predictions' in locals():
            inventory_rec = inventory_optimizer.calculate_optimal_inventory_level(
                "PROD001", predictions
            )
            print("Recomendación de inventario:")
            print(f"  Punto de reorden: {inventory_rec['reorder_point']:.1f}")
            print(f"  Cantidad económica: {inventory_rec['economic_order_quantity']:.1f}")
            print(f"  Pedido recomendado: {inventory_rec['recommended_order']:.1f}")
    except Exception as e:
        print(f"Error en optimización de inventario: {e}")
    
    # Analizar pricing
    print("\n6. Analizando precios...")
    try:
        if 'predictions' in locals():
            pricing_rec = inventory_optimizer.optimize_pricing(
                "PROD001", predictions, current_price=25.0
            )
            print("Recomendación de precio:")
            print(f"  Precio actual: ${pricing_rec['current_price']:.2f}")
            print(f"  Precio recomendado: ${pricing_rec['recommended_price']:.2f}")
            print(f"  Cambio: {pricing_rec['price_change_pct']:.1%}")
    except Exception as e:
        print(f"Error en análisis de precios: {e}")
    
    # Análisis de marketing
    print("\n7. Analizando impacto de marketing...")
    try:
        campaign_analysis = marketing_analyzer.analyze_campaign_impact(
            "CAMP001", 
            (datetime(2023, 1, 1), datetime(2023, 1, 31))
        )
        print("Análisis de campaña:")
        print(f"  ROAS promedio: {campaign_analysis.get('average_roas', 'N/A')}")
        print(f"  Impacto estimado: {campaign_analysis.get('estimated_demand_impact', 'N/A')}")
    except Exception as e:
        print(f"Error en análisis de marketing: {e}")
    
    print("\n=== Demostración completada ===")


if __name__ == "__main__":
    main()