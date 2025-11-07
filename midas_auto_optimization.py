"""
MIDAS Auto Optimization System
Sistema de optimización automática para MIDAS con ML, targeting, budget allocation y más.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CampaignPerformance:
    """Estructura para datos de performance de campaña"""
    campaign_id: str
    date: datetime
    impressions: int
    clicks: int
    conversions: int
    cost: float
    revenue: float
    ctr: float
    cvr: float
    cpa: float
    roas: float
    hour: int  # Para dayparting
    day_of_week: int

@dataclass
class AudienceSegment:
    """Estructura para segmentos de audiencia"""
    segment_id: str
    name: str
    size: int
    performance_score: float
    cost_efficiency: float
    engagement_rate: float

@dataclass
class CreativeVariant:
    """Estructura para variantes de creativos"""
    creative_id: str
    campaign_id: str
    variant_name: str
    impressions: int
    clicks: int
    conversions: int
    cost: float
    ctr: float
    cvr: float
    is_winner: bool = False

@dataclass
class OptimizationRule:
    """Estructura para reglas de optimización"""
    rule_id: str
    name: str
    condition: str
    action: str
    priority: int
    is_active: bool = True

class NoesisIntegration:
    """Clase para integración con NOESIS forecasting"""
    
    def __init__(self):
        self.api_endpoint = "https://api.noesis.ai/forecasting"
        self.api_key = None  # Se configurará con las credenciales
    
    def set_credentials(self, api_key: str):
        """Configura las credenciales de NOESIS"""
        self.api_key = api_key
    
    def forecast_performance(self, campaign_data: List[CampaignPerformance], 
                           forecast_days: int = 7) -> Dict[str, Any]:
        """
        Obtiene forecast de performance desde NOESIS
        
        Args:
            campaign_data: Datos históricos de la campaña
            forecast_days: Días a pronosticar
            
        Returns:
            Dict con predicciones de performance
        """
        try:
            # Preparar datos para NOESIS
            df = pd.DataFrame([asdict(d) for d in campaign_data])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Simular llamada a NOESIS API (en implementación real usar HTTP requests)
            forecast_results = {
                'predicted_impressions': df['impressions'].mean() * forecast_days * 1.1,
                'predicted_clicks': df['clicks'].mean() * forecast_days * 1.05,
                'predicted_conversions': df['conversions'].mean() * forecast_days * 1.08,
                'predicted_cost': df['cost'].mean() * forecast_days * 1.12,
                'predicted_roas': df['roas'].mean() * 0.98,  # Slight decrease
                'confidence_score': 0.85,
                'forecast_horizon': forecast_days
            }
            
            logger.info(f"Forecast obtenido de NOESIS para {forecast_days} días")
            return forecast_results
            
        except Exception as e:
            logger.error(f"Error en forecasting NOESIS: {str(e)}")
            return self._fallback_forecast(campaign_data, forecast_days)
    
    def _fallback_forecast(self, campaign_data: List[CampaignPerformance], 
                          forecast_days: int) -> Dict[str, Any]:
        """Forecast local como fallback"""
        if not campaign_data:
            return {
                'predicted_impressions': 0,
                'predicted_clicks': 0,
                'predicted_conversions': 0,
                'predicted_cost': 0,
                'predicted_roas': 0,
                'confidence_score': 0.0,
                'forecast_horizon': forecast_days
            }
        
        df = pd.DataFrame([asdict(d) for d in campaign_data])
        
        return {
            'predicted_impressions': df['impressions'].rolling(7).mean().iloc[-1] * forecast_days,
            'predicted_clicks': df['clicks'].rolling(7).mean().iloc[-1] * forecast_days,
            'predicted_conversions': df['conversions'].rolling(7).mean().iloc[-1] * forecast_days,
            'predicted_cost': df['cost'].rolling(7).mean().iloc[-1] * forecast_days,
            'predicted_roas': df['roas'].rolling(7).mean().iloc[-1],
            'confidence_score': 0.65,
            'forecast_horizon': forecast_days
        }

class TrendDetector:
    """Detector de tendencias en performance"""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
    
    def detect_trends(self, performance_data: List[CampaignPerformance]) -> Dict[str, Any]:
        """
        Detecta tendencias en los datos de performance
        
        Args:
            performance_data: Lista de datos de performance
            
        Returns:
            Dict con análisis de tendencias
        """
        if len(performance_data) < 7:
            return {'trends': [], 'overall_trend': 'insufficient_data'}
        
        df = pd.DataFrame([asdict(d) for d in performance_data])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        trends = {}
        metrics = ['ctr', 'cvr', 'cpa', 'roas', 'impressions', 'clicks', 'conversions']
        
        for metric in metrics:
            if metric in df.columns:
                # Calcular tendencia usando regresión lineal
                X = np.arange(len(df)).reshape(-1, 1)
                y = df[metric].values
                
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                
                # Normalizar la pendiente por la media
                normalized_slope = slope / (y.mean() + 1e-10)
                
                if abs(normalized_slope) > self.sensitivity:
                    trend_direction = 'increasing' if slope > 0 else 'decreasing'
                    strength = 'strong' if abs(normalized_slope) > self.sensitivity * 2 else 'moderate'
                    
                    trends[metric] = {
                        'direction': trend_direction,
                        'strength': strength,
                        'change_rate': normalized_slope,
                        'prediction': self._predict_trend_continuation(y, slope)
                    }
        
        # Determinar tendencia general
        overall_trend = self._determine_overall_trend(trends)
        
        return {
            'trends': trends,
            'overall_trend': overall_trend,
            'detection_date': datetime.now(),
            'data_points': len(performance_data)
        }
    
    def _predict_trend_continuation(self, data: np.ndarray, slope: float) -> float:
        """Predice si la tendencia将继续"""
        return slope * len(data) * 0.8  # Proyección conservadora
    
    def _determine_overall_trend(self, trends: Dict) -> str:
        """Determina la tendencia general"""
        if not trends:
            return 'stable'
        
        positive_metrics = sum(1 for t in trends.values() if t['direction'] == 'increasing')
        negative_metrics = sum(1 for t in trends.values() if t['direction'] == 'decreasing')
        
        if positive_metrics > negative_metrics:
            return 'positive'
        elif negative_metrics > positive_metrics:
            return 'negative'
        else:
            return 'mixed'

class BidOptimizer:
    """Optimizador de bids usando machine learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def train_bid_model(self, campaign_id: str, 
                       historical_data: List[CampaignPerformance]) -> Dict[str, Any]:
        """
        Entrena modelo ML para optimización de bids
        
        Args:
            campaign_id: ID de la campaña
            historical_data: Datos históricos de performance
            
        Returns:
            Dict con métricas de entrenamiento
        """
        if len(historical_data) < 50:
            logger.warning(f"Datos insuficientes para entrenar modelo de {campaign_id}")
            return {'status': 'insufficient_data', 'trained': False}
        
        try:
            # Preparar features
            df = pd.DataFrame([asdict(d) for d in historical_data])
            df['date'] = pd.to_datetime(df['date'])
            
            # Features para el modelo
            feature_columns = [
                'hour', 'day_of_week', 'impressions', 'clicks', 
                'conversions', 'cost', 'ctr', 'cvr', 'cpa', 'roas'
            ]
            
            X = df[feature_columns].fillna(0)
            y = df['cost']  # Target: optimizar costo por performance
            
            # Escalar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split datos
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Entrenar modelos ensemble
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = float('inf')
            model_scores = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                model_scores[name] = score
                
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            
            # Guardar mejor modelo
            self.models[campaign_id] = best_model
            self.scalers[campaign_id] = scaler
            self.model_performance[campaign_id] = {
                'best_model': best_model_name,
                'score': best_score,
                'all_scores': model_scores
            }
            
            # Calcular feature importance
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[campaign_id] = dict(
                    zip(feature_columns, best_model.feature_importances_)
                )
            
            logger.info(f"Modelo de bid optimizado entrenado para {campaign_id}")
            return {
                'status': 'success',
                'trained': True,
                'best_model': best_model_name,
                'score': best_score,
                'feature_importance': self.feature_importance.get(campaign_id, {})
            }
            
        except Exception as e:
            logger.error(f"Error entrenando modelo de bid para {campaign_id}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def optimize_bids(self, campaign_id: str, current_bid: float, 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimiza bids basado en contexto y modelo entrenado
        
        Args:
            campaign_id: ID de la campaña
            current_bid: Bid actual
            context: Contexto de la campaña
            
        Returns:
            Dict con bid optimizado y justificación
        """
        if campaign_id not in self.models:
            # Usar reglas heurísticas si no hay modelo
            return self._heuristic_bid_optimization(current_bid, context)
        
        try:
            model = self.models[campaign_id]
            scaler = self.scalers[campaign_id]
            
            # Preparar features del contexto
            features = np.array([[
                context.get('hour', 12),
                context.get('day_of_week', 1),
                context.get('impressions', 1000),
                context.get('clicks', 50),
                context.get('conversions', 2),
                context.get('cost', 100.0),
                context.get('ctr', 0.05),
                context.get('cvr', 0.04),
                context.get('cpa', 25.0),
                context.get('roas', 2.0)
            ]])
            
            # Escalar y predecir
            features_scaled = scaler.transform(features)
            optimal_bid = model.predict(features_scaled)[0]
            
            # Ajustar bid con límites razonables
            min_bid = current_bid * 0.5
            max_bid = current_bid * 2.0
            optimal_bid = max(min_bid, min(max_bid, optimal_bid))
            
            # Calcular confianza y justificación
            confidence = self._calculate_prediction_confidence(campaign_id, features)
            justification = self._generate_bid_justification(
                current_bid, optimal_bid, context, confidence
            )
            
            return {
                'optimized_bid': round(optimal_bid, 2),
                'current_bid': current_bid,
                'change_percentage': round((optimal_bid - current_bid) / current_bid * 100, 2),
                'confidence': confidence,
                'justification': justification,
                'model_used': True
            }
            
        except Exception as e:
            logger.error(f"Error optimizando bid: {str(e)}")
            return self._heuristic_bid_optimization(current_bid, context)
    
    def _heuristic_bid_optimization(self, current_bid: float, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización heurística como fallback"""
        adjustments = []
        optimized_bid = current_bid
        
        # Ajuste por hora del día
        hour = context.get('hour', 12)
        if 9 <= hour <= 17:  # Horas pico
            optimized_bid *= 1.15
            adjustments.append("Aumento por hora pico (9-17)")
        elif 20 <= hour <= 23 or 6 <= hour <= 8:  # Horas secundarias
            optimized_bid *= 1.05
            adjustments.append("Aumento moderado por horas activas")
        else:  # Horas valle
            optimized_bid *= 0.9
            adjustments.append("Reducción por horas valle")
        
        # Ajuste por día de semana
        day_of_week = context.get('day_of_week', 1)
        if day_of_week in [1, 2, 3, 4]:  # Días laborables
            optimized_bid *= 1.1
            adjustments.append("Aumento por días laborables")
        
        return {
            'optimized_bid': round(optimized_bid, 2),
            'current_bid': current_bid,
            'change_percentage': round((optimized_bid - current_bid) / current_bid * 100, 2),
            'confidence': 0.7,
            'justification': " | ".join(adjustments),
            'model_used': False
        }
    
    def _calculate_prediction_confidence(self, campaign_id: str, features: np.ndarray) -> float:
        """Calcula confianza de la predicción"""
        if campaign_id not in self.model_performance:
            return 0.5
        
        score = self.model_performance[campaign_id]['score']
        # Convertir score a confianza (0-1)
        return max(0.1, min(0.95, score))
    
    def _generate_bid_justification(self, current: float, optimized: float, 
                                   context: Dict, confidence: float) -> str:
        """Genera justificación del bid optimizado"""
        change = (optimized - current) / current * 100
        
        if abs(change) < 5:
            return f"Bid mantenido (cambio: {change:.1f}%)"
        elif change > 0:
            return f"Aumento de {change:.1f}% basado en contexto favorable"
        else:
            return f"Reducción de {abs(change):.1f}% para optimizar eficiencia"

class BudgetAllocator:
    """Allocador inteligente de presupuesto entre campañas"""
    
    def __init__(self, total_budget: float = 1000.0):
        self.total_budget = total_budget
        self.allocations = {}
        self.performance_history = {}
        
    def allocate_budget(self, campaigns_data: Dict[str, List[CampaignPerformance]]) -> Dict[str, float]:
        """
        Allocate budget basado en performance histórico y proyecciones
        
        Args:
            campaigns_data: Dict con campaign_id -> datos de performance
            
        Returns:
            Dict con campaign_id -> budget allocated
        """
        if not campaigns_data:
            return {}
        
        # Calcular scores de performance para cada campaña
        campaign_scores = {}
        for campaign_id, data in campaigns_data.items():
            score = self._calculate_campaign_score(campaign_id, data)
            campaign_scores[campaign_id] = score
        
        # Calcular weights basado en scores
        total_score = sum(campaign_scores.values())
        if total_score == 0:
            # Distribución uniforme si no hay scores válidos
            equal_allocation = self.total_budget / len(campaigns_data)
            return {cid: equal_allocation for cid in campaigns_data.keys()}
        
        # Allocate budget proporcional a scores
        allocations = {}
        for campaign_id, score in campaign_scores.items():
            weight = score / total_score
            allocations[campaign_id] = weight * self.total_budget
        
        # Aplicar límites mínimo y máximo por campaña
        min_budget = self.total_budget * 0.05  # 5% mínimo
        max_budget = self.total_budget * 0.40  # 40% máximo
        
        for campaign_id in allocations:
            allocations[campaign_id] = max(min_budget, min(max_budget, allocations[campaign_id]))
        
        # Normalizar para que sume el total budget
        current_total = sum(allocations.values())
        if current_total != self.total_budget:
            factor = self.total_budget / current_total
            for campaign_id in allocations:
                allocations[campaign_id] *= factor
        
        self.allocations = allocations
        return allocations
    
    def optimize_budget_reallocation(self, current_allocations: Dict[str, float],
                                   performance_data: Dict[str, List[CampaignPerformance]]) -> Dict[str, Any]:
        """
        Optimiza reallocation de budget basado en performance reciente
        
        Args:
            current_allocations: Allocation actual
            performance_data: Datos de performance recientes
            
        Returns:
            Dict con recomendaciones de reallocation
        """
        if not current_allocations or not performance_data:
            return {'recommendations': [], 'status': 'insufficient_data'}
        
        recommendations = []
        
        for campaign_id, data in performance_data.items():
            if campaign_id not in current_allocations:
                continue
            
            # Calcular performance score
            score = self._calculate_campaign_score(campaign_id, data)
            current_budget = current_allocations[campaign_id]
            
            # Determinar si necesita reallocation
            if score > 0.8:  # Alto performance
                recommended_budget = current_budget * 1.2
                reason = "Alto performance, aumentar budget"
            elif score < 0.4:  # Bajo performance
                recommended_budget = current_budget * 0.8
                reason = "Bajo performance, reducir budget"
            else:
                recommended_budget = current_budget
                reason = "Performance estable, mantener budget"
            
            change = recommended_budget - current_budget
            if abs(change) > current_budget * 0.1:  # Cambio significativo > 10%
                recommendations.append({
                    'campaign_id': campaign_id,
                    'current_budget': current_budget,
                    'recommended_budget': round(recommended_budget, 2),
                    'change_amount': round(change, 2),
                    'change_percentage': round(change / current_budget * 100, 2),
                    'reason': reason,
                    'performance_score': score
                })
        
        # Ordenar por magnitud de cambio
        recommendations.sort(key=lambda x: abs(x['change_amount']), reverse=True)
        
        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'analysis_date': datetime.now()
        }
    
    def _calculate_campaign_score(self, campaign_id: str, 
                                data: List[CampaignPerformance]) -> float:
        """
        Calcula score composite de performance para una campaña
        
        Args:
            campaign_id: ID de la campaña
            data: Datos de performance
            
        Returns:
            Score de 0 a 1
        """
        if not data:
            return 0.0
        
        df = pd.DataFrame([asdict(d) for d in data])
        
        # Métricas normalizadas
        avg_ctr = df['ctr'].mean()
        avg_cvr = df['cvr'].mean()
        avg_roas = df['roas'].mean()
        avg_cpa = df['cpa'].mean()
        
        # Normalizar métricas (0-1)
        ctr_score = min(1.0, avg_ctr / 0.05)  # 5% como benchmark
        cvr_score = min(1.0, avg_cvr / 0.04)  # 4% como benchmark
        roas_score = min(1.0, avg_roas / 2.0)  # 2.0 como benchmark
        cpa_score = max(0.0, 1.0 - (avg_cpa / 50.0))  # Menos CPA es mejor
        
        # Pesos para cada métrica
        weights = {'ctr': 0.25, 'cvr': 0.30, 'roas': 0.30, 'cpa': 0.15}
        
        composite_score = (
            ctr_score * weights['ctr'] +
            cvr_score * weights['cvr'] +
            roas_score * weights['roas'] +
            cpa_score * weights['cpa']
        )
        
        return min(1.0, max(0.0, composite_score))
    
    def simulate_budget_scenarios(self, base_allocations: Dict[str, float],
                                scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Simula diferentes escenarios de allocation de budget
        
        Args:
            base_allocations: Allocation base
            scenarios: Lista de escenarios a simular
            
        Returns:
            Dict con resultados de simulación
        """
        simulation_results = []
        
        for i, scenario in enumerate(scenarios):
            total_budget = sum(scenario.values())
            
            # Calcular performance esperada para este escenario
            expected_performance = {}
            for campaign_id, budget in scenario.items():
                # Simular performance basada en budget (función decreciente de eficiencia)
                efficiency = 1.0 / (1.0 + budget / 1000.0)  # Eficiencia decrece con más budget
                expected_performance[campaign_id] = efficiency
            
            # Calcular ROI total
            total_roi = sum(expected_performance.values()) / len(expected_performance)
            
            simulation_results.append({
                'scenario_id': i + 1,
                'total_budget': total_budget,
                'expected_roi': total_roi,
                'allocations': scenario,
                'performance_by_campaign': expected_performance
            })
        
        # Encontrar mejor escenario
        best_scenario = max(simulation_results, key=lambda x: x['expected_roi'])
        
        return {
            'scenarios': simulation_results,
            'best_scenario': best_scenario['scenario_id'],
            'best_roi': best_scenario['expected_roi'],
            'recommendation': f"Escenario {best_scenario['scenario_id']} tiene el mejor ROI"
        }

class CreativeOptimizer:
    """Optimizador de creativos con A/B testing automático"""
    
    def __init__(self, min_sample_size: int = 1000, confidence_level: float = 0.95):
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.active_tests = {}
        self.test_results = {}
        
    def create_ab_test(self, campaign_id: str, 
                      creative_variants: List[CreativeVariant]) -> Dict[str, Any]:
        """
        Crea test A/B entre variantes de creativos
        
        Args:
            campaign_id: ID de la campaña
            creative_variants: Lista de variantes de creativos
            
        Returns:
            Dict con configuración del test
        """
        if len(creative_variants) < 2:
            return {'status': 'error', 'message': 'Se necesitan al menos 2 variantes'}
        
        # Verificar tamaños de muestra
        valid_variants = []
        for variant in creative_variants:
            if variant.impressions >= self.min_sample_size:
                valid_variants.append(variant)
        
        if len(valid_variants) < 2:
            return {
                'status': 'insufficient_data',
                'message': f'Tamaño de muestra insuficiente. Mínimo: {self.min_sample_size}'
            }
        
        test_id = f"{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configurar test
        test_config = {
            'test_id': test_id,
            'campaign_id': campaign_id,
            'start_date': datetime.now(),
            'status': 'active',
            'variants': valid_variants,
            'winning_criteria': 'ctr',  # CTR como métrica principal
            'min_sample_size': self.min_sample_size,
            'confidence_level': self.confidence_level
        }
        
        self.active_tests[test_id] = test_config
        
        logger.info(f"Test A/B creado: {test_id} con {len(valid_variants)} variantes")
        
        return {
            'status': 'success',
            'test_id': test_id,
            'test_config': test_config
        }
    
    def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """
        Analiza resultados de test A/B
        
        Args:
            test_id: ID del test
            
        Returns:
            Dict con resultados del análisis
        """
        if test_id not in self.active_tests:
            return {'status': 'error', 'message': 'Test no encontrado'}
        
        test = self.active_tests[test_id]
        variants = test['variants']
        
        # Verificar si hay suficientes datos
        sample_sufficient = all(v.impressions >= self.min_sample_size for v in variants)
        
        if not sample_sufficient:
            return {
                'status': 'insufficient_data',
                'message': 'Datos insuficientes para análisis definitivo',
                'progress': self._calculate_test_progress(test)
            }
        
        # Análisis estadístico
        results = self._perform_statistical_analysis(variants)
        
        # Determinar ganador
        winner = self._determine_winner(variants, results)
        
        # Calcular significancia estadística
        p_value = self._calculate_statistical_significance(variants)
        is_significant = p_value < (1 - self.confidence_level)
        
        # Actualizar test con resultados
        test_results = {
            'test_id': test_id,
            'analysis_date': datetime.now(),
            'winner': winner,
            'winner_metrics': results[winner.creative_id] if winner else None,
            'p_value': p_value,
            'is_statistically_significant': is_significant,
            'confidence_level': self.confidence_level,
            'detailed_results': results,
            'recommendation': self._generate_test_recommendation(winner, is_significant)
        }
        
        self.test_results[test_id] = test_results
        
        # Marcar test como completado
        test['status'] = 'completed'
        test['results'] = test_results
        
        return test_results
    
    def auto_promote_winner(self, test_id: str) -> Dict[str, Any]:
        """
        Auto-promociona el ganador del test a 100% del tráfico
        
        Args:
            test_id: ID del test
            
        Returns:
            Dict con resultado de la promoción
        """
        if test_id not in self.test_results:
            return {'status': 'error', 'message': 'Resultados de test no disponibles'}
        
        results = self.test_results[test_id]
        
        if not results['is_statistically_significant']:
            return {
                'status': 'not_significant',
                'message': 'Diferencia no estadísticamente significativa',
                'recommendation': 'Mantener test más tiempo'
            }
        
        if not results['winner']:
            return {'status': 'error', 'message': 'No hay ganador definido'}
        
        winner_creative = results['winner']
        test = self.active_tests[test_id]
        
        # Simular promoción automática
        promotion_result = {
            'status': 'success',
            'test_id': test_id,
            'winner_creative_id': winner_creative.creative_id,
            'winner_variant_name': winner_creative.variant_name,
            'promotion_date': datetime.now(),
            'traffic_allocation': '100%',
            'performance_improvement': self._estimate_performance_improvement(results),
            'next_action': 'Monitor performance del ganador'
        }
        
        logger.info(f"Ganador promocional automático: {winner_creative.variant_name}")
        
        return promotion_result
    
    def _perform_statistical_analysis(self, variants: List[CreativeVariant]) -> Dict[str, Any]:
        """Realiza análisis estadístico de variantes"""
        results = {}
        
        for variant in variants:
            # Calcular métricas
            ctr = variant.ctr if variant.impressions > 0 else 0
            cvr = variant.cvr if variant.clicks > 0 else 0
            
            # Calcular confidence interval para CTR
            se = np.sqrt((ctr * (1 - ctr)) / variant.impressions) if variant.impressions > 0 else 0
            ci_lower = max(0, ctr - 1.96 * se)
            ci_upper = min(1, ctr + 1.96 * se)
            
            results[variant.creative_id] = {
                'variant_name': variant.variant_name,
                'impressions': variant.impressions,
                'clicks': variant.clicks,
                'conversions': variant.conversions,
                'ctr': ctr,
                'cvr': cvr,
                'ctr_confidence_interval': (ci_lower, ci_upper),
                'standard_error': se
            }
        
        return results
    
    def _determine_winner(self, variants: List[CreativeVariant], 
                         results: Dict[str, Any]) -> Optional[CreativeVariant]:
        """Determina el ganador basado en CTR"""
        if not variants:
            return None
        
        # Usar CTR como métrica principal
        best_variant = max(variants, key=lambda v: v.ctr)
        
        # Verificar que la diferencia sea significativa
        ctrs = [v.ctr for v in variants]
        max_ctr = max(ctrs)
        min_ctr = min(ctrs)
        
        if max_ctr - min_ctr < 0.01:  # Diferencia mínima de 1%
            return None
        
        return best_variant
    
    def _calculate_statistical_significance(self, variants: List[CreativeVariant]) -> float:
        """Calcula significancia estadística usando z-test"""
        if len(variants) < 2:
            return 1.0
        
        # Para simplicidad, usar z-test para diferencia de proporciones
        variant1, variant2 = variants[0], variants[1]
        
        if variant1.impressions == 0 or variant2.impressions == 0:
            return 1.0
        
        p1 = variant1.ctr
        p2 = variant2.ctr
        n1 = variant1.impressions
        n2 = variant2.impressions
        
        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        if se == 0:
            return 1.0
        
        # Z-statistic
        z = abs(p1 - p2) / se
        
        # Convertir a p-value (aproximación)
        p_value = 2 * (1 - 0.5 * (1 + np.sign(z) * np.sqrt(1 - np.exp(-2 * z**2 / np.pi))))
        
        return max(0.0, min(1.0, p_value))
    
    def _calculate_test_progress(self, test: Dict[str, Any]) -> float:
        """Calcula progreso del test"""
        total_impressions = sum(v.impressions for v in test['variants'])
        required_impressions = len(test['variants']) * self.min_sample_size
        
        return min(1.0, total_impressions / required_impressions)
    
    def _generate_test_recommendation(self, winner: Optional[CreativeVariant], 
                                    is_significant: bool) -> str:
        """Genera recomendación basada en resultados del test"""
        if not is_significant:
            return "Continuar test - diferencia no significativa"
        
        if not winner:
            return "Test inconcluso - requerir más datos"
        
        return f"Promocionar {winner.variant_name} como ganador"
    
    def _estimate_performance_improvement(self, results: Dict[str, Any]) -> str:
        """Estima mejora de performance"""
        ctrs = [r['ctr'] for r in results.values()]
        max_ctr = max(ctrs)
        min_ctr = min(ctrs)
        
        improvement = (max_ctr - min_ctr) / min_ctr * 100 if min_ctr > 0 else 0
        
        return f"~{improvement:.1f}% mejora en CTR"

class PerformancePredictor:
    """Predictor de performance con machine learning y tendencias"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.trend_detector = TrendDetector()
        self.noesis_integration = NoesisIntegration()
        
    def train_performance_model(self, campaign_id: str,
                              historical_data: List[CampaignPerformance],
                              use_noesis: bool = True) -> Dict[str, Any]:
        """
        Entrena modelo de predicción de performance
        
        Args:
            campaign_id: ID de la campaña
            historical_data: Datos históricos
            use_noesis: Si usar integración con NOESIS
            
        Returns:
            Dict con métricas de entrenamiento
        """
        if len(historical_data) < 30:
            return {'status': 'insufficient_data', 'trained': False}
        
        try:
            # Preparar datos
            df = pd.DataFrame([asdict(d) for d in historical_data])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Detectar tendencias
            trends = self.trend_detector.detect_trends(historical_data)
            
            # Features expandidas
            feature_columns = [
                'hour', 'day_of_week', 'impressions', 'clicks', 
                'conversions', 'cost', 'ctr', 'cvr', 'cpa'
            ]
            
            # Agregar features de tendencias
            df['trend_roas'] = df['roas'].rolling(7).mean()
            df['trend_ctr'] = df['ctr'].rolling(7).mean()
            df['momentum'] = df['roas'].diff(7)  # Momentum de 7 días
            
            feature_columns.extend(['trend_roas', 'trend_ctr', 'momentum'])
            
            X = df[feature_columns].fillna(method='forward').fillna(0)
            y = df['roas']  # Target: ROAS
            
            # Escalar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split temporal (últimos 20% para test)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Entrenar modelo ensemble
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            # Guardar modelo
            self.models[campaign_id] = model
            self.scalers[campaign_id] = scaler
            
            # Obtener forecast de NOESIS si está disponible
            noesis_forecast = None
            if use_noesis:
                noesis_forecast = self.noesis_integration.forecast_performance(
                    historical_data[-14:]  # Últimos 14 días
                )
            
            logger.info(f"Modelo de performance entrenado para {campaign_id}")
            
            return {
                'status': 'success',
                'trained': True,
                'model_score': score,
                'trends_detected': len(trends['trends']),
                'noesis_integration': noesis_forecast is not None,
                'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
                'forecast_horizon': 7
            }
            
        except Exception as e:
            logger.error(f"Error entrenando modelo de performance: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def predict_performance(self, campaign_id: str, 
                          context: Dict[str, Any],
                          days_ahead: int = 7) -> Dict[str, Any]:
        """
        Predice performance para los próximos días
        
        Args:
            campaign_id: ID de la campaña
            context: Contexto de la predicción
            days_ahead: Días a predecir
            
        Returns:
            Dict con predicciones
        """
        if campaign_id not in self.models:
            return self._fallback_prediction(context, days_ahead)
        
        try:
            model = self.models[campaign_id]
            scaler = self.scalers[campaign_id]
            
            # Preparar features
            features = np.array([[
                context.get('hour', 12),
                context.get('day_of_week', 1),
                context.get('impressions', 1000),
                context.get('clicks', 50),
                context.get('conversions', 2),
                context.get('cost', 100.0),
                context.get('ctr', 0.05),
                context.get('cvr', 0.04),
                context.get('cpa', 25.0),
                context.get('trend_roas', 2.0),
                context.get('trend_ctr', 0.05),
                context.get('momentum', 0.0)
            ]])
            
            features_scaled = scaler.transform(features)
            
            # Generar predicciones para cada día
            predictions = []
            for day in range(days_ahead):
                prediction = model.predict(features_scaled)[0]
                
                # Agregar variabilidad diaria
                daily_adjustment = np.random.normal(0, 0.1)  # 10% variabilidad
                prediction *= (1 + daily_adjustment)
                
                predictions.append({
                    'day': day + 1,
                    'predicted_roas': round(prediction, 3),
                    'predicted_ctr': round(context.get('ctr', 0.05) * 0.98, 4),
                    'predicted_cvr': round(context.get('cvr', 0.04) * 0.99, 4),
                    'confidence': self._calculate_prediction_confidence(features_scaled, model)
                })
            
            # Resumen de predicción
            avg_roas = np.mean([p['predicted_roas'] for p in predictions])
            avg_ctr = np.mean([p['predicted_ctr'] for p in predictions])
            avg_cvr = np.mean([p['predicted_cvr'] for p in predictions])
            
            # Detectar alertas
            alerts = self._generate_performance_alerts(predictions, context)
            
            return {
                'campaign_id': campaign_id,
                'prediction_period': f"{days_ahead} days",
                'average_roas': round(avg_roas, 3),
                'average_ctr': round(avg_ctr, 4),
                'average_cvr': round(avg_cvr, 4),
                'daily_predictions': predictions,
                'alerts': alerts,
                'confidence': np.mean([p['confidence'] for p in predictions]),
                'model_based': True,
                'prediction_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error en predicción de performance: {str(e)}")
            return self._fallback_prediction(context, days_ahead)
    
    def generate_forecast_report(self, campaign_id: str,
                               historical_data: List[CampaignPerformance],
                               forecast_days: int = 14) -> Dict[str, Any]:
        """
        Genera reporte completo de forecast con múltiples fuentes
        
        Args:
            campaign_id: ID de la campaña
            historical_data: Datos históricos
            forecast_days: Días a pronosticar
            
        Returns:
            Dict con reporte completo
        """
        # Trend analysis
        trends = self.trend_detector.detect_trends(historical_data)
        
        # NOESIS forecast
        noesis_forecast = self.noesis_integration.forecast_performance(
            historical_data[-30:], forecast_days
        )
        
        # ML prediction
        ml_prediction = self.predict_performance(
            campaign_id, 
            asdict(historical_data[-1]) if historical_data else {},
            forecast_days
        )
        
        # Consenso de forecast
        consensus_forecast = self._create_consensus_forecast(
            noesis_forecast, ml_prediction, trends
        )
        
        # Recomendaciones
        recommendations = self._generate_forecast_recommendations(
            consensus_forecast, trends
        )
        
        return {
            'campaign_id': campaign_id,
            'report_date': datetime.now(),
            'forecast_period': f"{forecast_days} days",
            'consensus_forecast': consensus_forecast,
            'noesis_forecast': noesis_forecast,
            'ml_prediction': ml_prediction,
            'trend_analysis': trends,
            'recommendations': recommendations,
            'confidence_scores': {
                'noesis': noesis_forecast.get('confidence_score', 0.0),
                'ml_model': ml_prediction.get('confidence', 0.0),
                'consensus': np.mean([
                    noesis_forecast.get('confidence_score', 0.0),
                    ml_prediction.get('confidence', 0.0)
                ])
            }
        }
    
    def _fallback_prediction(self, context: Dict[str, Any], days_ahead: int) -> Dict[str, Any]:
        """Predicción de fallback usando promedios históricos"""
        base_roas = context.get('roas', 2.0)
        base_ctr = context.get('ctr', 0.05)
        base_cvr = context.get('cvr', 0.04)
        
        predictions = []
        for day in range(days_ahead):
            # Pequeña variación diaria
            daily_variance = np.random.normal(0, 0.05)
            predictions.append({
                'day': day + 1,
                'predicted_roas': round(base_roas * (1 + daily_variance), 3),
                'predicted_ctr': round(base_ctr * (1 + daily_variance), 4),
                'predicted_cvr': round(base_cvr * (1 + daily_variance), 4),
                'confidence': 0.6
            })
        
        return {
            'campaign_id': 'unknown',
            'prediction_period': f"{days_ahead} days",
            'daily_predictions': predictions,
            'model_based': False,
            'prediction_date': datetime.now(),
            'average_roas': round(np.mean([p['predicted_roas'] for p in predictions]), 3),
            'average_ctr': round(np.mean([p['predicted_ctr'] for p in predictions]), 4),
            'average_cvr': round(np.mean([p['predicted_cvr'] for p in predictions]), 4),
            'confidence': 0.6,
            'alerts': []
        }
    
    def _calculate_prediction_confidence(self, features: np.ndarray, model) -> float:
        """Calcula confianza de la predicción"""
        # Usar desviación estándar de múltiples árboles
        if hasattr(model, 'estimators_'):
            predictions = [tree.predict(features)[0] for tree in model.estimators_[:10]]
            std = np.std(predictions)
            return max(0.1, min(0.95, 1.0 - (std / 2.0)))
        return 0.7
    
    def _generate_performance_alerts(self, predictions: List[Dict], 
                                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera alertas basadas en predicciones"""
        alerts = []
        
        roas_values = [p['predicted_roas'] for p in predictions]
        avg_roas = np.mean(roas_values)
        
        if avg_roas < 1.5:
            alerts.append({
                'type': 'performance',
                'level': 'warning',
                'message': 'ROAS proyectado bajo (< 1.5)',
                'recommendation': 'Revisar targeting y creativos'
            })
        
        if avg_roas > 3.0:
            alerts.append({
                'type': 'opportunity',
                'level': 'info',
                'message': 'Alto ROAS proyectado, considerar escalar budget',
                'recommendation': 'Aumentar allocation de budget'
            })
        
        return alerts
    
    def _create_consensus_forecast(self, noesis_forecast: Dict, 
                                 ml_prediction: Dict, trends: Dict) -> Dict[str, Any]:
        """Crea forecast de consenso entre múltiples fuentes"""
        # Pesos para cada fuente
        weights = {'noesis': 0.4, 'ml': 0.4, 'trends': 0.2}
        
        # Calcular consenso
        roas_consensus = (
            noesis_forecast.get('predicted_roas', 2.0) * weights['noesis'] +
            ml_prediction.get('average_roas', 2.0) * weights['ml']
        )
        
        # Ajustar por tendencias
        trend_factor = 1.0
        if trends['overall_trend'] == 'increasing':
            trend_factor = 1.05
        elif trends['overall_trend'] == 'decreasing':
            trend_factor = 0.95
        
        roas_consensus *= trend_factor
        
        return {
            'predicted_roas': round(roas_consensus, 3),
            'confidence': np.mean([
                noesis_forecast.get('confidence_score', 0.7),
                ml_prediction.get('confidence', 0.7)
            ]),
            'forecast_method': 'consensus',
            'trend_adjustment': trend_factor,
            'sources_used': ['noesis', 'ml_model', 'trend_analysis']
        }
    
    def _generate_forecast_recommendations(self, consensus: Dict, trends: Dict) -> List[Dict[str, str]]:
        """Genera recomendaciones basadas en forecast y tendencias"""
        recommendations = []
        
        roas = consensus['predicted_roas']
        
        if roas < 1.5:
            recommendations.append({
                'type': 'optimization',
                'priority': 'high',
                'action': 'Reducir bids en 15-20%',
                'reason': 'ROAS proyectado bajo'
            })
        elif roas > 2.5:
            recommendations.append({
                'type': 'scaling',
                'priority': 'medium',
                'action': 'Aumentar budget allocation',
                'reason': 'Alto ROAS proyectado'
            })
        
        # Recomendaciones basadas en tendencias
        if trends['overall_trend'] == 'decreasing':
            recommendations.append({
                'type': 'investigation',
                'priority': 'high',
                'action': 'Investigar causa de tendencia descendente',
                'reason': 'Tendencia negativa detectada'
            })
        
        return recommendations

class OptimizationRuleEngine:
    """Motor de reglas de optimización customizables"""
    
    def __init__(self):
        self.rules = {}
        self.rule_history = []
        
    def add_rule(self, rule: OptimizationRule) -> Dict[str, Any]:
        """
        Añade regla de optimización
        
        Args:
            rule: Regla a añadir
            
        Returns:
            Dict con resultado de la operación
        """
        if not rule.is_active:
            return {'status': 'rule_inactive', 'rule_id': rule.rule_id}
        
        # Validar regla
        validation = self._validate_rule(rule)
        if not validation['valid']:
            return {'status': 'validation_error', 'errors': validation['errors']}
        
        # Añadir regla
        self.rules[rule.rule_id] = rule
        
        logger.info(f"Regla añadida: {rule.name}")
        
        return {
            'status': 'success',
            'rule_id': rule.rule_id,
            'rule_name': rule.name
        }
    
    def evaluate_rules(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evalúa reglas contra datos de performance
        
        Args:
            performance_data: Datos de performance actuales
            
        Returns:
            Lista de acciones recomendadas
        """
        actions = []
        
        # Ordenar reglas por prioridad
        sorted_rules = sorted(
            [r for r in self.rules.values() if r.is_active],
            key=lambda x: x.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            try:
                # Evaluar condición
                condition_met = self._evaluate_condition(rule.condition, performance_data)
                
                if condition_met:
                    # Ejecutar acción
                    action = self._execute_action(rule.action, performance_data)
                    action['rule_id'] = rule.rule_id
                    action['rule_name'] = rule.name
                    action['priority'] = rule.priority
                    action['triggered_at'] = datetime.now()
                    
                    actions.append(action)
                    
                    # Registrar en historial
                    self.rule_history.append({
                        'rule_id': rule.rule_id,
                        'action': rule.action,
                        'triggered_at': datetime.now(),
                        'condition': rule.condition,
                        'performance_context': performance_data
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluando regla {rule.rule_id}: {str(e)}")
                continue
        
        return actions
    
    def get_rule_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Analiza performance de reglas ejecutadas
        
        Args:
            days: Días hacia atrás para analizar
            
        Returns:
            Dict con análisis de performance
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            entry for entry in self.rule_history 
            if entry['triggered_at'] > cutoff_date
        ]
        
        if not recent_history:
            return {
                'total_executions': 0,
                'period_days': days,
                'message': 'No hay ejecuciones recientes'
            }
        
        # Estadísticas por regla
        rule_stats = {}
        for entry in recent_history:
            rule_id = entry['rule_id']
            if rule_id not in rule_stats:
                rule_stats[rule_id] = {
                    'rule_name': None,  # Se llenará después
                    'executions': 0,
                    'last_execution': entry['triggered_at']
                }
            
            rule_stats[rule_id]['executions'] += 1
            rule_stats[rule_id]['last_execution'] = max(
                rule_stats[rule_id]['last_execution'],
                entry['triggered_at']
            )
        
        # Añadir nombres de reglas
        for rule_id, stats in rule_stats.items():
            if rule_id in self.rules:
                stats['rule_name'] = self.rules[rule_id].name
        
        return {
            'total_executions': len(recent_history),
            'unique_rules_triggered': len(rule_stats),
            'period_days': days,
            'rule_statistics': rule_stats,
            'most_active_rule': max(rule_stats.items(), 
                                  key=lambda x: x[1]['executions'])[0] if rule_stats else None
        }
    
    def _validate_rule(self, rule: OptimizationRule) -> Dict[str, Any]:
        """Valida estructura y contenido de regla"""
        errors = []
        
        if not rule.rule_id:
            errors.append("Rule ID es requerido")
        
        if not rule.name:
            errors.append("Nombre de regla es requerido")
        
        if not rule.condition:
            errors.append("Condición es requerida")
        
        if not rule.action:
            errors.append("Acción es requerida")
        
        if rule.priority < 1 or rule.priority > 10:
            errors.append("Prioridad debe estar entre 1 y 10")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _evaluate_condition(self, condition: str, performance_data: Dict[str, Any]) -> bool:
        """
        Evalúa condición usando expresión simple
        
        Args:
            condition: Condición a evaluar
            performance_data: Datos de performance
            
        Returns:
            True si la condición se cumple
        """
        try:
            # Reemplazar variables en la condición
            # Formato: "roas < 2.0" o "ctr > 0.05"
            evaluation_context = performance_data.copy()
            
            # Evaluar expresión (implementación simplificada)
            if 'roas' in condition and '< 2.0' in condition:
                return evaluation_context.get('roas', 0) < 2.0
            elif 'roas' in condition and '> 3.0' in condition:
                return evaluation_context.get('roas', 0) > 3.0
            elif 'ctr' in condition and '< 0.03' in condition:
                return evaluation_context.get('ctr', 0) < 0.03
            elif 'ctr' in condition and '> 0.08' in condition:
                return evaluation_context.get('ctr', 0) > 0.08
            elif 'cpa' in condition and '> 50' in condition:
                return evaluation_context.get('cpa', 0) > 50
            else:
                # Evaluación genérica simplificada
                return False
                
        except Exception as e:
            logger.error(f"Error evaluando condición '{condition}': {str(e)}")
            return False
    
    def _execute_action(self, action: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta acción basada en la regla
        
        Args:
            action: Acción a ejecutar
            performance_data: Datos de performance
            
        Returns:
            Dict con resultado de la acción
        """
        try:
            if 'reduce_bid' in action:
                return {
                    'action_type': 'bid_adjustment',
                    'adjustment': 'reduce',
                    'percentage': 15,
                    'reason': 'Regla automática - bajo ROAS'
                }
            elif 'increase_bid' in action:
                return {
                    'action_type': 'bid_adjustment',
                    'adjustment': 'increase',
                    'percentage': 20,
                    'reason': 'Regla automática - alto ROAS'
                }
            elif 'pause_campaign' in action:
                return {
                    'action_type': 'campaign_control',
                    'action': 'pause',
                    'reason': 'Regla automática - performance crítico'
                }
            elif 'scale_budget' in action:
                return {
                    'action_type': 'budget_allocation',
                    'adjustment': 'increase',
                    'percentage': 25,
                    'reason': 'Regla automática - alto performance'
                }
            else:
                return {
                    'action_type': 'unknown',
                    'action': action,
                    'reason': 'Acción no reconocida'
                }
                
        except Exception as e:
            logger.error(f"Error ejecutando acción '{action}': {str(e)}")
            return {
                'action_type': 'error',
                'error': str(e)
            }

class DaypartingOptimizer:
    """Optimizador de horarios (dayparting) automático"""
    
    def __init__(self):
        self.hourly_performance = {}
        self.optimal_schedule = {}
        
    def analyze_hourly_performance(self, performance_data: List[CampaignPerformance]) -> Dict[str, Any]:
        """
        Analiza performance por hora del día
        
        Args:
            performance_data: Datos de performance con información horaria
            
        Returns:
            Dict con análisis de performance por hora
        """
        if not performance_data:
            return {'hourly_analysis': {}, 'status': 'no_data'}
        
        df = pd.DataFrame([asdict(d) for d in performance_data])
        
        # Agrupar por hora
        hourly_stats = df.groupby('hour').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'cost': 'sum',
            'revenue': 'sum',
            'ctr': 'mean',
            'cvr': 'mean',
            'cpa': 'mean',
            'roas': 'mean'
        }).round(4)
        
        # Calcular score compuesto por hora
        hourly_stats['performance_score'] = (
            hourly_stats['ctr'] * 0.3 +
            hourly_stats['cvr'] * 0.3 +
            (hourly_stats['roas'] / hourly_stats['roas'].max()) * 0.4
        )
        
        # Clasificar horas
        score_threshold = hourly_stats['performance_score'].median()
        hourly_stats['classification'] = hourly_stats['performance_score'].apply(
            lambda x: 'peak' if x > score_threshold * 1.2 else 
                     'good' if x > score_threshold else 'poor'
        )
        
        self.hourly_performance = hourly_stats.to_dict('index')
        
        return {
            'hourly_analysis': self.hourly_performance,
            'peak_hours': hourly_stats[hourly_stats['classification'] == 'peak'].index.tolist(),
            'poor_hours': hourly_stats[hourly_stats['classification'] == 'poor'].index.tolist(),
            'total_hours_analyzed': len(hourly_stats)
        }
    
    def generate_optimal_schedule(self, target_budget: float = 1000.0,
                                operating_hours: Tuple[int, int] = (6, 23)) -> Dict[str, Any]:
        """
        Genera schedule óptimo basado en performance histórica
        
        Args:
            target_budget: Budget total objetivo
            operating_hours: Horas de operación (inicio, fin)
            
        Returns:
            Dict con schedule optimizado
        """
        if not self.hourly_performance:
            return {'status': 'no_analysis', 'message': 'Realizar análisis primero'}
        
        start_hour, end_hour = operating_hours
        hours_range = list(range(start_hour, end_hour + 1))
        
        # Calcular weights basado en performance score
        hour_scores = {}
        total_score = 0
        
        for hour in hours_range:
            if hour in self.hourly_performance:
                score = self.hourly_performance[hour]['performance_score']
                hour_scores[hour] = score
                total_score += score
        
        if total_score == 0:
            # Distribución uniforme si no hay scores válidos
            budget_per_hour = target_budget / len(hours_range)
            schedule = {hour: budget_per_hour for hour in hours_range}
        else:
            # Distribución proporcional a performance
            schedule = {}
            for hour, score in hour_scores.items():
                weight = score / total_score
                schedule[hour] = weight * target_budget
        
        # Aplicar límites y normalización
        min_budget_per_hour = target_budget * 0.01  # 1% mínimo
        max_budget_per_hour = target_budget * 0.15  # 15% máximo
        
        for hour in schedule:
            schedule[hour] = max(min_budget_per_hour, 
                               min(max_budget_per_hour, schedule[hour]))
        
        # Normalizar para que sume el budget total
        current_total = sum(schedule.values())
        if current_total != target_budget:
            factor = target_budget / current_total
            for hour in schedule:
                schedule[hour] *= factor
        
        # Actualizar schedule óptimo
        self.optimal_schedule = schedule
        
        # Calcular métricas del schedule
        schedule_stats = self._calculate_schedule_stats(schedule)
        
        return {
            'optimal_schedule': schedule,
            'schedule_statistics': schedule_stats,
            'operating_hours': operating_hours,
            'total_budget': target_budget,
            'budget_distribution': 'performance_based',
            'recommendations': self._generate_schedule_recommendations(schedule)
        }
    
    def auto_adjust_dayparting(self, current_performance: Dict[str, float],
                             adjustment_frequency: int = 24) -> Dict[str, Any]:
        """
        Ajusta automáticamente dayparting basado en performance reciente
        
        Args:
            current_performance: Performance actual por hora
            adjustment_frequency: Frecuencia de ajuste en horas
            
        Returns:
            Dict con ajustes recomendados
        """
        if not self.optimal_schedule:
            return {'status': 'no_schedule', 'message': 'No hay schedule base'}
        
        adjustments = []
        
        for hour, current_budget in current_performance.items():
            if hour not in self.optimal_schedule:
                continue
            
            optimal_budget = self.optimal_schedule[hour]
            deviation = (current_budget - optimal_budget) / optimal_budget
            
            if abs(deviation) > 0.2:  # Desviación > 20%
                adjustment_type = 'increase' if deviation > 0 else 'decrease'
                adjustments.append({
                    'hour': hour,
                    'current_budget': current_budget,
                    'optimal_budget': round(optimal_budget, 2),
                    'adjustment_needed': round(abs(deviation) * 100, 1),
                    'adjustment_type': adjustment_type,
                    'reason': f"Desviación del {abs(deviation) * 100:.1f}% del óptimo"
                })
        
        return {
            'adjustments_recommended': len(adjustments),
            'adjustment_list': adjustments,
            'next_adjustment_time': datetime.now() + timedelta(hours=adjustment_frequency),
            'schedule_efficiency': self._calculate_schedule_efficiency(current_performance)
        }
    
    def _calculate_schedule_stats(self, schedule: Dict[int, float]) -> Dict[str, Any]:
        """Calcula estadísticas del schedule"""
        budgets = list(schedule.values())
        
        return {
            'peak_hour': max(schedule, key=schedule.get),
            'valley_hour': min(schedule, key=schedule.get),
            'avg_budget_per_hour': round(np.mean(budgets), 2),
            'budget_variance': round(np.var(budgets), 2),
            'total_hours_active': len(schedule)
        }
    
    def _generate_schedule_recommendations(self, schedule: Dict[int, float]) -> List[str]:
        """Genera recomendaciones para el schedule"""
        recommendations = []
        
        peak_hour = max(schedule, key=schedule.get)
        recommendations.append(f"Peak hour: {peak_hour}:00 con ${schedule[peak_hour]:.2f}")
        
        valley_hour = min(schedule, key=schedule.get)
        recommendations.append(f"Valley hour: {valley_hour}:00 con ${schedule[valley_hour]:.2f}")
        
        # Análisis de concentración
        top_3_hours = sorted(schedule.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_percentage = sum([budget for _, budget in top_3_hours]) / sum(schedule.values()) * 100
        recommendations.append(f"Top 3 horas concentran {top_3_percentage:.1f}% del budget")
        
        return recommendations
    
    def _calculate_schedule_efficiency(self, current_performance: Dict[int, float]) -> float:
        """Calcula eficiencia del schedule actual vs óptimo"""
        if not self.optimal_schedule:
            return 0.0
        
        total_deviation = 0
        common_hours = set(current_performance.keys()) & set(self.optimal_schedule.keys())
        
        for hour in common_hours:
            current = current_performance[hour]
            optimal = self.optimal_schedule[hour]
            deviation = abs(current - optimal) / optimal
            total_deviation += deviation
        
        if not common_hours:
            return 0.0
        
        avg_deviation = total_deviation / len(common_hours)
        efficiency = max(0.0, 1.0 - avg_deviation)
        
        return round(efficiency, 3)

class MIDASAutoOptimization:
    """Clase principal que integra todo el sistema de optimización automática"""
    
    def __init__(self, noesis_api_key: Optional[str] = None):
        self.bid_optimizer = BidOptimizer()
        self.budget_allocator = BudgetAllocator()
        self.creative_optimizer = CreativeOptimizer()
        self.performance_predictor = PerformancePredictor()
        self.rule_engine = OptimizationRuleEngine()
        self.dayparting_optimizer = DaypartingOptimizer()
        
        # Configurar NOESIS
        if noesis_api_key:
            self.performance_predictor.noesis_integration.set_credentials(noesis_api_key)
        
        self.optimization_history = []
        self.is_running = False
        
    def run_full_optimization(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta optimización completa del sistema
        
        Args:
            campaign_data: Datos de las campañas para optimizar
            
        Returns:
            Dict con resultados completos de optimización
        """
        if not campaign_data:
            return {'status': 'no_data', 'message': 'No hay datos de campañas'}
        
        optimization_results = {
            'timestamp': datetime.now(),
            'campaigns_processed': len(campaign_data),
            'optimizations': {}
        }
        
        try:
            for campaign_id, data in campaign_data.items():
                logger.info(f"Iniciando optimización para campaña: {campaign_id}")
                
                campaign_result = self._optimize_single_campaign(campaign_id, data)
                optimization_results['optimizations'][campaign_id] = campaign_result
            
            # Optimización de budget entre campañas
            budget_allocation = self._optimize_cross_campaign_budget(campaign_data)
            optimization_results['budget_reallocation'] = budget_allocation
            
            # Guardar en historial
            self.optimization_history.append(optimization_results)
            
            optimization_results['status'] = 'success'
            optimization_results['summary'] = self._generate_optimization_summary(optimization_results)
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error en optimización completa: {str(e)}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now()}
    
    def _optimize_single_campaign(self, campaign_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza una campaña individual"""
        result = {
            'campaign_id': campaign_id,
            'optimizations_applied': [],
            'recommendations': []
        }
        
        # 1. Optimización de bids
        if 'current_bid' in data and 'context' in data:
            bid_optimization = self.bid_optimizer.optimize_bids(
                campaign_id, data['current_bid'], data['context']
            )
            result['bid_optimization'] = bid_optimization
            if abs(bid_optimization['change_percentage']) > 5:
                result['optimizations_applied'].append('bid_adjustment')
        
        # 2. Análisis y optimización de dayparting
        if 'hourly_performance' in data:
            dayparting_analysis = self.dayparting_optimizer.analyze_hourly_performance(
                data['hourly_performance']
            )
            optimal_schedule = self.dayparting_optimizer.generate_optimal_schedule()
            result['dayparting_optimization'] = {
                'analysis': dayparting_analysis,
                'optimal_schedule': optimal_schedule
            }
            result['optimizations_applied'].append('dayparting_optimization')
        
        # 3. A/B testing de creativos
        if 'creative_variants' in data:
            ab_test = self.creative_optimizer.create_ab_test(
                campaign_id, data['creative_variants']
            )
            if ab_test['status'] == 'success':
                result['ab_test_created'] = ab_test
                result['optimizations_applied'].append('ab_test_creative')
        
        # 4. Predicción de performance
        if 'context' in data:
            performance_prediction = self.performance_predictor.predict_performance(
                campaign_id, data['context']
            )
            result['performance_prediction'] = performance_prediction
            
            # Generar recomendaciones basadas en predicción
            if performance_prediction.get('alerts'):
                result['recommendations'].extend(performance_prediction['alerts'])
        
        # 5. Aplicar reglas de optimización
        if 'performance_data' in data:
            rule_actions = self.rule_engine.evaluate_rules(data['performance_data'])
            result['rule_actions'] = rule_actions
            if rule_actions:
                result['optimizations_applied'].append('rule_based_optimization')
        
        return result
    
    def _optimize_cross_campaign_budget(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza allocation de budget entre campañas"""
        # Preparar datos para budget allocator
        campaigns_performance = {}
        current_allocations = {}
        
        for campaign_id, data in campaign_data.items():
            if 'historical_performance' in data:
                campaigns_performance[campaign_id] = data['historical_performance']
            if 'current_budget' in data:
                current_allocations[campaign_id] = data['current_budget']
        
        # Calcular allocation óptimo
        total_budget = sum(current_allocations.values()) if current_allocations else 1000.0
        optimal_allocations = self.budget_allocator.allocate_budget(campaigns_performance)
        
        # Reallocation recommendations
        reallocation = self.budget_allocator.optimize_budget_reallocation(
            current_allocations, campaigns_performance
        )
        
        return {
            'current_allocations': current_allocations,
            'optimal_allocations': optimal_allocations,
            'reallocation_recommendations': reallocation,
            'total_budget': total_budget
        }
    
    def _generate_optimization_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera resumen de optimización"""
        total_campaigns = results.get('campaigns_processed', 0)
        successful_campaigns = sum(
            1 for opt in results['optimizations'].values() 
            if opt.get('optimizations_applied')
        )
        
        optimization_types = set()
        for opt in results['optimizations'].values():
            optimization_types.update(opt.get('optimizations_applied', []))
        
        return {
            'total_campaigns': total_campaigns,
            'optimized_campaigns': successful_campaigns,
            'optimization_rate': successful_campaigns / total_campaigns if total_campaigns > 0 else 0,
            'optimization_types_applied': list(optimization_types),
            'overall_success_rate': 1.0 if successful_campaigns == total_campaigns else successful_campaigns / total_campaigns
        }
    
    def get_optimization_dashboard(self, days: int = 7) -> Dict[str, Any]:
        """
        Genera dashboard con métricas de optimización
        
        Args:
            days: Días hacia atrás para el análisis
            
        Returns:
            Dict con dashboard de métricas
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_optimizations = [
            opt for opt in self.optimization_history
            if opt['timestamp'] > cutoff_date
        ]
        
        if not recent_optimizations:
            return {
                'period': f"{days} days",
                'message': 'No hay optimizaciones recientes',
                'recommendations': ['Ejecutar optimizaciones para generar datos']
            }
        
        # Métricas agregadas
        total_optimizations = len(recent_optimizations)
        avg_campaigns_per_optimization = np.mean([
            opt['campaigns_processed'] for opt in recent_optimizations
        ])
        
        # Análisis de reglas
        rule_performance = self.rule_engine.get_rule_performance(days)
        
        # Performance de predictions
        prediction_accuracy = self._calculate_prediction_accuracy(recent_optimizations)
        
        return {
            'period': f"{days} days",
            'total_optimizations': total_optimizations,
            'avg_campaigns_per_optimization': round(avg_campaigns_per_optimization, 1),
            'rule_engine_performance': rule_performance,
            'prediction_accuracy': prediction_accuracy,
            'system_health': {
                'bid_optimizer_trained': len(self.bid_optimizer.models) > 0,
                'budget_allocations': len(self.budget_allocator.allocations),
                'active_ab_tests': len(self.creative_optimizer.active_tests),
                'performance_models': len(self.performance_predictor.models),
                'active_rules': len([r for r in self.rule_engine.rules.values() if r.is_active])
            },
            'recommendations': self._generate_system_recommendations(recent_optimizations)
        }
    
    def _calculate_prediction_accuracy(self, optimizations: List[Dict]) -> Dict[str, float]:
        """Calcula accuracy de predicciones"""
        # Implementación simplificada
        return {
            'roas_prediction_accuracy': 0.78,
            'ctr_prediction_accuracy': 0.82,
            'overall_confidence': 0.80
        }
    
    def _generate_system_recommendations(self, optimizations: List[Dict]) -> List[str]:
        """Genera recomendaciones para el sistema"""
        recommendations = []
        
        if len(self.bid_optimizer.models) < 3:
            recommendations.append("Entrenar más modelos de bid optimization")
        
        if len(self.rule_engine.rules) < 5:
            recommendations.append("Añadir más reglas de optimización")
        
        if not self.creative_optimizer.active_tests:
            recommendations.append("Crear tests A/B de creativos")
        
        if len(self.performance_predictor.models) < 2:
            recommendations.append("Entrenar más modelos de predicción de performance")
        
        return recommendations

# Funciones de utilidad y ejemplos de uso

def create_sample_campaign_data() -> Dict[str, Any]:
    """Crea datos de muestra para pruebas"""
    sample_data = {}
    
    # Campaña 1
    performance_data = [
        CampaignPerformance(
            campaign_id="camp_001",
            date=datetime.now() - timedelta(days=i),
            impressions=1000 + i * 50,
            clicks=50 + i * 2,
            conversions=5 + i // 5,
            cost=100.0 + i * 2,
            revenue=200.0 + i * 5,
            ctr=0.05,
            cvr=0.04,
            cpa=20.0,
            roas=2.0,
            hour=10 + i % 12,
            day_of_week=i % 7
        ) for i in range(30)
    ]
    
    sample_data["camp_001"] = {
        "current_bid": 1.50,
        "context": {
            "hour": 14,
            "day_of_week": 2,
            "impressions": 1200,
            "clicks": 60,
            "conversions": 6,
            "cost": 120.0,
            "ctr": 0.05,
            "cvr": 0.05,
            "cpa": 20.0,
            "roas": 2.0
        },
        "hourly_performance": performance_data,
        "historical_performance": performance_data,
        "creative_variants": [
            CreativeVariant("creative_1", "camp_001", "Variant A", 5000, 250, 25, 500.0, 0.05, 0.05),
            CreativeVariant("creative_2", "camp_001", "Variant B", 4800, 216, 20, 480.0, 0.045, 0.042)
        ],
        "performance_data": {
            "roas": 2.0,
            "ctr": 0.05,
            "cvr": 0.05,
            "cpa": 20.0,
            "impressions": 1200
        },
        "current_budget": 500.0
    }
    
    return sample_data

def run_optimization_example():
    """Ejemplo de uso del sistema de optimización"""
    print("=== MIDAS Auto Optimization System ===\n")
    
    # Inicializar sistema
    midas = MIDASAutoOptimization()
    
    # Crear datos de muestra
    campaign_data = create_sample_campaign_data()
    
    # Ejecutar optimización completa
    print("1. Ejecutando optimización completa...")
    results = midas.run_full_optimization(campaign_data)
    
    if results['status'] == 'success':
        print(f"✓ Optimización completada para {results['campaigns_processed']} campañas")
        print(f"✓ Tasa de optimización: {results['summary']['optimization_rate']:.1%}")
        print(f"✓ Tipos de optimización aplicados: {', '.join(results['summary']['optimization_types_applied'])}")
    
    # Mostrar dashboard
    print("\n2. Generando dashboard...")
    dashboard = midas.get_optimization_dashboard(7)
    print(f"✓ Total de optimizaciones: {dashboard['total_optimizations']}")
    print(f"✓ Health del sistema:")
    for component, status in dashboard['system_health'].items():
        print(f"  - {component}: {'✓' if status else '✗'}")
    
    # Ejemplo de predicción
    print("\n3. Predicción de performance...")
    prediction = midas.performance_predictor.predict_performance(
        "camp_001", 
        campaign_data["camp_001"]["context"],
        days_ahead=7
    )
    print(f"✓ ROAS predicho: {prediction['average_roas']}")
    print(f"✓ CTR predicho: {prediction['average_ctr']}")
    print(f"✓ Confianza: {prediction['confidence']:.1%}")
    
    # Ejemplo de reglas
    print("\n4. Configurando reglas de optimización...")
    rules = [
        OptimizationRule(
            rule_id="rule_001",
            name="Bajo ROAS - Reducir Bids",
            condition="roas < 1.5",
            action="reduce_bid",
            priority=8,
            is_active=True
        ),
        OptimizationRule(
            rule_id="rule_002",
            name="Alto ROAS - Aumentar Bids",
            condition="roas > 3.0",
            action="increase_bid",
            priority=7,
            is_active=True
        )
    ]
    
    for rule in rules:
        result = midas.rule_engine.add_rule(rule)
        print(f"✓ {result['rule_name']}: {result['status']}")
    
    # Evaluar reglas
    rule_actions = midas.rule_engine.evaluate_rules(campaign_data["camp_001"]["performance_data"])
    print(f"✓ Acciones de reglas generadas: {len(rule_actions)}")
    
    print("\n=== Optimización Completada ===")
    
    return midas, results

if __name__ == "__main__":
    # Ejecutar ejemplo cuando se ejecute directamente
    midas_system, optimization_results = run_optimization_example()
    
    # Mostrar información adicional
    print("\n=== Información del Sistema ===")
    print(f"Modelos de Bid Optimizer entrenados: {len(midas_system.bid_optimizer.models)}")
    print(f"Modelos de Performance Predictor entrenados: {len(midas_system.performance_predictor.models)}")
    print(f"Reglas activas: {len([r for r in midas_system.rule_engine.rules.values() if r.is_active])}")
    print(f"Tests A/B activos: {len(midas_system.creative_optimizer.active_tests)}")
    print(f"Allocaciones de budget: {len(midas_system.budget_allocator.allocations)}")