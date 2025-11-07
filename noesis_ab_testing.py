#!/usr/bin/env python3
"""
NOESIS A/B Testing System - Sistema de A/B Testing Automático

Un sistema completo de A/B testing que incluye:
- Diseño experimental automático
- Análisis estadístico avanzado
- Multi-armed bandit para optimización continua
- Reportes automáticos y dashboard

Autor: Sistema NOESIS
Fecha: 2025-11-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import norm, ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuración del experimento A/B"""
    name: str
    description: str
    control_name: str = "Control"
    variant_names: List[str] = None
    primary_metric: str = "conversion_rate"
    secondary_metrics: List[str] = None
    min_sample_size_per_variant: int = 1000
    max_sample_size_per_variant: int = 100000
    significance_level: float = 0.05
    power: float = 0.8
    min_detectable_effect: float = 0.02
    min_duration_days: int = 7
    max_duration_days: int = 30
    early_stopping_enabled: bool = True
    bandit_enabled: bool = False
    segments: List[str] = None
    
    def __post_init__(self):
        if self.variant_names is None:
            self.variant_names = ["Variant A", "Variant B"]
        if self.secondary_metrics is None:
            self.secondary_metrics = []
        if self.segments is None:
            self.segments = ["all"]

@dataclass
class ExperimentResults:
    """Resultados del experimento"""
    experiment_id: str
    status: str
    start_date: datetime
    end_date: Optional[datetime]
    control_data: Dict
    variant_data: Dict
    statistical_results: Dict
    recommendations: List[str]
    lift_analysis: Dict
    segment_results: Dict
    early_stopping_triggered: bool = False

class ExperimentDesigner:
    """Diseñador automático de experimentos A/B"""
    
    def __init__(self):
        self.statistics_cache = {}
    
    def calculate_sample_size(self, config: ExperimentConfig, 
                            baseline_rate: float, effect_size: float) -> int:
        """Calcula el tamaño de muestra necesario para detectar el efecto"""
        logger.info(f"Calculando tamaño de muestra para efecto: {effect_size:.4f}")
        
        # Z-scores para el nivel de significancia y poder
        z_alpha = stats.norm.ppf(1 - config.significance_level / 2)
        z_beta = stats.norm.ppf(config.power)
        
        # Tasa esperada en la variante
        treatment_rate = baseline_rate * (1 + effect_size)
        
        # Varianzas combinadas
        p1 = baseline_rate
        p2 = treatment_rate
        p_combined = (p1 + p2) / 2
        
        # Fórmula de tamaño de muestra para proporciones
        numerator = (z_alpha * np.sqrt(2 * p_combined * (1 - p_combined)) + 
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        
        sample_size_per_variant = int(np.ceil(numerator / denominator))
        
        # Aplicar límites
        sample_size_per_variant = max(config.min_sample_size_per_variant, 
                                    min(sample_size_per_variant, 
                                        config.max_sample_size_per_variant))
        
        logger.info(f"Tamaño de muestra calculado: {sample_size_per_variant} por variante")
        return sample_size_per_variant
    
    def estimate_experiment_duration(self, config: ExperimentConfig, 
                                  daily_traffic_per_variant: int, 
                                  sample_size_per_variant: int) -> int:
        """Estima la duración del experimento en días"""
        days_needed = np.ceil(sample_size_per_variant / daily_traffic_per_variant)
        
        # Aplicar límites de duración
        duration = max(config.min_duration_days, 
                      min(int(days_needed), config.max_duration_days))
        
        logger.info(f"Duración estimada: {duration} días")
        return duration
    
    def design_experiment(self, config: ExperimentConfig, 
                         baseline_rates: Dict[str, float],
                         expected_effects: Dict[str, float],
                         daily_traffic: int) -> Dict[str, Any]:
        """Diseña un experimento completo"""
        logger.info(f"Diseñando experimento: {config.name}")
        
        # Calcular tamaños de muestra para cada variante
        sample_sizes = {}
        for variant in [config.control_name] + config.variant_names:
            if variant == config.control_name:
                sample_sizes[variant] = config.min_sample_size_per_variant
            else:
                effect = expected_effects.get(variant, config.min_detectable_effect)
                sample_sizes[variant] = self.calculate_sample_size(
                    config, baseline_rates.get('control', 0.1), effect
                )
        
        # Calcular duraciones
        durations = {}
        max_sample_size = max(sample_sizes.values())
        estimated_duration = self.estimate_experiment_duration(
            config, daily_traffic // len(sample_sizes), max_sample_size
        )
        
        # Segmentación
        segments_info = {}
        if config.segments != ["all"]:
            segment_ratio = 1.0 / len(config.segments)
            for segment in config.segments:
                segments_info[segment] = {
                    'traffic_allocation': segment_ratio,
                    'sample_size': int(max_sample_size * segment_ratio)
                }
        
        experiment_design = {
            'config': asdict(config),
            'sample_sizes': sample_sizes,
            'estimated_duration_days': estimated_duration,
            'daily_traffic_per_variant': daily_traffic // len(sample_sizes),
            'segments': segments_info,
            'power_analysis': {
                'significance_level': config.significance_level,
                'power': config.power,
                'effect_sizes': expected_effects
            },
            'created_at': datetime.now().isoformat()
        }
        
        logger.info("Diseño de experimento completado")
        return experiment_design

class StatisticalAnalyzer:
    """Analizador estadístico para experimentos A/B"""
    
    def __init__(self):
        pass
    
    def analyze_conversion_rates(self, control_data: List[float], 
                               variant_data: List[float], 
                               significance_level: float = 0.05) -> Dict[str, Any]:
        """Analiza tasas de conversión usando t-test y chi-square"""
        logger.info("Analizando tasas de conversión")
        
        # Convertir datos a formato binario (conversión/no conversión)
        control_conversions = np.array(control_data)
        variant_conversions = np.array(variant_data)
        
        # Estadísticas descriptivas
        control_rate = np.mean(control_conversions)
        variant_rate = np.mean(variant_conversions)
        
        # T-test para diferencias en medias
        t_stat, t_p_value = ttest_ind(control_conversions, variant_conversions)
        
        # Chi-square test para proporciones
        control_success = int(np.sum(control_conversions))
        control_fail = len(control_conversions) - control_success
        variant_success = int(np.sum(variant_conversions))
        variant_fail = len(variant_conversions) - variant_success
        
        contingency_table = np.array([[control_success, control_fail],
                                    [variant_success, variant_fail]])
        
        chi2_stat, chi2_p_value, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
        
        # Prueba de proporciones
        z_stat, z_p_value = self._z_test_proportions(
            control_success, len(control_conversions),
            variant_success, len(variant_conversions)
        )
        
        # Intervalos de confianza
        control_ci = self._proportion_confidence_interval(control_success, len(control_conversions))
        variant_ci = self._proportion_confidence_interval(variant_success, len(variant_conversions))
        
        results = {
            'control': {
                'conversions': control_success,
                'total': len(control_conversions),
                'rate': control_rate,
                'ci_95': control_ci
            },
            'variant': {
                'conversions': variant_success,
                'total': len(variant_conversions),
                'rate': variant_rate,
                'ci_95': variant_ci
            },
            'statistical_tests': {
                't_test': {
                    'statistic': float(t_stat),
                    'p_value': float(t_p_value),
                    'significant': t_p_value < significance_level
                },
                'chi_square': {
                    'statistic': float(chi2_stat),
                    'p_value': float(chi2_p_value),
                    'significant': chi2_p_value < significance_level
                },
                'z_test_proportions': {
                    'statistic': float(z_stat),
                    'p_value': float(z_p_value),
                    'significant': z_p_value < significance_level
                }
            },
            'effect_size': {
                'absolute_difference': variant_rate - control_rate,
                'relative_lift': (variant_rate - control_rate) / control_rate if control_rate > 0 else 0
            }
        }
        
        return results
    
    def bayesian_analysis(self, control_data: List[float], 
                         variant_data: List[float], 
                         prior_alpha: float = 1, 
                         prior_beta: float = 1) -> Dict[str, Any]:
        """Análisis bayesiano de los resultados"""
        logger.info("Realizando análisis bayesiano")
        
        # Datos observados
        control_success = int(np.sum(control_data))
        control_total = len(control_data)
        variant_success = int(np.sum(variant_data))
        variant_total = len(variant_data)
        
        # Posterior distributions (Beta distribution)
        control_posterior_alpha = prior_alpha + control_success
        control_posterior_beta = prior_beta + control_total - control_success
        
        variant_posterior_alpha = prior_alpha + variant_success
        variant_posterior_beta = prior_beta + variant_total - variant_success
        
        # Probabilidad de que la variante sea mejor
        samples_control = np.random.beta(control_posterior_alpha, control_posterior_beta, 10000)
        samples_variant = np.random.beta(variant_posterior_alpha, variant_posterior_beta, 10000)
        
        prob_variant_better = np.mean(samples_variant > samples_control)
        
        # Intervalos creíbles
        control_ci = np.percentile(samples_control, [2.5, 97.5])
        variant_ci = np.percentile(samples_variant, [2.5, 97.5])
        
        # Lift esperado
        expected_lift = np.mean((samples_variant - samples_control) / samples_control)
        
        results = {
            'posterior_distributions': {
                'control': {
                    'alpha': control_posterior_alpha,
                    'beta': control_posterior_beta,
                    'mean': control_posterior_alpha / (control_posterior_alpha + control_posterior_beta),
                    'credible_interval': [float(control_ci[0]), float(control_ci[1])]
                },
                'variant': {
                    'alpha': variant_posterior_alpha,
                    'beta': variant_posterior_beta,
                    'mean': variant_posterior_alpha / (variant_posterior_alpha + variant_posterior_beta),
                    'credible_interval': [float(variant_ci[0]), float(variant_ci[1])]
                }
            },
            'probability_variant_better': float(prob_variant_better),
            'expected_lift': float(expected_lift),
            'confidence_level': 'high' if prob_variant_better > 0.95 else 'medium' if prob_variant_better > 0.8 else 'low'
        }
        
        return results
    
    def detect_early_stopping(self, cumulative_data: List[Dict], 
                            config: ExperimentConfig) -> Dict[str, Any]:
        """Detecta si se debe hacer early stopping"""
        logger.info("Verificando condiciones para early stopping")
        
        if not config.early_stopping_enabled:
            return {'should_stop': False, 'reason': 'Early stopping deshabilitado'}
        
        # Obtener los datos más recientes
        latest_data = cumulative_data[-1]
        
        # Verificar significancia estadística
        control_results = latest_data.get('control', {})
        variant_results = latest_data.get('variant', {})
        
        if 'statistical_tests' in variant_results:
            p_value = variant_results['statistical_tests'].get('t_test', {}).get('p_value', 1.0)
            
            if p_value < config.significance_level / 10:  # Criterio más estricto para early stopping
                return {
                    'should_stop': True, 
                    'reason': f'Significancia estadística muy alta (p={p_value:.6f})'
                }
        
        # Verificar si la duración mínima se ha cumplido
        start_date = datetime.fromisoformat(latest_data.get('timestamp'))
        if (datetime.now() - start_date).days < config.min_duration_days:
            return {'should_stop': False, 'reason': 'Duración mínima no cumplida'}
        
        # Verificar tamaño de muestra mínimo
        control_sample_size = control_results.get('total', 0)
        variant_sample_size = variant_results.get('total', 0)
        
        if (control_sample_size < config.min_sample_size_per_variant or 
            variant_sample_size < config.min_sample_size_per_variant):
            return {'should_stop': False, 'reason': 'Tamaño de muestra insuficiente'}
        
        return {'should_stop': False, 'reason': 'Condiciones de early stopping no cumplidas'}
    
    def _z_test_proportions(self, success1: int, total1: int, 
                          success2: int, total2: int) -> Tuple[float, float]:
        """Prueba Z para comparar dos proporciones"""
        p1 = success1 / total1
        p2 = success2 / total2
        p_pool = (success1 + success2) / (total1 + total2)
        
        se = np.sqrt(p_pool * (1 - p_pool) * (1/total1 + 1/total2))
        z_stat = (p1 - p2) / se
        
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        
        return z_stat, p_value
    
    def _proportion_confidence_interval(self, successes: int, total: int, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Intervalo de confianza para una proporción"""
        p = successes / total
        z = norm.ppf(1 - (1 - confidence) / 2)
        
        se = np.sqrt(p * (1 - p) / total)
        
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)
        
        return lower, upper

class BanditOptimizer:
    """Optimizador Multi-Armed Bandit para experimentación continua"""
    
    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.995):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.arm_stats = {}
        self.total_pulls = 0
        self.algorithm_type = "epsilon_greedy"
    
    def add_arm(self, arm_id: str, initial_estimate: float = 0.1):
        """Añade un brazo al bandit"""
        self.arm_stats[arm_id] = {
            'pulls': 0,
            'reward_sum': 0,
            'estimated_value': initial_estimate,
            'confidence_interval': [0, 1]
        }
        logger.info(f"Arm {arm_id} añadido con estimación inicial {initial_estimate:.4f}")
    
    def select_arm(self) -> str:
        """Selecciona un brazo usando epsilon-greedy con decay"""
        if not self.arm_stats:
            raise ValueError("No arms available")
        
        # Decay epsilon
        self.epsilon *= self.decay_rate
        
        # Exploration vs exploitation
        if np.random.random() < self.epsilon:
            # Exploration: seleccionar un brazo al azar
            return np.random.choice(list(self.arm_stats.keys()))
        else:
            # Exploitation: seleccionar el brazo con mayor valor estimado
            return max(self.arm_stats.keys(), 
                      key=lambda arm: self.arm_stats[arm]['estimated_value'])
    
    def update(self, arm_id: str, reward: float):
        """Actualiza las estadísticas del brazo seleccionado"""
        if arm_id not in self.arm_stats:
            self.add_arm(arm_id)
        
        self.arm_stats[arm_id]['pulls'] += 1
        self.arm_stats[arm_id]['reward_sum'] += reward
        
        # Actualizar estimación del valor
        n = self.arm_stats[arm_id]['pulls']
        value = self.arm_stats[arm_id]['reward_sum'] / n
        self.arm_stats[arm_id]['estimated_value'] = value
        
        # Actualizar intervalo de confianza
        confidence_level = 0.95
        z = norm.ppf((1 + confidence_level) / 2)
        
        if n > 1:
            std = np.sqrt(self._calculate_variance(arm_id))
            margin = z * std / np.sqrt(n)
            self.arm_stats[arm_id]['confidence_interval'] = [
                max(0, value - margin), min(1, value + margin)
            ]
        
        self.total_pulls += 1
        logger.info(f"Arm {arm_id} actualizado: reward={reward:.4f}, "
                   f"estimated_value={value:.4f}")
    
    def _calculate_variance(self, arm_id: str) -> float:
        """Calcula la varianza de un brazo"""
        stats = self.arm_stats[arm_id]
        if stats['pulls'] < 2:
            return 0.1  # Varianza por defecto
        
        # Estimación simple de varianza
        mean = stats['estimated_value']
        # Para datos binarios, la varianza es p*(1-p)
        return mean * (1 - mean)
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones basadas en el estado actual del bandit"""
        recommendations = []
        
        # Encontrar el mejor brazo
        best_arm = max(self.arm_stats.keys(), 
                      key=lambda arm: self.arm_stats[arm]['estimated_value'])
        
        # Calcular estadísticas de confianza
        for arm_id, stats in self.arm_stats.items():
            ci = stats['confidence_interval']
            confidence_score = 1 - (ci[1] - ci[0])  # Intervalo más pequeño = más confianza
            
            recommendation = {
                'arm_id': arm_id,
                'estimated_value': stats['estimated_value'],
                'pulls': stats['pulls'],
                'confidence_score': confidence_score,
                'confidence_interval': ci,
                'is_recommended': arm_id == best_arm,
                'exploitation_ratio': 1 - self.epsilon
            }
            recommendations.append(recommendation)
        
        # Ordenar por valor estimado
        recommendations.sort(key=lambda x: x['estimated_value'], reverse=True)
        
        return recommendations
    
    def thompson_sampling_select_arm(self) -> str:
        """Selección de brazo usando Thompson Sampling"""
        if not self.arm_stats:
            raise ValueError("No arms available")
        
        # Para cada brazo, muestrear de la distribución Beta
        samples = {}
        for arm_id, stats in self.arm_stats.items():
            alpha = 1 + stats['reward_sum']  # Éxitos + 1
            beta = 1 + stats['pulls'] - stats['reward_sum']  # Fallos + 1
            
            sample = np.random.beta(alpha, beta)
            samples[arm_id] = sample
        
        return max(samples.keys(), key=lambda arm: samples[arm])
    
    def ucb_select_arm(self, confidence_level: float = 0.95) -> str:
        """Selección de brazo usando Upper Confidence Bound"""
        if not self.arm_stats:
            raise ValueError("No arms available")
        
        best_arm = None
        best_value = -float('inf')
        
        for arm_id, stats in self.arm_stats.items():
            if stats['pulls'] == 0:
                return arm_id  # Si nunca se ha probado, seleccionarlo
            
            exploitation = stats['estimated_value']
            exploration = np.sqrt(2 * np.log(self.total_pulls) / stats['pulls'])
            z = norm.ppf((1 + confidence_level) / 2)
            
            ucb_value = exploitation + z * exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_arm = arm_id
        
        return best_arm

class LiftAnalyzer:
    """Analizador de lift e impacto incremental"""
    
    def __init__(self):
        pass
    
    def calculate_lift(self, control_data: List[float], 
                      variant_data: List[float]) -> Dict[str, Any]:
        """Calcula el lift entre control y variante"""
        logger.info("Calculando análisis de lift")
        
        control_rate = np.mean(control_data)
        variant_rate = np.mean(variant_data)
        
        # Lift absoluto
        absolute_lift = variant_rate - control_rate
        
        # Lift relativo
        relative_lift = (absolute_lift / control_rate) if control_rate > 0 else 0
        
        # Lift porcentual
        percent_lift = relative_lift * 100
        
        # Intervalos de confianza para el lift
        control_ci = self._bootstrap_ci(control_data, confidence=0.95, n_bootstrap=1000)
        variant_ci = self._bootstrap_ci(variant_data, confidence=0.95, n_bootstrap=1000)
        
        # Lift con intervalos de confianza
        lift_ci = self._bootstrap_lift_ci(control_data, variant_data, 
                                        confidence=0.95, n_bootstrap=1000)
        
        results = {
            'absolute_lift': absolute_lift,
            'relative_lift': relative_lift,
            'percent_lift': percent_lift,
            'control_rate': control_rate,
            'variant_rate': variant_rate,
            'lift_confidence_interval': lift_ci,
            'control_ci_95': control_ci,
            'variant_ci_95': variant_ci,
            'statistical_significance': self._is_significant_lift(control_data, variant_data)
        }
        
        return results
    
    def calculate_incremental_impact(self, control_data: List[float],
                                   variant_data: List[float],
                                   baseline_conversions: int = 1000) -> Dict[str, Any]:
        """Calcula el impacto incremental en conversiones"""
        logger.info("Calculando impacto incremental")
        
        lift_results = self.calculate_lift(control_data, variant_data)
        
        # Impacto incremental en conversiones
        incremental_conversions = baseline_conversions * lift_results['absolute_lift']
        incremental_revenue = incremental_conversions * 50  # Asumiendo $50 por conversión
        
        # Impacto proyectado a diferentes escalas
        impact_scenarios = []
        for scale in [100, 1000, 10000, 100000]:
            scenario = {
                'scale': scale,
                'incremental_conversions': scale * lift_results['absolute_lift'],
                'incremental_revenue': scale * lift_results['absolute_lift'] * 50
            }
            impact_scenarios.append(scenario)
        
        results = {
            'lift_analysis': lift_results,
            'incremental_impact': {
                'conversions': incremental_conversions,
                'revenue': incremental_revenue
            },
            'projected_scenarios': impact_scenarios,
            'roi_estimate': incremental_revenue / 1000 if baseline_conversions > 0 else 0  # Asumiendo $1000 de costo
        }
        
        return results
    
    def _bootstrap_ci(self, data: List[float], confidence: float = 0.95, 
                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Intervalo de confianza bootstrap"""
        bootstrap_means = []
        data_array = np.array(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return lower, upper
    
    def _bootstrap_lift_ci(self, control_data: List[float], variant_data: List[float],
                          confidence: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Intervalo de confianza bootstrap para el lift"""
        bootstrap_lifts = []
        control_array = np.array(control_data)
        variant_array = np.array(variant_data)
        
        for _ in range(n_bootstrap):
            control_sample = np.random.choice(control_array, size=len(control_array), replace=True)
            variant_sample = np.random.choice(variant_array, size=len(variant_array), replace=True)
            
            control_rate = np.mean(control_sample)
            variant_rate = np.mean(variant_sample)
            lift = (variant_rate - control_rate) / control_rate if control_rate > 0 else 0
            
            bootstrap_lifts.append(lift)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_lifts, lower_percentile)
        upper = np.percentile(bootstrap_lifts, upper_percentile)
        
        return lower, upper
    
    def _is_significant_lift(self, control_data: List[float], 
                           variant_data: List[float], 
                           alpha: float = 0.05) -> bool:
        """Determina si el lift es estadísticamente significativo"""
        _, p_value = ttest_ind(control_data, variant_data)
        return p_value < alpha

class ReportGenerator:
    """Generador de reportes automáticos"""
    
    def __init__(self):
        self.template_cache = {}
    
    def generate_experiment_report(self, experiment_results: ExperimentResults) -> str:
        """Genera un reporte completo del experimento"""
        logger.info(f"Generando reporte para experimento: {experiment_results.experiment_id}")
        
        report = f"""
# NOESIS A/B Testing - Reporte de Experimento

## Información General
- **ID del Experimento**: {experiment_results.experiment_id}
- **Estado**: {experiment_results.status}
- **Fecha de Inicio**: {experiment_results.start_date.strftime('%Y-%m-%d %H:%M:%S')}
- **Fecha de Finalización**: {experiment_results.end_date.strftime('%Y-%m-%d %H:%M:%S') if experiment_results.end_date else 'En curso'}
- **Early Stopping Activado**: {'Sí' if experiment_results.early_stopping_triggered else 'No'}

## Resumen Ejecutivo
{self._generate_executive_summary(experiment_results)}

## Resultados Estadísticos
{self._format_statistical_results(experiment_results.statistical_results)}

## Análisis de Lift
{self._format_lift_analysis(experiment_results.lift_analysis)}

## Segmentación
{self._format_segment_results(experiment_results.segment_results)}

## Recomendaciones
{self._format_recommendations(experiment_results.recommendations)}

## Apéndices
{self._format_appendices(experiment_results)}
        """
        
        return report
    
    def generate_dashboard_data(self, experiments: List[ExperimentResults]) -> Dict[str, Any]:
        """Genera datos para el dashboard"""
        logger.info("Generando datos para dashboard")
        
        dashboard_data = {
            'summary': {
                'total_experiments': len(experiments),
                'active_experiments': len([e for e in experiments if e.status == 'running']),
                'completed_experiments': len([e for e in experiments if e.status == 'completed']),
                'significant_results': len([e for e in experiments if self._has_significant_results(e)])
            },
            'experiments': [asdict(exp) for exp in experiments],
            'performance_metrics': self._calculate_performance_metrics(experiments),
            'recommendations_overview': self._generate_recommendations_overview(experiments)
        }
        
        return dashboard_data
    
    def _generate_executive_summary(self, results: ExperimentResults) -> str:
        """Genera el resumen ejecutivo"""
        status = "exitoso" if results.status == "completed" else "en proceso"
        lift = results.lift_analysis.get('percent_lift', 0)
        significant = self._has_significant_results(results)
        
        summary = f"""
Este experimento {status} con un lift del {lift:.2f}%. 

**Hallazgos principales:**
- El análisis {'confirma' if significant else 'no confirma'} significancia estadística
- Se observaron mejoras en las métricas principales
- {'Se recomienda la implementación del cambio' if significant and lift > 0 else 'Se requiere mayor investigación'}

**Impacto estimado:**
- Conversiones incrementales: {results.lift_analysis.get('incremental_impact', {}).get('conversions', 0):.0f}
- Ingresos adicionales: ${results.lift_analysis.get('incremental_impact', {}).get('revenue', 0):.2f}
        """
        return summary
    
    def _format_statistical_results(self, results: Dict) -> str:
        """Formatea los resultados estadísticos"""
        if not results:
            return "No hay resultados estadísticos disponibles."
        
        formatted = "### Control\n"
        if 'control' in results:
            control = results['control']
            formatted += f"- Tasa de conversión: {control.get('rate', 0):.4f}\n"
            formatted += f"- Intervalo de confianza 95%: [{control.get('ci_95', [0, 0])[0]:.4f}, {control.get('ci_95', [0, 0])[1]:.4f}]\n"
        
        formatted += "\n### Variante\n"
        if 'variant' in results:
            variant = results['variant']
            formatted += f"- Tasa de conversión: {variant.get('rate', 0):.4f}\n"
            formatted += f"- Intervalo de confianza 95%: [{variant.get('ci_95', [0, 0])[0]:.4f}, {variant.get('ci_95', [0, 0])[1]:.4f}]\n"
        
        if 'statistical_tests' in results:
            formatted += "\n### Pruebas Estadísticas\n"
            tests = results['statistical_tests']
            for test_name, test_results in tests.items():
                if 'significant' in test_results:
                    formatted += f"- {test_name}: {'Significativo' if test_results['significant'] else 'No significativo'}\n"
        
        return formatted
    
    def _format_lift_analysis(self, analysis: Dict) -> str:
        """Formatea el análisis de lift"""
        if not analysis:
            return "No hay análisis de lift disponible."
        
        formatted = f"""
- **Lift Absoluto**: {analysis.get('absolute_lift', 0):.6f}
- **Lift Relativo**: {analysis.get('relative_lift', 0):.4f}
- **Lift Porcentual**: {analysis.get('percent_lift', 0):.2f}%
- **Intervalo de Confianza del Lift**: [{analysis.get('lift_confidence_interval', [0, 0])[0]:.4f}, {analysis.get('lift_confidence_interval', [0, 0])[1]:.4f}]
        """
        return formatted
    
    def _format_segment_results(self, results: Dict) -> str:
        """Formatea los resultados por segmento"""
        if not results:
            return "No hay resultados de segmentación disponibles."
        
        formatted = ""
        for segment, data in results.items():
            formatted += f"\n### {segment}\n"
            if 'lift' in data:
                formatted += f"- Lift: {data['lift']:.4f}\n"
            if 'conversion_rate' in data:
                formatted += f"- Tasa de conversión: {data['conversion_rate']:.4f}\n"
        
        return formatted if formatted else "No hay resultados de segmentación disponibles."
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Formatea las recomendaciones"""
        if not recommendations:
            return "No hay recomendaciones específicas."
        
        formatted = ""
        for i, rec in enumerate(recommendations, 1):
            formatted += f"{i}. {rec}\n"
        
        return formatted
    
    def _format_appendices(self, results: ExperimentResults) -> str:
        """Formatea los apéndices"""
        return f"""
### Metodología
- Pruebas utilizadas: t-test, chi-square, análisis bayesiano
- Nivel de significancia: 0.05
- Método de early stopping: Análisis secuencial
- Análisis de lift: Bootstrap con 1000 iteraciones

### Configuración del Experimento
- Control: {results.control_data.get('name', 'N/A')}
- Variantes: {', '.join([str(v) for v in results.variant_data.get('names', [])])}
- Métrica primaria: {results.control_data.get('primary_metric', 'N/A')}
        """
    
    def _has_significant_results(self, results: ExperimentResults) -> bool:
        """Verifica si los resultados son estadísticamente significativos"""
        if not results.statistical_results:
            return False
        
        tests = results.statistical_results.get('statistical_tests', {})
        return any(test.get('significant', False) for test in tests.values())
    
    def _calculate_performance_metrics(self, experiments: List[ExperimentResults]) -> Dict[str, Any]:
        """Calcula métricas de rendimiento"""
        completed = [e for e in experiments if e.status == 'completed']
        significant = [e for e in completed if self._has_significant_results(e)]
        
        avg_lift = np.mean([e.lift_analysis.get('percent_lift', 0) for e in significant]) if significant else 0
        
        return {
            'success_rate': len(significant) / len(completed) if completed else 0,
            'average_lift': avg_lift,
            'total_experiments': len(experiments)
        }
    
    def _generate_recommendations_overview(self, experiments: List[ExperimentResults]) -> List[str]:
        """Genera un resumen de recomendaciones"""
        recommendations = []
        
        active = [e for e in experiments if e.status == 'running']
        if active:
            recommendations.append(f"Hay {len(active)} experimento(s) activo(s) que requieren monitoreo.")
        
        completed = [e for e in experiments if e.status == 'completed']
        significant_results = [e for e in completed if self._has_significant_results(e)]
        
        if significant_results:
            recommendations.append(f"{len(significant_results)} experimento(s) han mostrado resultados significativos.")
        
        if completed:
            success_rate = len(significant_results) / len(completed)
            if success_rate > 0.5:
                recommendations.append("La tasa de éxito de experimentos es alta (>50%).")
            elif success_rate < 0.2:
                recommendations.append("La tasa de éxito es baja. Considere revisar la estrategia de experimentación.")
        
        return recommendations

class NoesisABTestingSystem:
    """Sistema principal de A/B Testing de NOESIS"""
    
    def __init__(self):
        self.experiment_designer = ExperimentDesigner()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.bandit_optimizer = BanditOptimizer()
        self.lift_analyzer = LiftAnalyzer()
        self.report_generator = ReportGenerator()
        self.experiments = {}
        self.active_experiments = {}
        
        logger.info("Sistema NOESIS A/B Testing inicializado")
    
    def create_experiment(self, config: ExperimentConfig,
                         baseline_rates: Dict[str, float],
                         expected_effects: Dict[str, float],
                         daily_traffic: int) -> str:
        """Crea un nuevo experimento"""
        logger.info(f"Creando nuevo experimento: {config.name}")
        
        # Diseñar el experimento
        design = self.experiment_designer.design_experiment(
            config, baseline_rates, expected_effects, daily_traffic
        )
        
        # Generar ID del experimento
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Crear registro del experimento
        self.experiments[experiment_id] = {
            'config': config,
            'design': design,
            'status': 'created',
            'created_at': datetime.now(),
            'results': None
        }
        
        logger.info(f"Experimento creado con ID: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Inicia un experimento"""
        if experiment_id not in self.experiments:
            logger.error(f"Experimento {experiment_id} no encontrado")
            return False
        
        self.experiments[experiment_id]['status'] = 'running'
        self.active_experiments[experiment_id] = self.experiments[experiment_id]
        
        logger.info(f"Experimento {experiment_id} iniciado")
        return True
    
    def add_data_point(self, experiment_id: str, variant: str, 
                      data_point: float, timestamp: datetime = None) -> bool:
        """Añade un punto de datos al experimento"""
        if experiment_id not in self.experiments:
            logger.error(f"Experimento {experiment_id} no encontrado")
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if 'data' not in self.experiments[experiment_id]:
            self.experiments[experiment_id]['data'] = {'control': [], 'variants': {}}
        
        if variant == 'control':
            self.experiments[experiment_id]['data']['control'].append({
                'value': data_point,
                'timestamp': timestamp
            })
        else:
            if variant not in self.experiments[experiment_id]['data']['variants']:
                self.experiments[experiment_id]['data']['variants'][variant] = []
            
            self.experiments[experiment_id]['data']['variants'][variant].append({
                'value': data_point,
                'timestamp': timestamp
            })
        
        logger.info(f"Punto de datos añadido a {experiment_id} - {variant}: {data_point}")
        return True
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResults:
        """Analiza un experimento completo"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experimento {experiment_id} no encontrado")
        
        experiment = self.experiments[experiment_id]
        data = experiment.get('data', {'control': [], 'variants': {}})
        
        if not data['control'] or not any(data['variants'].values()):
            raise ValueError("No hay datos suficientes para el análisis")
        
        # Obtener datos más recientes
        control_data = [d['value'] for d in data['control']]
        variant_data = [d['value'] for d in data['variants'][list(data['variants'].keys())[0]]]
        
        # Análisis estadístico
        config = experiment['config']
        statistical_results = self.statistical_analyzer.analyze_conversion_rates(
            control_data, variant_data, config.significance_level
        )
        
        # Análisis bayesiano
        bayesian_results = self.statistical_analyzer.bayesian_analysis(
            control_data, variant_data
        )
        statistical_results['bayesian'] = bayesian_results
        
        # Análisis de lift
        lift_analysis = self.lift_analyzer.calculate_incremental_impact(
            control_data, variant_data
        )
        
        # Detectar early stopping
        cumulative_data = [{
            'timestamp': datetime.now().isoformat(),
            'control': statistical_results.get('control', {}),
            'variant': statistical_results.get('variant', {})
        }]
        early_stopping = self.statistical_analyzer.detect_early_stopping(
            cumulative_data, config
        )
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            statistical_results, lift_analysis, early_stopping
        )
        
        # Resultados por segmento (simulado)
        segment_results = self._analyze_segments(experiment, control_data, variant_data)
        
        # Crear resultados del experimento
        results = ExperimentResults(
            experiment_id=experiment_id,
            status=experiment['status'],
            start_date=experiment['created_at'],
            end_date=datetime.now() if experiment['status'] == 'completed' else None,
            control_data={'name': config.control_name, 'data_points': len(control_data)},
            variant_data={'names': config.variant_names, 'data_points': len(variant_data)},
            statistical_results=statistical_results,
            recommendations=recommendations,
            lift_analysis=lift_analysis,
            segment_results=segment_results,
            early_stopping_triggered=early_stopping['should_stop']
        )
        
        # Actualizar estado del experimento
        self.experiments[experiment_id]['results'] = results
        if early_stopping['should_stop']:
            self.experiments[experiment_id]['status'] = 'completed'
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
        
        logger.info(f"Experimento {experiment_id} analizado")
        return results
    
    def _generate_recommendations(self, statistical_results: Dict, 
                                lift_analysis: Dict, early_stopping: Dict) -> List[str]:
        """Genera recomendaciones basadas en los resultados"""
        recommendations = []
        
        # Verificar significancia estadística
        tests = statistical_results.get('statistical_tests', {})
        significant_tests = [name for name, test in tests.items() if test.get('significant', False)]
        
        if significant_tests:
            recommendations.append("Los resultados muestran significancia estadística en: " + 
                                 ", ".join(significant_tests))
        else:
            recommendations.append("No se detectó significancia estadística. Considere aumentar el tamaño de muestra o la duración del experimento.")
        
        # Analizar lift
        lift = lift_analysis.get('lift_analysis', {}).get('percent_lift', 0)
        if abs(lift) > 2:  # Lift significativo (>2%)
            if lift > 0:
                recommendations.append(f"Se observa un lift positivo del {lift:.2f}%. Se recomienda implementar el cambio.")
            else:
                recommendations.append(f"Se observa un lift negativo del {lift:.2f}%. Se recomienda mantener el control.")
        else:
            recommendations.append(f"El lift observado ({lift:.2f}%) no es significativo. Considere hipótesis alternativas.")
        
        # Early stopping
        if early_stopping['should_stop']:
            recommendations.append(f"Early stopping activado: {early_stopping['reason']}")
        
        # Análisis bayesiano
        bayesian = statistical_results.get('bayesian', {})
        prob_better = bayesian.get('probability_variant_better', 0.5)
        if prob_better > 0.95:
            recommendations.append("La probabilidad bayesiana de que la variante sea mejor es muy alta (>95%).")
        elif prob_better < 0.05:
            recommendations.append("La probabilidad bayesiana de que la variante sea mejor es muy baja (<5%).")
        
        return recommendations
    
    def _analyze_segments(self, experiment: Dict, control_data: List[float], 
                         variant_data: List[float]) -> Dict[str, Any]:
        """Analiza resultados por segmento (simulado)"""
        config = experiment['config']
        segments = config.segments if config.segments else ['all']
        
        segment_results = {}
        base_lift = np.mean(variant_data) - np.mean(control_data)
        
        for segment in segments:
            if segment == 'all':
                segment_results[segment] = {
                    'lift': base_lift,
                    'conversion_rate': np.mean(control_data + variant_data),
                    'sample_size': len(control_data) + len(variant_data)
                }
            else:
                # Simular variación por segmento
                segment_lift = base_lift * (0.8 + 0.4 * np.random.random())
                segment_results[segment] = {
                    'lift': segment_lift,
                    'conversion_rate': np.mean(control_data) + segment_lift,
                    'sample_size': (len(control_data) + len(variant_data)) // len(segments)
                }
        
        return segment_results
    
    def get_experiment_report(self, experiment_id: str) -> str:
        """Obtiene el reporte completo de un experimento"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experimento {experiment_id} no encontrado")
        
        experiment = self.experiments[experiment_id]
        if not experiment.get('results'):
            # Analizar si no se ha hecho aún
            results = self.analyze_experiment(experiment_id)
        else:
            results = experiment['results']
        
        return self.report_generator.generate_experiment_report(results)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para el dashboard"""
        experiments_list = []
        for exp in self.experiments.values():
            if exp.get('results'):
                experiments_list.append(exp['results'])
            else:
                # Crear resultados preliminares
                preliminary = ExperimentResults(
                    experiment_id=exp.get('id', 'unknown'),
                    status=exp['status'],
                    start_date=exp['created_at'],
                    end_date=None,
                    control_data={},
                    variant_data={},
                    statistical_results={},
                    recommendations=['Esperando más datos'],
                    lift_analysis={},
                    segment_results={}
                )
                experiments_list.append(preliminary)
        
        return self.report_generator.generate_dashboard_data(experiments_list)
    
    def optimize_with_bandit(self, arm_performance: Dict[str, float]) -> str:
        """Optimiza usando multi-armed bandit"""
        # Inicializar brazos
        for arm_id in arm_performance.keys():
            if arm_id not in self.bandit_optimizer.arm_stats:
                self.bandit_optimizer.add_arm(arm_id, arm_performance[arm_id])
        
        # Selección de brazo
        selected_arm = self.bandit_optimizer.select_arm()
        
        # Obtener recomendaciones
        recommendations = self.bandit_optimizer.get_recommendations()
        
        logger.info(f"Bandit seleccionó brazo: {selected_arm}")
        return selected_arm
    
    def simulate_ab_test(self, config: ExperimentConfig, n_days: int = 14) -> Dict[str, Any]:
        """Simula un A/B test completo para demostración"""
        logger.info(f"Simulando A/B test para {n_days} días")
        
        # Crear experimento
        baseline_rate = 0.1
        expected_effect = 0.15
        
        experiment_id = self.create_experiment(
            config,
            {'control': baseline_rate},
            {'Variant A': expected_effect},
            daily_traffic=1000
        )
        
        # Iniciar experimento
        self.start_experiment(experiment_id)
        
        # Simular datos diarios
        control_data = []
        variant_data = []
        
        for day in range(n_days):
            # Simular datos de control
            control_daily = np.random.binomial(500, baseline_rate)
            control_data.extend([1] * control_daily + [0] * (500 - control_daily))
            
            # Simular datos de variante
            variant_rate = baseline_rate * (1 + expected_effect)
            variant_daily = np.random.binomial(500, variant_rate)
            variant_data.extend([1] * variant_daily + [0] * (500 - variant_daily))
            
            # Añadir puntos de datos
            for i in range(500):
                self.add_data_point(experiment_id, 'control', 
                                  np.random.binomial(1, baseline_rate))
                self.add_data_point(experiment_id, 'Variant A', 
                                  np.random.binomial(1, variant_rate))
        
        # Analizar resultados
        results = self.analyze_experiment(experiment_id)
        
        # Marcar como completado
        self.experiments[experiment_id]['status'] = 'completed'
        
        simulation_results = {
            'experiment_id': experiment_id,
            'results': results,
            'report': self.get_experiment_report(experiment_id),
            'simulation_data': {
                'total_control_samples': len(control_data),
                'total_variant_samples': len(variant_data),
                'observed_control_rate': np.mean(control_data),
                'observed_variant_rate': np.mean(variant_data),
                'simulation_days': n_days
            }
        }
        
        return simulation_results

def create_dashboard_visualization(dashboard_data: Dict[str, Any]) -> str:
    """Crea visualización para el dashboard"""
    # Esta función crearía gráficos para el dashboard
    # Por simplicidad, devolvemos una descripción
    
    dashboard_description = f"""
# NOESIS A/B Testing Dashboard

## Resumen General
- Total de experimentos: {dashboard_data['summary']['total_experiments']}
- Experimentos activos: {dashboard_data['summary']['active_experiments']}
- Experimentos completados: {dashboard_data['summary']['completed_experiments']}
- Resultados significativos: {dashboard_data['summary']['significant_results']}

## Métricas de Rendimiento
- Tasa de éxito: {dashboard_data['performance_metrics']['success_rate']:.2%}
- Lift promedio: {dashboard_data['performance_metrics']['average_lift']:.2f}%

## Recomendaciones
"""
    
    for rec in dashboard_data['recommendations_overview']:
        dashboard_description += f"- {rec}\n"
    
    return dashboard_description

# Ejemplo de uso y demostración
if __name__ == "__main__":
    # Inicializar el sistema
    noesis_ab = NoesisABTestingSystem()
    
    # Configurar experimento
    config = ExperimentConfig(
        name="Test de Call-to-Action",
        description="Probando diferentes textos de botón de conversión",
        control_name="Botón Original",
        variant_names=["Botón Mejorado"],
        primary_metric="conversion_rate",
        min_sample_size_per_variant=500,
        significance_level=0.05,
        power=0.8,
        min_detectable_effect=0.1,
        early_stopping_enabled=True,
        bandit_enabled=True
    )
    
    # Simular experimento
    print("=== NOESIS A/B Testing System - Demostración ===\n")
    
    results = noesis_ab.simulate_ab_test(config, n_days=7)
    
    print(f"Experimento simulado: {results['experiment_id']}")
    print(f"Tasa de control observada: {results['simulation_data']['observed_control_rate']:.4f}")
    print(f"Tasa de variante observada: {results['simulation_data']['observed_variant_rate']:.4f}")
    
    print("\n" + "="*60)
    print("REPORTE COMPLETO")
    print("="*60)
    print(results['report'])
    
    # Dashboard
    print("\n" + "="*60)
    print("DASHBOARD")
    print("="*60)
    dashboard_data = noesis_ab.get_dashboard_data()
    dashboard_viz = create_dashboard_visualization(dashboard_data)
    print(dashboard_viz)
    
    # Ejemplo de bandit
    print("\n" + "="*60)
    print("OPTIMIZACIÓN BANDIT")
    print("="*60)
    
    arm_performance = {
        'homepage_v1': 0.12,
        'homepage_v2': 0.15,
        'homepage_v3': 0.08
    }
    
    best_arm = noesis_ab.optimize_with_bandit(arm_performance)
    print(f"Mejor brazo seleccionado por bandit: {best_arm}")
    
    recommendations = noesis_ab.bandit_optimizer.get_recommendations()
    print("\nRecomendaciones del Bandit:")
    for rec in recommendations:
        print(f"- {rec['arm_id']}: {rec['estimated_value']:.4f} (confidence: {rec['confidence_score']:.2f})")