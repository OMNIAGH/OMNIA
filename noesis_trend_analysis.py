"""
NOESIS - Sistema de Análisis de Tendencias
===========================================

Sistema completo para el análisis de tendencias en datos financieros y económicos.
Incluye detección automática de tendencias, análisis de estacionalidad,
identificación de puntos de inflexión, correlación entre variables,
análisis de volatilidad y sistema de alertas.

Autor: NOESIS Team
Fecha: 2025-11-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-v0_8')

class TrendDirection(Enum):
    """Direcciones de tendencia"""
    ALCISTA = "Alcista"
    BAJISTA = "Bajista"
    LATERAL = "Lateral"
    INDETERMINADO = "Indeterminado"

class AlertLevel(Enum):
    """Niveles de alerta"""
    BAJO = "Bajo"
    MEDIO = "Medio"
    ALTO = "Alto"
    CRITICO = "Crítico"

@dataclass
class TrendMetrics:
    """Métricas de tendencia"""
    direction: TrendDirection
    strength: float
    r_squared: float
    slope: float
    confidence: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@dataclass
class InflectionPoint:
    """Punto de inflexión"""
    date: str
    price: float
    type: str  # 'min', 'max'
    significance: float
    confidence: float

@dataclass
class Alert:
    """Alerta del sistema"""
    level: AlertLevel
    message: str
    timestamp: str
    metric: str
    value: float

class TrendDetector:
    """
    Detector de tendencias automáticas
    Identifica tendencias alcistas, bajistas y laterales
    """
    
    def __init__(self, window_size: int = 20, min_trend_length: int = 10):
        """
        Inicializa el detector de tendencias
        
        Args:
            window_size: Tamaño de la ventana móvil
            min_trend_length: Longitud mínima de tendencia
        """
        self.window_size = window_size
        self.min_trend_length = min_trend_length
        self.scaler = StandardScaler()
        
    def detect_trend_linear(self, prices: pd.Series) -> TrendMetrics:
        """
        Detecta tendencia usando regresión lineal
        
        Args:
            prices: Serie de precios
            
        Returns:
            Métricas de tendencia
        """
        # Preparar datos
        y = prices.values
        x = np.arange(len(y))
        
        # Regresión lineal
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        
        # Calcular métricas
        slope = model.coef_[0]
        r_squared = model.score(x.reshape(-1, 1), y)
        
        # Determinar dirección
        if slope > 0:
            direction = TrendDirection.ALCISTA
        elif slope < 0:
            direction = TrendDirection.BAJISTA
        else:
            direction = TrendDirection.LATERAL
        
        # Calcular fuerza
        strength = abs(slope) / np.mean(np.abs(np.diff(y)))
        
        # Calcular confianza basada en R²
        confidence = r_squared
        
        return TrendMetrics(
            direction=direction,
            strength=strength,
            r_squared=r_squared,
            slope=slope,
            confidence=confidence
        )
    
    def detect_trend_moving_average(self, prices: pd.Series) -> Dict[str, TrendDirection]:
        """
        Detecta tendencia usando múltiples medias móviles
        
        Args:
            prices: Serie de precios
            
        Returns:
            Diccionario con señales de diferentes marcos temporales
        """
        signals = {}
        
        # Medias móviles de diferentes períodos
        ma_short = prices.rolling(window=5).mean()
        ma_medium = prices.rolling(window=20).mean()
        ma_long = prices.rolling(window=50).mean()
        
        # Señal de corto plazo
        if ma_short.iloc[-1] > ma_medium.iloc[-1]:
            signals['corto_plazo'] = TrendDirection.ALCISTA
        elif ma_short.iloc[-1] < ma_medium.iloc[-1]:
            signals['corto_plazo'] = TrendDirection.BAJISTA
        else:
            signals['corto_plazo'] = TrendDirection.LATERAL
        
        # Señal de medio plazo
        if ma_medium.iloc[-1] > ma_long.iloc[-1]:
            signals['medio_plazo'] = TrendDirection.ALCISTA
        elif ma_medium.iloc[-1] < ma_long.iloc[-1]:
            signals['medio_plazo'] = TrendDirection.BAJISTA
        else:
            signals['medio_plazo'] = TrendDirection.LATERAL
        
        # Señal a largo plazo
        if len(prices) >= 200:
            ma_vlong = prices.rolling(window=200).mean()
            if ma_long.iloc[-1] > ma_vlong.iloc[-1]:
                signals['largo_plazo'] = TrendDirection.ALCISTA
            elif ma_long.iloc[-1] < ma_vlong.iloc[-1]:
                signals['largo_plazo'] = TrendDirection.BAJISTA
            else:
                signals['largo_plazo'] = TrendDirection.LATERAL
        
        return signals
    
    def detect_trend_momentum(self, prices: pd.Series) -> Dict[str, float]:
        """
        Detecta tendencia usando indicadores de momentum
        
        Args:
            prices: Serie de precios
            
        Returns:
            Diccionario con indicadores de momentum
        """
        momentum = {}
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        momentum['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        
        momentum['macd'] = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
        momentum['macd_signal'] = signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0
        
        # Estocástico
        low_min = prices.rolling(window=14).min()
        high_max = prices.rolling(window=14).max()
        k_percent = 100 * (prices - low_min) / (high_max - low_min)
        d_percent = k_percent.rolling(window=3).mean()
        
        momentum['stoch_k'] = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50
        momentum['stoch_d'] = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50
        
        return momentum
    
    def detect_multiple_timeframes(self, prices: pd.Series) -> Dict:
        """
        Detecta tendencias en múltiples marcos temporales
        
        Args:
            prices: Serie de precios
            
        Returns:
            Análisis completo de tendencias
        """
        analysis = {}
        
        # Tendencia lineal
        analysis['linear'] = self.detect_trend_linear(prices)
        
        # Tendencia por medias móviles
        analysis['moving_averages'] = self.detect_trend_moving_average(prices)
        
        # Momentum
        analysis['momentum'] = self.detect_trend_momentum(prices)
        
        # Consenso
        signals = [analysis['linear'].direction]
        signals.extend(analysis['moving_averages'].values())
        
        if signals.count(TrendDirection.ALCISTA) > signals.count(TrendDirection.BAJISTA):
            consensus = TrendDirection.ALCISTA
        elif signals.count(TrendDirection.BAJISTA) > signals.count(TrendDirection.ALCISTA):
            consensus = TrendDirection.BAJISTA
        else:
            consensus = TrendDirection.LATERAL
        
        analysis['consensus'] = consensus
        analysis['consensus_strength'] = abs(signals.count(consensus) / len(signals) - 0.5) * 2
        
        return analysis
    
    def plot_trend_analysis(self, prices: pd.Series, save_path: str = None):
        """
        Visualiza el análisis de tendencias
        
        Args:
            prices: Serie de precios
            save_path: Ruta para guardar la imagen
        """
        analysis = self.detect_multiple_timeframes(prices)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Tendencias - NOESIS', fontsize=16, fontweight='bold')
        
        # Gráfico principal
        axes[0, 0].plot(prices.index, prices.values, linewidth=2, label='Precio')
        
        # Medias móviles
        ma_short = prices.rolling(window=5).mean()
        ma_medium = prices.rolling(window=20).mean()
        ma_long = prices.rolling(window=50).mean()
        
        axes[0, 0].plot(prices.index, ma_short, '--', alpha=0.7, label='MA 5')
        axes[0, 0].plot(prices.index, ma_medium, '--', alpha=0.7, label='MA 20')
        axes[0, 0].plot(prices.index, ma_long, '--', alpha=0.7, label='MA 50')
        
        axes[0, 0].set_title('Precio y Medias Móviles')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Momentum
        momentum = analysis['momentum']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        axes[0, 1].plot(rsi.index, rsi.values)
        axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Sobrecompra')
        axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Sobreventa')
        axes[0, 1].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('RSI (Relative Strength Index)')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MACD
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        
        axes[1, 0].plot(macd.index, macd.values, label='MACD', linewidth=2)
        axes[1, 0].plot(signal.index, signal.values, label='Signal', linewidth=2)
        axes[1, 0].bar(macd.index, macd.values - signal.values, alpha=0.3, label='Histograma')
        axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('MACD')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Estocástico
        low_min = prices.rolling(window=14).min()
        high_max = prices.rolling(window=14).max()
        k_percent = 100 * (prices - low_min) / (high_max - low_min)
        d_percent = k_percent.rolling(window=3).mean()
        
        axes[1, 1].plot(k_percent.index, k_percent.values, label='%K', linewidth=2)
        axes[1, 1].plot(d_percent.index, d_percent.values, label='%D', linewidth=2)
        axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Sobrecompra')
        axes[1, 1].axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Sobreventa')
        axes[1, 1].set_title('Estocástico')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Resumen de tendencias
        trend_summary = []
        ma_summary = analysis['moving_averages']
        for timeframe, direction in ma_summary.items():
            trend_summary.append(f"{timeframe}: {direction.value}")
        
        consensus = analysis['consensus']
        consensus_strength = analysis['consensus_strength']
        
        axes[2, 0].text(0.1, 0.8, "Resumen de Tendencias:", fontweight='bold', fontsize=12)
        for i, trend in enumerate(trend_summary):
            axes[2, 0].text(0.1, 0.6 - i*0.1, trend, fontsize=10)
        
        axes[2, 0].text(0.1, 0.2, f"Consenso: {consensus.value}", fontweight='bold', fontsize=14)
        axes[2, 0].text(0.1, 0.1, f"Fuerza del Consenso: {consensus_strength:.2%}", fontsize=10)
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].axis('off')
        
        # Métricas detalladas
        linear = analysis['linear']
        metrics_text = f"""
Métricas de Tendencia Lineal:
- Dirección: {linear.direction.value}
- Pendiente: {linear.slope:.4f}
- R²: {linear.r_squared:.3f}
- Fuerza: {linear.strength:.3f}
- Confianza: {linear.confidence:.3f}
        """
        
        axes[2, 1].text(0.1, 0.8, metrics_text, fontsize=10, verticalalignment='top')
        axes[2, 1].set_xlim(0, 1)
        axes[2, 1].set_ylim(0, 1)
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()

class SeasonalityAnalyzer:
    """
    Analizador de estacionalidad y patrones cíclicos
    """
    
    def __init__(self, periods: List[int] = None):
        """
        Inicializa el analizador de estacionalidad
        
        Args:
            periods: Lista de períodos a analizar (ej: [12 para mensual, 4 para trimestral])
        """
        self.periods = periods or [7, 30, 90, 365]  # Semanal, mensual, trimestral, anual
        
    def detect_seasonality(self, series: pd.Series, method: str = 'classical') -> Dict:
        """
        Detecta patrones estacionales
        
        Args:
            series: Serie temporal
            method: Método de análisis ('classical', 'x11', 'stl')
            
        Returns:
            Diccionario con análisis de estacionalidad
        """
        results = {}
        
        for period in self.periods:
            if len(series) < period * 2:
                continue
                
            if method == 'classical':
                results[period] = self._classical_decomposition(series, period)
            elif method == 'stl':
                results[period] = self._stl_decomposition(series, period)
        
        return results
    
    def _classical_decomposition(self, series: pd.Series, period: int) -> Dict:
        """
        Descomposición clásica de la serie temporal
        
        Args:
            series: Serie temporal
            period: Período estacional
            
        Returns:
            Componentes de la descomposición
        """
        # Calcular tendencia usando media móvil
        trend = series.rolling(window=period, center=True).mean()
        
        # Calcular componente estacional
        detrended = series - trend
        seasonal = detrended.groupby(detrended.index.dayofyear % period).transform('mean')
        
        # Componente irregular
        irregular = series - trend - seasonal
        
        # Estadísticas
        seasonal_strength = np.std(seasonal.dropna()) / np.std(series.dropna())
        trend_strength = np.std(trend.dropna()) / np.std(series.dropna())
        
        return {
            'original': series,
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular,
            'seasonal_strength': seasonal_strength,
            'trend_strength': trend_strength,
            'period': period
        }
    
    def _stl_decomposition(self, series: pd.Series, period: int) -> Dict:
        """
        Descomposición STL simplificada
        
        Args:
            series: Serie temporal
            period: Período estacional
            
        Returns:
            Componentes de la descomposición STL
        """
        # Tendencia usando regresión polinomial
        x = np.arange(len(series))
        y = series.values
        
        # Ajuste polinomial de grado 3 para tendencia
        coeffs = np.polyfit(x, y, 3)
        trend = pd.Series(np.polyval(coeffs, x), index=series.index)
        
        # Componente estacional
        detrended = series - trend
        seasonal = detrended.groupby(detrended.index.dayofyear % period).transform('mean')
        irregular = series - trend - seasonal
        
        return {
            'original': series,
            'trend': trend,
            'seasonal': seasonal,
            'irregular': irregular,
            'seasonal_strength': np.std(seasonal.dropna()) / np.std(series.dropna()),
            'trend_strength': np.std(trend.dropna()) / np.std(series.dropna()),
            'period': period
        }
    
    def detect_cycles(self, series: pd.Series, min_period: int = 8, max_period: int = None) -> Dict:
        """
        Detecta ciclos en la serie temporal
        
        Args:
            series: Serie temporal
            min_period: Período mínimo del ciclo
            max_period: Período máximo del ciclo
            
        Returns:
            Información sobre ciclos detectados
        """
        if max_period is None:
            max_period = len(series) // 3
        
        # Calcular autocorrelación
        autocorr = [series.autocorr(lag=lag) for lag in range(min_period, max_period)]
        
        # Encontrar picos en autocorrelación
        peaks, properties = find_peaks(autocorr, height=0.3, distance=5)
        
        cycle_periods = [min_period + peak for peak in peaks]
        cycle_strengths = [autocorr[peak] for peak in peaks]
        
        return {
            'cycle_periods': cycle_periods,
            'cycle_strengths': cycle_strengths,
            'autocorrelation': autocorr,
            'lag_range': list(range(min_period, max_period))
        }
    
    def test_stationarity(self, series: pd.Series) -> Dict:
        """
        Prueba de estacionariedad usando test de Augmented Dickey-Fuller
        
        Args:
            series: Serie temporal
            
        Returns:
            Resultados de la prueba de estacionariedad
        """
        from statsmodels.tsa.stattools import adfuller
        
        # Test ADF
        adf_result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05,
            'used_lags': adf_result[2],
            'n_observations': adf_result[3]
        }
    
    def get_seasonal_indicators(self, series: pd.Series) -> pd.DataFrame:
        """
        Calcula indicadores estacionales
        
        Args:
            series: Serie temporal
            
        Returns:
            DataFrame con indicadores estacionales
        """
        indicators = pd.DataFrame(index=series.index)
        
        # Indicadores de tiempo
        indicators['month'] = series.index.month
        indicators['quarter'] = series.index.quarter
        indicators['day_of_year'] = series.index.dayofyear
        indicators['week_of_year'] = series.index.isocalendar().week
        
        # Estacionalidad simple
        monthly_avg = series.groupby(series.index.month).transform('mean')
        quarterly_avg = series.groupby(series.index.quarter).transform('mean')
        
        indicators['monthly_sa'] = series - monthly_avg
        indicators['quarterly_sa'] = series - quarterly_avg
        
        return indicators
    
    def plot_seasonality_analysis(self, series: pd.Series, save_path: str = None):
        """
        Visualiza el análisis de estacionalidad
        
        Args:
            series: Serie temporal
            save_path: Ruta para guardar la imagen
        """
        seasonality = self.detect_seasonality(series)
        cycles = self.detect_cycles(series)
        stationarity = self.test_stationarity(series)
        
        n_plots = len(seasonality) + 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
        fig.suptitle('Análisis de Estacionalidad y Ciclos - NOESIS', fontsize=16, fontweight='bold')
        
        # Serie original
        axes[0].plot(series.index, series.values, linewidth=2, color='blue')
        axes[0].set_title('Serie Original')
        axes[0].grid(True, alpha=0.3)
        
        # Test de estacionariedad
        is_stationary = stationarity['is_stationary']
        status = "Estacionaria" if is_stationary else "No Estacionaria"
        p_value = stationarity['p_value']
        
        axes[0].text(0.02, 0.95, f"Test ADF: {status} (p-valor: {p_value:.4f})", 
                    transform=axes[0].transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen' if is_stationary else 'lightcoral'))
        
        # Descomposiciones estacionales
        for i, (period, decomp) in enumerate(seasonality.items(), 1):
            ax = axes[i]
            
            # Graficar componentes
            ax.plot(decomp['original'].index, decomp['original'].values, 
                   label='Original', alpha=0.7, linewidth=1)
            ax.plot(decomp['trend'].index, decomp['trend'].values, 
                   label='Tendencia', linewidth=2)
            ax.plot(decomp['seasonal'].index, decomp['seasonal'].values, 
                   label='Estacional', alpha=0.8)
            
            ax.set_title(f'Descomposición Estacional (Período: {period})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Añadir información de fuerza
            seasonal_strength = decomp['seasonal_strength']
            trend_strength = decomp['trend_strength']
            
            ax.text(0.02, 0.95, f"Fuerza Estacional: {seasonal_strength:.3f}\nFuerza Tendencia: {trend_strength:.3f}", 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
        
        # Análisis de ciclos
        cycle_ax = axes[-1]
        lag_range = cycles['lag_range']
        autocorr = cycles['autocorrelation']
        
        cycle_ax.plot(lag_range, autocorr, linewidth=2, marker='o', markersize=4)
        cycle_ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        cycle_ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Umbral (0.3)')
        cycle_ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.7)
        
        # Marcar ciclos detectados
        for period, strength in zip(cycles['cycle_periods'], cycles['cycle_strengths']):
            cycle_ax.plot(period, strength, 'ro', markersize=8)
            cycle_ax.annotate(f'{period}', (period, strength), xytext=(5, 5), 
                            textcoords='offset points')
        
        cycle_ax.set_title('Análisis de Autocorrelación (Detección de Ciclos)')
        cycle_ax.set_xlabel('Lag')
        cycle_ax.set_ylabel('Autocorrelación')
        cycle_ax.legend()
        cycle_ax.grid(True, alpha=0.3)
        
        # Información de ciclos
        cycles_text = f"Ciclos Detectados: {len(cycles['cycle_periods'])}\n"
        for period, strength in zip(cycles['cycle_periods'], cycles['cycle_strengths']):
            cycles_text += f"Período {period}: Fuerza {strength:.3f}\n"
        
        cycle_ax.text(0.02, 0.95, cycles_text, transform=cycle_ax.transAxes, 
                     bbox=dict(boxstyle="round", facecolor='lightyellow'), verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        
        return {
            'seasonality': seasonality,
            'cycles': cycles,
            'stationarity': stationarity
        }

class VolatilityAnalyzer:
    """
    Analizador de volatilidad y riesgo
    """
    
    def __init__(self, window_sizes: List[int] = None):
        """
        Inicializa el analizador de volatilidad
        
        Args:
            window_sizes: Tamaños de ventana para cálculos
        """
        self.window_sizes = window_sizes or [5, 10, 20, 60]
        
    def calculate_volatility_metrics(self, returns: pd.Series) -> Dict:
        """
        Calcula métricas de volatilidad
        
        Args:
            returns: Serie de rendimientos
            
        Returns:
            Diccionario con métricas de volatilidad
        """
        metrics = {}
        
        # Volatilidad histórica
        for window in self.window_sizes:
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Anualizada
            metrics[f'volatility_{window}d'] = volatility
        
        # Volatilidad promedio
        metrics['volatility_mean'] = returns.std() * np.sqrt(252)
        
        # Volatilidad en ventanas móviles
        vol_5d = returns.rolling(window=5).std() * np.sqrt(252)
        vol_20d = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Ratio de volatilidad
        metrics['volatility_ratio'] = vol_5d / vol_20d
        
        # Percentiles de volatilidad
        vol_20d_clean = vol_20d.dropna()
        metrics['volatility_percentiles'] = {
            'p25': vol_20d_clean.quantile(0.25),
            'p50': vol_20d_clean.quantile(0.50),
            'p75': vol_20d_clean.quantile(0.75),
            'p90': vol_20d_clean.quantile(0.90),
            'p95': vol_20d_clean.quantile(0.95)
        }
        
        return metrics
    
    def calculate_var_cvar(self, returns: pd.Series, confidence_levels: List[float] = None) -> Dict:
        """
        Calcula Value at Risk (VaR) y Conditional Value at Risk (CVaR)
        
        Args:
            returns: Serie de rendimientos
            confidence_levels: Niveles de confianza
            
        Returns:
            Diccionario con VaR y CVaR
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99, 0.995]
        
        results = {}
        
        for level in confidence_levels:
            # VaR (pérdida máxima esperada con probabilidad given confidence level)
            var = np.percentile(returns.dropna(), (1 - level) * 100)
            
            # CVaR (pérdida esperada condicionada a superar el VaR)
            tail_returns = returns[returns <= var]
            cvar = tail_returns.mean() if len(tail_returns) > 0 else var
            
            results[f'var_{int(level*100)}'] = var
            results[f'cvar_{int(level*100)}'] = cvar
        
        return results
    
    def calculate_risk_metrics(self, returns: pd.Series, benchmark: pd.Series = None) -> Dict:
        """
        Calcula métricas de riesgo adicionales
        
        Args:
            returns: Serie de rendimientos
            benchmark: Rendimientos de referencia (opcional)
            
        Returns:
            Diccionario con métricas de riesgo
        """
        risk_metrics = {}
        
        # Métricas básicas
        risk_metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Máximo drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        risk_metrics['max_drawdown'] = drawdown.min()
        risk_metrics['max_drawdown_date'] = drawdown.idxmin()
        
        # Calmar ratio (retorno anualizado / máximo drawdown absoluto)
        annual_return = returns.mean() * 252
        risk_metrics['calmar_ratio'] = annual_return / abs(risk_metrics['max_drawdown']) if risk_metrics['max_drawdown'] != 0 else 0
        
        # Ratio de Sortino (retorno / desviación de rendimientos negativos)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        risk_metrics['sortino_ratio'] = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # VaR y CVaR
        var_cvar = self.calculate_var_cvar(returns)
        risk_metrics.update(var_cvar)
        
        # Si hay benchmark, calcular métricas relativas
        if benchmark is not None:
            aligned_returns = returns.align(benchmark, join='inner')[0]
            aligned_benchmark = returns.align(benchmark, join='inner')[1]
            
            # Beta
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            risk_metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha
            risk_metrics['alpha'] = aligned_returns.mean() - risk_metrics['beta'] * aligned_benchmark.mean()
            risk_metrics['alpha'] *= 252  # Anualizado
            
            # Tracking error
            excess_returns = aligned_returns - aligned_benchmark
            risk_metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            
            # Information ratio
            risk_metrics['information_ratio'] = excess_returns.mean() * 252 / risk_metrics['tracking_error'] if risk_metrics['tracking_error'] > 0 else 0
        
        return risk_metrics
    
    def detect_volatility_regimes(self, returns: pd.Series, n_regimes: int = 2) -> Dict:
        """
        Detecta regímenes de volatilidad usando clustering
        
        Args:
            returns: Serie de rendimientos
            n_regimes: Número de regímenes a detectar
            
        Returns:
            Información sobre regímenes de volatilidad
        """
        from sklearn.mixture import GaussianMixture
        
        # Preparar datos para clustering
        vol_data = returns.rolling(window=20).std().dropna().values.reshape(-1, 1)
        
        # Aplicar Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regime_labels = gmm.fit_predict(vol_data)
        
        # Calcular parámetros de cada régimen
        regimes = {}
        for regime in range(n_regimes):
            regime_mask = regime_labels == regime
            regime_data = vol_data[regime_mask]
            
            regimes[f'regime_{regime}'] = {
                'mean_volatility': np.mean(regime_data),
                'std_volatility': np.std(regime_data),
                'proportion': np.sum(regime_mask) / len(regime_labels),
                'labels': regime_labels[regime_mask]
            }
        
        # Clasificar regímenes como "baja" y "alta" volatilidad
        regime_means = [regimes[f'regime_{i}']['mean_volatility'] for i in range(n_regimes)]
        low_vol_regime = np.argmin(regime_means)
        high_vol_regime = np.argmax(regime_means)
        
        regimes['low_volatility_regime'] = low_vol_regime
        regimes['high_volatility_regime'] = high_vol_regime
        
        # Asignar etiquetas de régimen a la serie original
        regime_series = pd.Series(index=returns.index, dtype='object')
        regime_series[20:] = [f'regime_{label}' for label in regime_labels]
        
        regimes['regime_series'] = regime_series
        regimes['transition_dates'] = self._find_regime_transitions(regime_labels)
        
        return regimes
    
    def _find_regime_transitions(self, labels: np.ndarray) -> List[int]:
        """
        Encuentra fechas de transición entre regímenes
        
        Args:
            labels: Etiquetas de régimen
            
        Returns:
            Índices donde ocurren transiciones
        """
        transitions = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions.append(i)
        return transitions
    
    def stress_testing(self, returns: pd.Series, scenarios: Dict = None) -> Dict:
        """
        Realiza pruebas de estrés
        
        Args:
            returns: Serie de rendimientos
            scenarios: Escenarios de stress test
            
        Returns:
            Resultados de las pruebas de estrés
        """
        if scenarios is None:
            scenarios = {
                'market_crash': -0.20,  # Caída del 20%
                'moderate_decline': -0.10,  # Caída del 10%
                'high_volatility': 0.05,  # Aumento del 5% con alta volatilidad
                'correlation_shock': -0.15  # Shock de correlación
            }
        
        results = {}
        current_vol = returns.tail(20).std() * np.sqrt(252)
        
        for scenario, shock in scenarios.items():
            # Calcular impacto inmediato
            immediate_impact = shock
            
            # Calcular impacto en volatilidad (amplificación)
            vol_multiplier = 1.5 if 'volatility' in scenario else 1.0
            new_vol = current_vol * vol_multiplier
            
            # Simular recuperación
            if scenario == 'market_crash':
                recovery_periods = 60
                recovery_rate = 0.02 / 252  # 2% anual
            elif scenario == 'moderate_decline':
                recovery_periods = 30
                recovery_rate = 0.01 / 252  # 1% anual
            else:
                recovery_periods = 10
                recovery_rate = 0.005 / 252  # 0.5% anual
            
            results[scenario] = {
                'immediate_impact': immediate_impact,
                'volatility_impact': new_vol,
                'recovery_periods': recovery_periods,
                'recovery_rate': recovery_rate,
                'stress_level': 'Crítico' if abs(shock) > 0.15 else 'Alto' if abs(shock) > 0.10 else 'Medio'
            }
        
        return results
    
    def plot_volatility_analysis(self, prices: pd.Series, returns: pd.Series, save_path: str = None):
        """
        Visualiza el análisis de volatilidad
        
        Args:
            prices: Serie de precios
            returns: Serie de rendimientos
            save_path: Ruta para guardar la imagen
        """
        # Calcular métricas
        vol_metrics = self.calculate_volatility_metrics(returns)
        risk_metrics = self.calculate_risk_metrics(returns)
        regimes = self.detect_volatility_regimes(returns)
        stress_results = self.stress_testing(returns)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Volatilidad y Riesgo - NOESIS', fontsize=16, fontweight='bold')
        
        # Volatilidad móvil
        ax1 = axes[0, 0]
        for window in [5, 20, 60]:
            vol_key = f'volatility_{window}d'
            if vol_key in vol_metrics:
                ax1.plot(vol_metrics[vol_key].index, vol_metrics[vol_key].values, 
                        label=f'Volatilidad {window}d', linewidth=2)
        
        ax1.set_title('Volatilidad Móvil (Anualizada)')
        ax1.set_ylabel('Volatilidad')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Percentiles de volatilidad
        vol_20d = vol_metrics['volatility_20d'].dropna()
        percentiles = vol_metrics['volatility_percentiles']
        
        ax2 = axes[0, 1]
        ax2.plot(vol_20d.index, vol_20d.values, linewidth=2, label='Volatilidad 20d')
        ax2.axhline(y=percentiles['p75'], color='orange', linestyle='--', alpha=0.7, label='P75')
        ax2.axhline(y=percentiles['p90'], color='red', linestyle='--', alpha=0.7, label='P90')
        ax2.axhline(y=percentiles['p95'], color='darkred', linestyle='--', alpha=0.7, label='P95')
        
        ax2.set_title('Volatilidad vs Percentiles Históricos')
        ax2.set_ylabel('Volatilidad')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Distribución de rendimientos
        ax3 = axes[1, 0]
        ax3.hist(returns.dropna(), bins=50, alpha=0.7, density=True, color='skyblue')
        
        # Añadir estadísticas
        mean_return = returns.mean()
        std_return = returns.std()
        ax3.axvline(mean_return, color='green', linestyle='-', linewidth=2, label=f'Media: {mean_return:.4f}')
        ax3.axvline(mean_return - 2*std_return, color='red', linestyle='--', alpha=0.7, label='-2σ')
        ax3.axvline(mean_return + 2*std_return, color='red', linestyle='--', alpha=0.7, label='+2σ')
        
        ax3.set_title('Distribución de Rendimientos')
        ax3.set_xlabel('Rendimiento')
        ax3.set_ylabel('Densidad')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Regímenes de volatilidad
        ax4 = axes[1, 1]
        regime_series = regimes['regime_series']
        
        # Mapear regímenes a colores
        colors = {0: 'green', 1: 'red', 2: 'orange', 3: 'purple'}
        regime_colors = [colors.get(label, 'gray') for label in regime_series.iloc[20:].values if not pd.isna(label)]
        
        # Gráfico de líneas con colores por régimen
        for i, (date, label) in enumerate(regime_series.iloc[20:].items()):
            if not pd.isna(label):
                regime_num = int(label.split('_')[1])
                color = colors.get(regime_num, 'gray')
                ax4.scatter(date, prices.iloc[20+i], c=color, s=20, alpha=0.7)
        
        ax4.set_title('Regímenes de Volatilidad')
        ax4.set_ylabel('Precio')
        ax4.grid(True, alpha=0.3)
        
        # Añadir leyenda
        legend_elements = [plt.scatter([], [], c=colors.get(i, 'gray'), s=50, label=f'Regimen {i}') 
                          for i in range(len(regimes)-3)]  # -3 para excluir las claves adicionales
        ax4.legend(handles=legend_elements)
        
        # Drawdown
        ax5 = axes[2, 0]
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        ax5.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax5.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        ax5.set_title('Drawdown')
        ax5.set_ylabel('Drawdown')
        ax5.grid(True, alpha=0.3)
        
        # Información de riesgo
        ax6 = axes[2, 1]
        risk_text = f"""
Métricas de Riesgo:

Volatilidad Anualizada: {risk_metrics['volatility_mean']:.2%}

Ratios de Retorno-Riesgo:
• Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}
• Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}
• Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}

Riesgo Extremo:
• Max Drawdown: {risk_metrics['max_drawdown']:.2%}
• VaR 95%: {risk_metrics['var_95']:.2%}
• CVaR 95%: {risk_metrics['cvar_95']:.2%}

Pruebas de Estrés:
"""
        
        for scenario, result in stress_results.items():
            risk_text += f"• {scenario}: {result['immediate_impact']:.1%} ({result['stress_level']})\n"
        
        ax6.text(0.05, 0.95, risk_text, transform=ax6.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        
        return {
            'volatility_metrics': vol_metrics,
            'risk_metrics': risk_metrics,
            'regimes': regimes,
            'stress_tests': stress_results
        }

class CorrelationAnalyzer:
    """
    Analizador de correlaciones entre múltiples variables
    """
    
    def __init__(self):
        """Inicializa el analizador de correlaciones"""
        pass
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Calcula matriz de correlación
        
        Args:
            data: DataFrame con datos
            method: Método de correlación ('pearson', 'spearman', 'kendall')
            
        Returns:
            Matriz de correlación
        """
        return data.corr(method=method)
    
    def detect_structural_breaks(self, series1: pd.Series, series2: pd.Series, 
                               min_distance: int = 20) -> List[Dict]:
        """
        Detecta rupturas estructurales en la correlación
        
        Args:
            series1: Primera serie
            series2: Segunda serie
            min_distance: Distancia mínima entre rupturas
            
        Returns:
            Lista de rupturas estructurales detectadas
        """
        # Alinear series
        aligned_data = pd.concat([series1, series2], axis=1).dropna()
        rolling_corr = aligned_data.iloc[:, 1].rolling(window=min_distance).corr(aligned_data.iloc[:, 0])
        
        # Detectar cambios significativos
        changes = rolling_corr.diff().abs()
        threshold = changes.quantile(0.95)  # Top 5% de cambios
        
        break_points = []
        for i in range(min_distance, len(changes) - min_distance):
            if changes.iloc[i] > threshold:
                break_points.append({
                    'date': rolling_corr.index[i],
                    'correlation_before': rolling_corr.iloc[i-min_distance:i].mean(),
                    'correlation_after': rolling_corr.iloc[i:i+min_distance].mean(),
                    'change_magnitude': changes.iloc[i]
                })
        
        return break_points
    
    def analyze_regime_correlations(self, data: pd.DataFrame, regime_series: pd.Series) -> Dict:
        """
        Analiza correlaciones por régimen
        
        Args:
            data: DataFrame con datos
            serie_de_regimenes: Serie con etiquetas de régimen
            
        Returns:
            Correlaciones por régimen
        """
        correlations_by_regime = {}
        
        for regime in regime_series.unique():
            if pd.isna(regime):
                continue
            
            regime_mask = regime_series == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) > 10:  # Mínimo de observaciones
                correlations_by_regime[regime] = regime_data.corr()
        
        return correlations_by_regime
    
    def plot_correlation_analysis(self, data: pd.DataFrame, save_path: str = None):
        """
        Visualiza el análisis de correlaciones
        
        Args:
            data: DataFrame con datos
            save_path: Ruta para guardar la imagen
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Correlaciones - NOESIS', fontsize=16, fontweight='bold')
        
        # Matriz de correlación global
        corr_matrix = self.calculate_correlation_matrix(data)
        
        im1 = axes[0, 0].imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 0].set_title('Matriz de Correlación Global')
        axes[0, 0].set_xticks(range(len(corr_matrix.columns)))
        axes[0, 0].set_yticks(range(len(corr_matrix.columns)))
        axes[0, 0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        axes[0, 0].set_yticklabels(corr_matrix.columns)
        
        # Añadir valores de correlación
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = axes[0, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Correlaciones más importantes
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append({
                    'pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                    'correlation': abs(corr_matrix.iloc[i, j]),
                    'raw_correlation': corr_matrix.iloc[i, j]
                })
        
        corr_values.sort(key=lambda x: x['correlation'], reverse=True)
        
        top_correlations = corr_values[:10]  # Top 10 correlaciones
        
        y_pos = np.arange(len(top_correlations))
        correlations = [item['raw_correlation'] for item in top_correlations]
        pairs = [item['pair'] for item in top_correlations]
        
        colors = ['green' if corr > 0 else 'red' for corr in correlations]
        bars = axes[0, 1].barh(y_pos, correlations, color=colors, alpha=0.7)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(pairs, fontsize=8)
        axes[0, 1].set_xlabel('Correlación')
        axes[0, 1].set_title('Top 10 Correlaciones por Magnitud')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Evolución temporal de correlación (si hay más de 2 columnas)
        if len(data.columns) >= 2:
            rolling_corr = data.iloc[:, 1].rolling(window=30).corr(data.iloc[:, 0])
            axes[1, 0].plot(rolling_corr.index, rolling_corr.values, linewidth=2)
            axes[1, 0].set_title(f'Evolución Temporal: {data.columns[0]} vs {data.columns[1]}')
            axes[1, 0].set_ylabel('Correlación 30d')
            axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Estadísticas de correlación
            corr_mean = rolling_corr.mean()
            corr_std = rolling_corr.std()
            
            axes[1, 0].axhline(y=corr_mean, color='blue', linestyle='--', alpha=0.7, 
                             label=f'Media: {corr_mean:.3f}')
            axes[1, 0].axhline(y=corr_mean + corr_std, color='red', linestyle=':', alpha=0.7, 
                             label=f'+1σ: {corr_mean + corr_std:.3f}')
            axes[1, 0].axhline(y=corr_mean - corr_std, color='red', linestyle=':', alpha=0.7, 
                             label=f'-1σ: {corr_mean - corr_std:.3f}')
            axes[1, 0].legend()
        
        # Network de correlaciones
        axes[1, 1].set_title('Red de Correlaciones')
        
        # Crear network simple
        n_vars = len(data.columns)
        angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
        
        # Dibujar nodos
        for i, (angle, var) in enumerate(zip(angles, data.columns)):
            x = np.cos(angle)
            y = np.sin(angle)
            axes[1, 1].scatter(x, y, s=500, alpha=0.8)
            axes[1, 1].text(x*1.1, y*1.1, var, ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Dibujar conexiones (solo correlaciones fuertes)
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Solo correlaciones fuertes
                    x1, y1 = np.cos(angles[i]), np.sin(angles[i])
                    x2, y2 = np.cos(angles[j]), np.sin(angles[j])
                    
                    # Grosor de línea basado en correlación
                    linewidth = abs(corr_val) * 3
                    color = 'green' if corr_val > 0 else 'red'
                    alpha = 0.6 + abs(corr_val) * 0.4
                    
                    axes[1, 1].plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
        
        axes[1, 1].set_xlim(-1.5, 1.5)
        axes[1, 1].set_ylim(-1.5, 1.5)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        
        return {
            'correlation_matrix': corr_matrix,
            'top_correlations': top_correlations
        }

class InflectionPointDetector:
    """
    Detector de puntos de inflexión
    """
    
    def __init__(self, min_prominence: float = 0.02, min_distance: int = 10):
        """
        Inicializa el detector de puntos de inflexión
        
        Args:
            min_prominence: Prominencia mínima para considerar un punto
            min_distance: Distancia mínima entre puntos
        """
        self.min_prominence = min_prominence
        self.min_distance = min_distance
    
    def detect_peaks_and_troughs(self, series: pd.Series) -> List[InflectionPoint]:
        """
        Detecta picos y valles en la serie
        
        Args:
            series: Serie temporal
            
        Returns:
            Lista de puntos de inflexión
        """
        # Normalizar la serie para el análisis
        normalized_series = (series - series.mean()) / series.std()
        
        # Detectar picos (máximos locales)
        peaks, peak_properties = find_peaks(normalized_series.values, 
                                          prominence=self.min_prominence,
                                          distance=self.min_distance)
        
        # Detectar valles (mínimos locales)
        troughs, trough_properties = find_peaks(-normalized_series.values,
                                              prominence=self.min_prominence,
                                              distance=self.min_distance)
        
        inflection_points = []
        
        # Procesar picos
        for i, peak_idx in enumerate(peaks):
            peak_props = peak_properties[i]
            confidence = min(peak_props['prominence'] / normalized_series.std(), 1.0)
            
            inflection_points.append(InflectionPoint(
                date=series.index[peak_idx].strftime('%Y-%m-%d'),
                price=series.iloc[peak_idx],
                type='max',
                significance=peak_props['prominence'],
                confidence=confidence
            ))
        
        # Procesar valles
        for i, trough_idx in enumerate(troughs):
            trough_props = trough_properties[i]
            confidence = min(trough_props['prominence'] / normalized_series.std(), 1.0)
            
            inflection_points.append(InflectionPoint(
                date=series.index[trough_idx].strftime('%Y-%m-%d'),
                price=series.iloc[trough_idx],
                type='min',
                significance=trough_props['prominence'],
                confidence=confidence
            ))
        
        # Ordenar por fecha
        inflection_points.sort(key=lambda x: x.date)
        
        return inflection_points
    
    def detect_trend_changes(self, series: pd.Series, window: int = 20) -> List[InflectionPoint]:
        """
        Detecta cambios de tendencia usando pendientes móviles
        
        Args:
            series: Serie temporal
            window: Tamaño de ventana para cálculo de pendientes
            
        Returns:
            Lista de puntos de cambio de tendencia
        """
        slopes = []
        dates = []
        
        for i in range(window, len(series) - window):
            # Calcular pendiente en ventana
            x = np.arange(window * 2)
            y = series.iloc[i-window:i+window].values
            
            slope, _, r_value, _, _ = stats.linregress(x, y)
            slopes.append(slope)
            dates.append(series.index[i])
        
        slope_series = pd.Series(slopes, index=dates)
        
        # Detectar cambios en el signo de la pendiente
        changes = []
        for i in range(1, len(slope_series)):
            prev_slope = slope_series.iloc[i-1]
            curr_slope = slope_series.iloc[i]
            
            # Cambio de pendiente positiva a negativa o viceversa
            if (prev_slope > 0 and curr_slope < 0) or (prev_slope < 0 and curr_slope > 0):
                # Calcular confianza basada en la magnitud del cambio
                change_magnitude = abs(curr_slope - prev_slope)
                confidence = min(change_magnitude / abs(prev_slope), 1.0) if prev_slope != 0 else 0.5
                
                trend_type = 'cambio_bajista' if prev_slope > 0 else 'cambio_alcista'
                
                changes.append(InflectionPoint(
                    date=slope_series.index[i].strftime('%Y-%m-%d'),
                    price=series.loc[slope_series.index[i]],
                    type=trend_type,
                    significance=change_magnitude,
                    confidence=confidence
                ))
        
        return changes
    
    def analyze_invalidation_points(self, series: pd.Series, peaks: List[InflectionPoint]) -> Dict:
        """
        Analiza puntos de invalidación de tendencias
        
        Args:
            series: Serie temporal
            peaks: Lista de puntos de inflexión
            
        Returns:
            Análisis de puntos de invalidación
        """
        invalidation_analysis = {}
        
        # Clasificar puntos como soporte o resistencia
        support_points = [p for p in peaks if p.type == 'min']
        resistance_points = [p for p in peaks if p.type == 'max']
        
        # Analizar niveles de soporte
        if support_points:
            support_levels = [p.price for p in support_points[-5:]]  # Últimos 5 niveles
            avg_support = np.mean(support_levels)
            support_std = np.std(support_levels)
            
            # Detectar ruptura de soporte
            current_price = series.iloc[-1]
            invalidation_level = avg_support - 2 * support_std
            
            invalidation_analysis['support'] = {
                'average_level': avg_support,
                'std_deviation': support_std,
                'invalidation_level': invalidation_level,
                'current_break': current_price < invalidation_level,
                'strength': len(support_points) / len(peaks) if peaks else 0
            }
        
        # Analizar niveles de resistencia
        if resistance_points:
            resistance_levels = [p.price for p in resistance_points[-5:]]  # Últimos 5 niveles
            avg_resistance = np.mean(resistance_levels)
            resistance_std = np.std(resistance_levels)
            
            # Detectar ruptura de resistencia
            current_price = series.iloc[-1]
            breakout_level = avg_resistance + 2 * resistance_std
            
            invalidation_analysis['resistance'] = {
                'average_level': avg_resistance,
                'std_deviation': resistance_std,
                'breakout_level': breakout_level,
                'current_break': current_price > breakout_level,
                'strength': len(resistance_points) / len(peaks) if peaks else 0
            }
        
        return invalidation_analysis
    
    def plot_inflection_points(self, series: pd.Series, save_path: str = None):
        """
        Visualiza los puntos de inflexión detectados
        
        Args:
            series: Serie temporal
            save_path: Ruta para guardar la imagen
        """
        peaks_troughs = self.detect_peaks_and_troughs(series)
        trend_changes = self.detect_trend_changes(series)
        invalidation = self.analyze_invalidation_points(series, peaks_troughs)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Detección de Puntos de Inflexión - NOESIS', fontsize=16, fontweight='bold')
        
        # Gráfico principal con picos y valles
        axes[0].plot(series.index, series.values, linewidth=2, color='blue', label='Precio')
        
        # Marcar picos y valles
        max_points = [p for p in peaks_troughs if p.type == 'max']
        min_points = [p for p in peaks_troughs if p.type == 'min']
        
        if max_points:
            max_dates = [datetime.strptime(p.date, '%Y-%m-%d') for p in max_points]
            max_prices = [p.price for p in max_points]
            axes[0].scatter(max_dates, max_prices, color='red', s=100, marker='^', 
                          label='Picos', zorder=5)
        
        if min_points:
            min_dates = [datetime.strptime(p.date, '%Y-%m-%d') for p in min_points]
            min_prices = [p.price for p in min_points]
            axes[0].scatter(min_dates, min_prices, color='green', s=100, marker='v', 
                          label='Valles', zorder=5)
        
        # Marcar cambios de tendencia
        if trend_changes:
            change_dates = [datetime.strptime(p.date, '%Y-%m-%d') for p in trend_changes]
            change_prices = [p.price for p in trend_changes]
            axes[0].scatter(change_dates, change_prices, color='orange', s=150, marker='o', 
                          label='Cambios de Tendencia', zorder=6, edgecolors='black', linewidth=2)
        
        axes[0].set_title('Detección de Picos, Valles y Cambios de Tendencia')
        axes[0].set_ylabel('Precio')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Información de puntos de inflexión
        inflection_text = f"""
Puntos de Inflexión Detectados:

Picos (Máximos): {len(max_points)}
Valles (Mínimos): {len(min_points)}
Cambios de Tendencia: {len(trend_changes)}

Análisis de Soporte/Resistencia:
"""
        
        if 'support' in invalidation:
            support = invalidation['support']
            inflection_text += f"""
Soporte:
• Nivel promedio: {support['average_level']:.2f}
• Nivel de invalidación: {support['invalidation_level']:.2f}
• Ruptura actual: {'Sí' if support['current_break'] else 'No'}
"""
        
        if 'resistance' in invalidation:
            resistance = invalidation['resistance']
            inflection_text += f"""
Resistencia:
• Nivel promedio: {resistance['average_level']:.2f}
• Nivel de ruptura: {resistance['breakout_level']:.2f}
• Ruptura actual: {'Sí' if resistance['current_break'] else 'No'}
"""
        
        axes[0].text(0.02, 0.98, inflection_text, transform=axes[0].transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        
        # Análisis de pendientes
        slopes = []
        dates = []
        window = 20
        
        for i in range(window, len(series) - window):
            x = np.arange(window * 2)
            y = series.iloc[i-window:i+window].values
            slope, _, r_value, _, _ = stats.linregress(x, y)
            slopes.append(slope)
            dates.append(series.index[i])
        
        slope_series = pd.Series(slopes, index=dates)
        
        axes[1].plot(slope_series.index, slope_series.values, linewidth=2, color='purple')
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_title('Evolución de Pendientes (Ventana Móvil)')
        axes[1].set_ylabel('Pendiente')
        axes[1].grid(True, alpha=0.3)
        
        # Marcar cambios de signo
        for i in range(1, len(slope_series)):
            prev_slope = slope_series.iloc[i-1]
            curr_slope = slope_series.iloc[i]
            
            if (prev_slope > 0 and curr_slope < 0) or (prev_slope < 0 and curr_slope > 0):
                axes[1].scatter(slope_series.index[i], 0, color='red', s=100, 
                              marker='o', zorder=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        
        return {
            'peaks_troughs': peaks_troughs,
            'trend_changes': trend_changes,
            'invalidation_analysis': invalidation
        }

class AlertSystem:
    """
    Sistema de alertas para cambios significativos
    """
    
    def __init__(self, alert_rules: Dict = None):
        """
        Inicializa el sistema de alertas
        
        Args:
            alert_rules: Reglas de alerta personalizadas
        """
        self.alert_rules = alert_rules or self._default_alert_rules()
        self.active_alerts = []
        self.alert_history = []
        
    def _default_alert_rules(self) -> Dict:
        """
        Define reglas de alerta por defecto
        
        Returns:
            Diccionario con reglas de alerta
        """
        return {
            'volatility_spike': {
                'threshold': 0.95,  # Percentil 95 de volatilidad
                'level': AlertLevel.ALTO,
                'message': 'Spike de volatilidad detectado'
            },
            'trend_change': {
                'threshold': 0.8,  # Cambio de tendencia con 80% confianza
                'level': AlertLevel.MEDIO,
                'message': 'Cambio de tendencia detectado'
            },
            'support_resistance_break': {
                'threshold': 2.0,  # Ruptura de 2 desviaciones estándar
                'level': AlertLevel.CRITICO,
                'message': 'Ruptura de nivel de soporte/resistencia'
            },
            'correlation_break': {
                'threshold': 0.5,  # Cambio de correlación de 0.5
                'level': AlertLevel.MEDIO,
                'message': 'Cambio significativo en correlaciones'
            },
            'extreme_movement': {
                'threshold': 0.05,  # Movimiento del 5% en un día
                'level': AlertLevel.ALTO,
                'message': 'Movimiento extremo detectado'
            }
        }
    
    def check_volatility_alert(self, current_vol: float, vol_percentiles: Dict[str, float]) -> Optional[Alert]:
        """
        Verifica alertas de volatilidad
        
        Args:
            vol_actual: Volatilidad actual
            percentiles_vol: Percentiles de volatilidad histórica
            
        Returns:
            Alerta si se cumple la condición
        """
        p95 = vol_percentiles.get('p95', np.inf)
        
        if current_vol > p95:
            return Alert(
                level=AlertLevel.ALTO,
                message=f"Volatilidad extrema: {current_vol:.2%} (P95: {p95:.2%})",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metric='volatility',
                value=current_vol
            )
        return None
    
    def check_trend_alert(self, trend_analysis: Dict) -> Optional[Alert]:
        """
        Verifica alertas de cambio de tendencia
        
        Args:
            trend_analysis: Análisis de tendencias
            
        Returns:
            Alerta si se detecta cambio de tendencia
        """
        consensus = trend_analysis.get('consensus', TrendDirection.INDETERMINADO)
        consensus_strength = trend_analysis.get('consensus_strength', 0)
        
        # Cambios de tendencia con alta confianza
        if consensus_strength > 0.8 and consensus != TrendDirection.LATERAL:
            return Alert(
                level=AlertLevel.MEDIO,
                message=f"Cambio de tendencia detectado: {consensus.value} (Confianza: {consensus_strength:.1%})",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metric='trend',
                value=consensus_strength
            )
        return None
    
    def check_extreme_movement_alert(self, returns: pd.Series, threshold: float = 0.05) -> Optional[Alert]:
        """
        Verifica alertas de movimientos extremos
        
        Args:
            returns: Serie de rendimientos
            threshold: Umbral para considerar movimiento extremo
            
        Returns:
            Alerta si se detecta movimiento extremo
        """
        if len(returns) == 0:
            return None
            
        latest_return = returns.iloc[-1]
        
        if abs(latest_return) > threshold:
            return Alert(
                level=AlertLevel.ALTO if abs(latest_return) > 0.10 else AlertLevel.MEDIO,
                message=f"Movimiento extremo: {latest_return:.2%}",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metric='return',
                value=latest_return
            )
        return None
    
    def check_support_resistance_alert(self, current_price: float, support: Dict, resistance: Dict) -> Optional[Alert]:
        """
        Verifica alertas de ruptura de soporte/resistencia
        
        Args:
            current_price: Precio actual
            support: Datos de soporte
            resistance: Datos de resistencia
            
        Returns:
            Alerta si se detecta ruptura
        """
        alerts = []
        
        # Verificar ruptura de soporte
        if support.get('current_break', False):
            alerts.append(Alert(
                level=AlertLevel.CRITICO,
                message=f"Ruptura de soporte en {support['invalidation_level']:.2f}",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metric='support_break',
                value=current_price
            ))
        
        # Verificar ruptura de resistencia
        if resistance.get('current_break', False):
            alerts.append(Alert(
                level=AlertLevel.ALTO,
                message=f"Ruptura de resistencia en {resistance['breakout_level']:.2f}",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metric='resistance_break',
                value=current_price
            ))
        
        return alerts if alerts else None
    
    def check_correlation_alert(self, current_corr: float, historical_corr: List[float]) -> Optional[Alert]:
        """
        Verifica alertas de cambio en correlaciones
        
        Args:
            current_corr: Correlación actual
            historical_corr: Correlaciones históricas
            
        Returns:
            Alerta si se detecta cambio significativo
        """
        if len(historical_corr) < 20:
            return None
        
        # Calcular desviación estándar de correlaciones históricas
        corr_mean = np.mean(historical_corr[-20:])  # Últimas 20 observaciones
        corr_std = np.std(historical_corr[-20:])
        
        # Verificar cambio significativo
        if abs(current_corr - corr_mean) > 2 * corr_std:
            return Alert(
                level=AlertLevel.MEDIO,
                message=f"Cambio de correlación: {current_corr:.3f} (Media histórica: {corr_mean:.3f})",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metric='correlation',
                value=current_corr
            )
        return None
    
    def run_comprehensive_alert_check(self, data: Dict) -> List[Alert]:
        """
        Ejecuta verificación completa de alertas
        
        Args:
            data: Diccionario con todos los datos necesarios
            
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        
        # Alerta de volatilidad
        if 'volatility_metrics' in data and 'current_vol' in data['volatility_metrics']:
            vol_alert = self.check_volatility_alert(
                data['volatility_metrics']['current_vol'],
                data['volatility_metrics'].get('percentiles', {})
            )
            if vol_alert:
                alerts.append(vol_alert)
        
        # Alerta de tendencia
        if 'trend_analysis' in data:
            trend_alert = self.check_trend_alert(data['trend_analysis'])
            if trend_alert:
                alerts.append(trend_alert)
        
        # Alerta de movimiento extremo
        if 'returns' in data:
            movement_alert = self.check_extreme_movement_alert(data['returns'])
            if movement_alert:
                alerts.append(movement_alert)
        
        # Alerta de soporte/resistencia
        if 'invalidation_analysis' in data:
            invalidation = data['invalidation_analysis']
            sr_alerts = self.check_support_resistance_alert(
                data.get('current_price', 0),
                invalidation.get('support', {}),
                invalidation.get('resistance', {})
            )
            if sr_alerts:
                alerts.extend(sr_alerts)
        
        # Alerta de correlación
        if 'current_correlation' in data and 'historical_correlations' in data:
            corr_alert = self.check_correlation_alert(
                data['current_correlation'],
                data['historical_correlations']
            )
            if corr_alert:
                alerts.append(corr_alert)
        
        # Actualizar historial de alertas
        self.alert_history.extend(alerts)
        self.active_alerts = [alert for alert in alerts if alert.level in [AlertLevel.ALTO, AlertLevel.CRITICO]]
        
        return alerts
    
    def generate_alert_report(self, alerts: List[Alert]) -> str:
        """
        Genera reporte de alertas
        
        Args:
            alerts: Lista de alertas
            
        Returns:
            Reporte en formato texto
        """
        if not alerts:
            return "No se han generado alertas en este período."
        
        # Agrupar alertas por nivel
        alerts_by_level = {}
        for alert in alerts:
            if alert.level not in alerts_by_level:
                alerts_by_level[alert.level] = []
            alerts_by_level[alert.level].append(alert)
        
        # Generar reporte
        report = "REPORTE DE ALERTAS - NOESIS\n"
        report += "=" * 50 + "\n"
        report += f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total de alertas: {len(alerts)}\n\n"
        
        for level in [AlertLevel.CRITICO, AlertLevel.ALTO, AlertLevel.MEDIO, AlertLevel.BAJO]:
            if level in alerts_by_level:
                level_alerts = alerts_by_level[level]
                report += f"{level.value.upper()} ({len(level_alerts)} alertas):\n"
                report += "-" * 30 + "\n"
                
                for alert in level_alerts:
                    report += f"• {alert.message}\n"
                    report += f"  Métrica: {alert.metric}, Valor: {alert.value:.4f}\n"
                    report += f"  Timestamp: {alert.timestamp}\n\n"
        
        return report
    
    def save_alert_log(self, filename: str = None):
        """
        Guarda el log de alertas
        
        Args:
            filename: Nombre del archivo
        """
        if filename is None:
            filename = f"noesis_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convertir alertas a formato serializable
        alert_data = []
        for alert in self.alert_history:
            alert_dict = {
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'metric': alert.metric,
                'value': alert.value
            }
            alert_data.append(alert_dict)
        
        # Guardar como JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(alert_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Log de alertas guardado en: {filename}")

def generate_sample_data() -> pd.DataFrame:
    """
    Genera datos de ejemplo para demostración
    
    Returns:
        DataFrame con datos sintéticos
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    n_days = len(dates)
    
    # Simular precio con tendencia y estacionalidad
    trend = np.linspace(100, 150, n_days)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # Estacionalidad anual
    noise = np.random.normal(0, 5, n_days)
    price = trend + seasonal + noise
    
    # Rendimientos
    returns = pd.Series(np.diff(np.log(price)), index=dates[1:])
    
    # Crear DataFrame con múltiples variables
    data = pd.DataFrame({
        'precio': pd.Series(price, index=dates),
        'rendimiento': returns,
        'volumen': np.random.lognormal(10, 1, n_days),
        'vix': np.maximum(10, 20 + 5 * np.sin(2 * np.pi * np.arange(n_days) / 252) + np.random.normal(0, 3, n_days))
    })
    
    return data

def main_analysis_example():
    """
    Ejemplo principal de análisis completo
    """
    print("NOESIS - Sistema de Análisis de Tendencias")
    print("=" * 50)
    
    # Generar datos de ejemplo
    print("Generando datos de ejemplo...")
    data = generate_sample_data()
    
    # Inicializar analizadores
    trend_detector = TrendDetector()
    seasonality_analyzer = SeasonalityAnalyzer()
    volatility_analyzer = VolatilityAnalyzer()
    correlation_analyzer = CorrelationAnalyzer()
    inflection_detector = InflectionPointDetector()
    alert_system = AlertSystem()
    
    print("Ejecutando análisis de tendencias...")
    # Análisis de tendencias
    trend_analysis = trend_detector.detect_multiple_timeframes(data['precio'])
    
    print("Ejecutando análisis de estacionalidad...")
    # Análisis de estacionalidad
    seasonality_results = seasonality_analyzer.plot_seasonality_analysis(data['precio'])
    
    print("Ejecutando análisis de volatilidad...")
    # Análisis de volatilidad
    volatility_results = volatility_analyzer.plot_volatility_analysis(
        data['precio'], data['rendimiento']
    )
    
    print("Ejecutando análisis de correlaciones...")
    # Análisis de correlaciones
    correlation_results = correlation_analyzer.plot_correlation_analysis(
        data[['precio', 'vix', 'volumen']]
    )
    
    print("Detectando puntos de inflexión...")
    # Detección de puntos de inflexión
    inflection_results = inflection_detector.plot_inflection_points(data['precio'])
    
    print("Generando alertas...")
    # Preparar datos para sistema de alertas
    alert_data = {
        'volatility_metrics': {
            'current_vol': volatility_results['volatility_metrics']['volatility_20d'].iloc[-1],
            'percentiles': volatility_results['volatility_metrics']['volatility_percentiles']
        },
        'trend_analysis': trend_analysis,
        'returns': data['rendimiento'],
        'current_price': data['precio'].iloc[-1],
        'invalidation_analysis': inflection_results['invalidation_analysis'],
        'current_correlation': correlation_results['correlation_matrix'].iloc[0, 1],
        'historical_correlations': [0.3] * 50  # Ejemplo
    }
    
    # Generar alertas
    alerts = alert_system.run_comprehensive_alert_check(alert_data)
    
    # Mostrar reporte de alertas
    alert_report = alert_system.generate_alert_report(alerts)
    print("\n" + alert_report)
    
    # Guardar log de alertas
    alert_system.save_alert_log("noesis_demo_alerts.json")
    
    print("Análisis completado!")
    
    return {
        'trend_analysis': trend_analysis,
        'seasonality_results': seasonality_results,
        'volatility_results': volatility_results,
        'correlation_results': correlation_results,
        'inflection_results': inflection_results,
        'alerts': alerts
    }

if __name__ == "__main__":
    # Ejecutar ejemplo principal
    results = main_analysis_example()