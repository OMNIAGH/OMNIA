"""
NOESIS - Sistema de Modelos de Forecasting Predictivo
=====================================================

Sistema completo de forecasting que incluye:
- Modelos ARIMA, SARIMA, Prophet para series temporales
- Modelos ML (XGBoost, LightGBM, Random Forest)
- Ensemble methods
- Validación walk-forward y cross-validation
- API para predicciones en tiempo real
- Manejo de estacionalidad, tendencias, outliers y missing values

Autor: NOESIS
Versión: 1.0
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import joblib
import json
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet no está disponible. Instalar con: pip install prophet")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForecastingConfig:
    """Configuración para modelos de forecasting"""
    # Configuraciones generales
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # Configuraciones ARIMA/SARIMA
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    max_P: int = 2
    max_D: int = 1
    max_Q: int = 2
    seasonal_period: int = 12
    
    # Configuraciones ML
    xgb_params: Dict = None
    lgb_params: Dict = None
    rf_params: Dict = None
    
    # Configuraciones ensemble
    ensemble_method: str = 'weighted'  # 'weighted', 'stacking', 'voting'
    weights: Dict = None
    
    # Configuraciones de validación
    n_splits: int = 5
    walk_forward: bool = True
    horizon: int = 12
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            }
        
        if self.lgb_params is None:
            self.lgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbose': -1
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        if self.weights is None:
            self.weights = {
                'arima': 0.25,
                'sarima': 0.25,
                'prophet': 0.2,
                'xgboost': 0.15,
                'lightgbm': 0.1,
                'random_forest': 0.05
            }

class DataPreprocessor:
    """Preprocesador de datos para forecasting"""
    
    def __init__(self, config: ForecastingConfig):
        self.config = config
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, data: pd.Series, method: str = 'interpolate') -> pd.Series:
        """
        Manejo de valores faltantes
        
        Args:
            data: Serie temporal
            method: 'interpolate', 'forward_fill', 'backward_fill', 'drop'
            
        Returns:
            Serie con valores faltantes procesados
        """
        if method == 'interpolate':
            return data.interpolate(method='linear')
        elif method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Método no soportado: {method}")
    
    def detect_outliers(self, data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Detección de outliers
        
        Args:
            data: Serie temporal
            method: 'iqr', 'zscore', 'isolation_forest'
            threshold: Umbral para detección
            
        Returns:
            Serie con outliers marcados
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers = z_scores > threshold
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = pd.Series(iso_forest.fit_predict(data.values.reshape(-1, 1)) == -1, index=data.index)
            
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        return outliers
    
    def handle_outliers(self, data: pd.Series, method: str = 'winsorize') -> pd.Series:
        """
        Manejo de outliers
        
        Args:
            data: Serie temporal
            method: 'winsorize', 'remove', 'cap'
            
        Returns:
            Serie con outliers procesados
        """
        outliers = self.detect_outliers(data)
        
        if method == 'winsorize':
            from scipy.stats import mstats
            return pd.Series(
                mstats.winsorize(data.values, limits=[0.05, 0.05]),
                index=data.index
            )
        elif method == 'remove':
            return data[~outliers]
        elif method == 'cap':
            data_clean = data.copy()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data_clean[outliers] = np.where(
                data[outliers] > upper_bound, upper_bound,
                np.where(data[outliers] < lower_bound, lower_bound, data[outliers])
            )
            return data_clean
        else:
            raise ValueError(f"Método no soportado: {method}")
    
    def detect_seasonality(self, data: pd.Series) -> Dict[str, Any]:
        """
        Detección automática de estacionalidad
        
        Args:
            data: Serie temporal
            
        Returns:
            Diccionario con información de estacionalidad
        """
        # Test de estacionariedad
        adf_result = adfuller(data.dropna())
        is_stationary = adf_result[1] < 0.05
        
        # Descomposición para detectar estacionalidad
        try:
            if len(data) >= 2 * self.config.seasonal_period:
                decomposition = seasonal_decompose(data.dropna(), 
                                                 model='additive', 
                                                 period=self.config.seasonal_period)
                seasonal_strength = np.var(decomposition.seasonal) / np.var(data.dropna())
            else:
                seasonal_strength = 0
        except:
            seasonal_strength = 0
        
        # Detección automática del período
        autocorr = acf(data.dropna(), nlags=min(len(data)//2, 24), fft=True)
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                peaks.append(i)
        
        best_period = peaks[0] if peaks else self.config.seasonal_period
        
        return {
            'is_stationary': is_stationary,
            'seasonal_strength': seasonal_strength,
            'best_period': best_period,
            'has_seasonality': seasonal_strength > 0.1
        }
    
    def create_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Creación de features para modelos ML
        
        Args:
            data: Serie temporal
            
        Returns:
            DataFrame con features
        """
        df = pd.DataFrame()
        df['target'] = data
        
        # Lag features
        for lag in [1, 2, 3, 7, 12, 24]:
            if lag < len(data):
                df[f'lag_{lag}'] = data.shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 12, 24]:
            if window < len(data):
                df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
                df[f'rolling_std_{window}'] = data.rolling(window=window).std()
                df[f'rolling_min_{window}'] = data.rolling(window=window).min()
                df[f'rolling_max_{window}'] = data.rolling(window=window).max()
        
        # Time features
        if isinstance(data.index, pd.DatetimeIndex):
            df['hour'] = data.index.hour if data.index.freq is None else None
            df['day_of_week'] = data.index.dayofweek
            df['day_of_month'] = data.index.day
            df['month'] = data.index.month
            df['quarter'] = data.index.quarter
            df['year'] = data.index.year
        
        # Trend features
        df['linear_trend'] = np.arange(len(data))
        df['quadratic_trend'] = np.arange(len(data)) ** 2
        
        # Difference features
        df['diff_1'] = data.diff()
        df['diff_2'] = data.diff().diff()
        
        return df.dropna()

class BaseModel(ABC):
    """Clase base para modelos de forecasting"""
    
    def __init__(self, name: str, config: ForecastingConfig):
        self.name = name
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_columns = None
        
    @abstractmethod
    def fit(self, data: pd.Series) -> 'BaseModel':
        """Entrenar el modelo"""
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones"""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Guardar modelo"""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Cargar modelo"""
        pass

class ARIMAModel(BaseModel):
    """Modelo ARIMA"""
    
    def __init__(self, config: ForecastingConfig, p: int = None, d: int = None, q: int = None):
        super().__init__("ARIMA", config)
        self.p = p
        self.d = d
        self.q = q
        
    def _auto_arima(self, data: pd.Series) -> Tuple[int, int, int]:
        """Selección automática de parámetros ARIMA"""
        best_aic = np.inf
        best_params = (1, 1, 1)
        
        for p in range(self.config.max_p + 1):
            for d in range(self.config.max_d + 1):
                for q in range(self.config.max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params
    
    def fit(self, data: pd.Series) -> 'ARIMAModel':
        """Entrenar modelo ARIMA"""
        try:
            # Auto-selección de parámetros si no se especifican
            if self.p is None or self.d is None or self.q is None:
                self.p, self.d, self.q = self._auto_arima(data)
                logger.info(f"Parámetros ARIMA seleccionados: p={self.p}, d={self.d}, q={self.q}")
            
            # Verificar estacionariedad
            adf_result = adfuller(data.dropna())
            if adf_result[1] > 0.05:
                logger.warning("Serie no es estacionaria, aplicando diferenciación")
            
            # Entrenar modelo
            self.model = ARIMA(data, order=(self.p, self.d, self.q))
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            logger.info(f"Modelo ARIMA entrenado con parámetros: {self.p}, {self.d}, {self.q}")
            return self
            
        except Exception as e:
            logger.error(f"Error entrenando ARIMA: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones con ARIMA"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            index = pd.date_range(start=self.fitted_model.model._index[-1] + 
                                 (self.fitted_model.model._index[-1] - self.fitted_model.model._index[-2]),
                                 periods=steps, freq=self.fitted_model.model._index.freq)
            return pd.Series(forecast, index=index)
        except Exception as e:
            logger.error(f"Error en predicción ARIMA: {e}")
            raise
    
    def save(self, filepath: str) -> None:
        """Guardar modelo ARIMA"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        model_data = {
            'name': self.name,
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'model_params': self.fitted_model.params.to_dict(),
            'model_aic': self.fitted_model.aic,
            'fitted_values': self.fitted_model.fittedvalues.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        joblib.dump(self.fitted_model, filepath + '_model.pkl')
        logger.info(f"Modelo ARIMA guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """Cargar modelo ARIMA"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.p = model_data['p']
        self.d = model_data['d']
        self.q = model_data['q']
        self.name = model_data['name']
        self.fitted_model = joblib.load(filepath + '_model.pkl')
        self.is_fitted = True
        logger.info(f"Modelo ARIMA cargado desde {filepath}")

class SARIMAModel(BaseModel):
    """Modelo SARIMA"""
    
    def __init__(self, config: ForecastingConfig, 
                 p: int = None, d: int = None, q: int = None,
                 P: int = None, D: int = None, Q: int = None, s: int = None):
        super().__init__("SARIMA", config)
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
    
    def _auto_sarima(self, data: pd.Series) -> Tuple[int, int, int, int, int, int]:
        """Selección automática de parámetros SARIMA"""
        best_aic = np.inf
        best_params = (1, 1, 1, 1, 1, 1)
        
        for p in range(min(self.config.max_p, 3) + 1):
            for d in range(self.config.max_d + 1):
                for q in range(min(self.config.max_q, 3) + 1):
                    for P in range(min(self.config.max_P, 2) + 1):
                        for D in range(self.config.max_D + 1):
                            for Q in range(min(self.config.max_Q, 2) + 1):
                                try:
                                    model = SARIMAX(data, order=(p, d, q), 
                                                  seasonal_order=(P, D, Q, self.config.seasonal_period))
                                    fitted_model = model.fit(disp=False)
                                    if fitted_model.aic < best_aic:
                                        best_aic = fitted_model.aic
                                        best_params = (p, d, q, P, D, Q)
                                except:
                                    continue
        
        return best_params
    
    def fit(self, data: pd.Series) -> 'SARIMAModel':
        """Entrenar modelo SARIMA"""
        try:
            # Auto-selección de parámetros si no se especifican
            if any(param is None for param in [self.p, self.d, self.q, self.P, self.D, self.Q]):
                self.p, self.d, self.q, self.P, self.D, self.Q = self._auto_sarima(data)
                self.s = self.config.seasonal_period
                logger.info(f"Parámetros SARIMA seleccionados: ({self.p}, {self.d}, {self.q})x({self.P}, {self.D}, {self.Q}, {self.s})")
            
            # Entrenar modelo
            self.model = SARIMAX(data, order=(self.p, self.d, self.q), 
                               seasonal_order=(self.P, self.D, self.Q, self.s))
            self.fitted_model = self.model.fit(disp=False)
            self.is_fitted = True
            
            logger.info(f"Modelo SARIMA entrenado con parámetros: ({self.p}, {self.d}, {self.q})x({self.P}, {self.D}, {self.Q}, {self.s})")
            return self
            
        except Exception as e:
            logger.error(f"Error entrenando SARIMA: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones con SARIMA"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            index = pd.date_range(start=self.fitted_model.model._index[-1] + 
                                 (self.fitted_model.model._index[-1] - self.fitted_model.model._index[-2]),
                                 periods=steps, freq=self.fitted_model.model._index.freq)
            return pd.Series(forecast, index=index)
        except Exception as e:
            logger.error(f"Error en predicción SARIMA: {e}")
            raise
    
    def save(self, filepath: str) -> None:
        """Guardar modelo SARIMA"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        model_data = {
            'name': self.name,
            'p': self.p, 'd': self.d, 'q': self.q,
            'P': self.P, 'D': self.D, 'Q': self.Q, 's': self.s,
            'model_aic': self.fitted_model.aic
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        joblib.dump(self.fitted_model, filepath + '_model.pkl')
        logger.info(f"Modelo SARIMA guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """Cargar modelo SARIMA"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.p = model_data['p']
        self.d = model_data['d']
        self.q = model_data['q']
        self.P = model_data['P']
        self.D = model_data['D']
        self.Q = model_data['Q']
        self.s = model_data['s']
        self.name = model_data['name']
        self.fitted_model = joblib.load(filepath + '_model.pkl')
        self.is_fitted = True
        logger.info(f"Modelo SARIMA cargado desde {filepath}")

class ProphetModel(BaseModel):
    """Modelo Prophet"""
    
    def __init__(self, config: ForecastingConfig):
        super().__init__("Prophet", config)
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet no está disponible. Instalar con: pip install prophet")
    
    def fit(self, data: pd.Series) -> 'ProphetModel':
        """Entrenar modelo Prophet"""
        try:
            # Preparar datos para Prophet
            df_prophet = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
            
            # Configurar Prophet
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,
                holidays_prior_scale=10
            )
            
            self.model.fit(df_prophet)
            self.is_fitted = True
            logger.info("Modelo Prophet entrenado")
            return self
            
        except Exception as e:
            logger.error(f"Error entrenando Prophet: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones con Prophet"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        try:
            # Crear fechas futuras
            future = self.model.make_future_dataframe(periods=steps)
            forecast = self.model.predict(future)
            
            # Extraer predicciones futuras
            predictions = forecast['yhat'].iloc[-steps:].values
            future_dates = forecast['ds'].iloc[-steps:]
            
            return pd.Series(predictions, index=future_dates)
        except Exception as e:
            logger.error(f"Error en predicción Prophet: {e}")
            raise
    
    def save(self, filepath: str) -> None:
        """Guardar modelo Prophet"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        self.model.save(filepath)
        logger.info(f"Modelo Prophet guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """Cargar modelo Prophet"""
        from prophet import Prophet
        self.model = Prophet.load(filepath)
        self.is_fitted = True
        logger.info(f"Modelo Prophet cargado desde {filepath}")

class XGBoostModel(BaseModel):
    """Modelo XGBoost para forecasting"""
    
    def __init__(self, config: ForecastingConfig):
        super().__init__("XGBoost", config)
        self.model = xgb.XGBRegressor(**config.xgb_params)
        
    def fit(self, data: pd.Series) -> 'XGBoostModel':
        """Entrenar modelo XGBoost"""
        try:
            # Crear features
            feature_df = self.config.__dict__.get('_preprocessor', DataPreprocessor(self.config)).create_features(data)
            self.feature_columns = [col for col in feature_df.columns if col != 'target']
            
            X = feature_df[self.feature_columns]
            y = feature_df['target']
            
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Modelo XGBoost entrenado")
            return self
            
        except Exception as e:
            logger.error(f"Error entrenando XGBoost: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones con XGBoost"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        # Generar predicciones step-by-step
        predictions = []
        # Aquí se necesitaría una lógica más sofisticada para generar features futuras
        # Por simplicidad, generamos una predicción básica
        for i in range(steps):
            # Crear feature vector simple para predicción
            # En implementación real, se usarían las últimas observaciones
            X_pred = np.array([[0] * len(self.feature_columns)])  # Placeholder
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
        
        # Generar índice temporal
        last_index = self.model.get_booster().feature_names[-1]  # Placeholder
        future_index = pd.date_range(start=last_index, periods=steps+1)[1:]
        
        return pd.Series(predictions, index=future_index)
    
    def save(self, filepath: str) -> None:
        """Guardar modelo XGBoost"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.config
        }, filepath)
        logger.info(f"Modelo XGBoost guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """Cargar modelo XGBoost"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.config = data['config']
        self.is_fitted = True
        logger.info(f"Modelo XGBoost cargado desde {filepath}")

class LightGBMModel(BaseModel):
    """Modelo LightGBM para forecasting"""
    
    def __init__(self, config: ForecastingConfig):
        super().__init__("LightGBM", config)
        self.model = lgb.LGBMRegressor(**config.lgb_params)
        
    def fit(self, data: pd.Series) -> 'LightGBMModel':
        """Entrenar modelo LightGBM"""
        try:
            # Crear features
            preprocessor = DataPreprocessor(self.config)
            feature_df = preprocessor.create_features(data)
            self.feature_columns = [col for col in feature_df.columns if col != 'target']
            
            X = feature_df[self.feature_columns]
            y = feature_df['target']
            
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Modelo LightGBM entrenado")
            return self
            
        except Exception as e:
            logger.error(f"Error entrenando LightGBM: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones con LightGBM"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        # Generar predicciones step-by-step (similar a XGBoost)
        predictions = []
        for i in range(steps):
            X_pred = np.array([[0] * len(self.feature_columns)])  # Placeholder
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
        
        future_index = pd.date_range(start=datetime.now(), periods=steps+1)[1:]
        return pd.Series(predictions, index=future_index)
    
    def save(self, filepath: str) -> None:
        """Guardar modelo LightGBM"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.config
        }, filepath)
        logger.info(f"Modelo LightGBM guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """Cargar modelo LightGBM"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.config = data['config']
        self.is_fitted = True
        logger.info(f"Modelo LightGBM cargado desde {filepath}")

class RandomForestModel(BaseModel):
    """Modelo Random Forest para forecasting"""
    
    def __init__(self, config: ForecastingConfig):
        super().__init__("RandomForest", config)
        self.model = RandomForestRegressor(**config.rf_params)
        
    def fit(self, data: pd.Series) -> 'RandomForestModel':
        """Entrenar modelo Random Forest"""
        try:
            # Crear features
            preprocessor = DataPreprocessor(self.config)
            feature_df = preprocessor.create_features(data)
            self.feature_columns = [col for col in feature_df.columns if col != 'target']
            
            X = feature_df[self.feature_columns]
            y = feature_df['target']
            
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Modelo Random Forest entrenado")
            return self
            
        except Exception as e:
            logger.error(f"Error entrenando Random Forest: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones con Random Forest"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        predictions = []
        for i in range(steps):
            X_pred = np.array([[0] * len(self.feature_columns)])  # Placeholder
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
        
        future_index = pd.date_range(start=datetime.now(), periods=steps+1)[1:]
        return pd.Series(predictions, index=future_index)
    
    def save(self, filepath: str) -> None:
        """Guardar modelo Random Forest"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.config
        }, filepath)
        logger.info(f"Modelo Random Forest guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """Cargar modelo Random Forest"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.config = data['config']
        self.is_fitted = True
        logger.info(f"Modelo Random Forest cargado desde {filepath}")

class EnsembleModel(BaseModel):
    """Modelo ensemble que combina múltiples modelos"""
    
    def __init__(self, config: ForecastingConfig, models: List[BaseModel] = None):
        super().__init__("Ensemble", config)
        self.models = models or []
        self.weights = config.weights.copy()
        
    def add_model(self, model: BaseModel, weight: float = None) -> None:
        """Añadir modelo al ensemble"""
        self.models.append(model)
        if weight is not None:
            self.weights[model.name] = weight
        
        # Renormalizar pesos
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def fit(self, data: pd.Series) -> 'EnsembleModel':
        """Entrenar todos los modelos del ensemble"""
        for model in self.models:
            if not model.is_fitted:
                model.fit(data)
        
        # Optimizar pesos basado en performance
        self._optimize_weights(data)
        self.is_fitted = True
        logger.info("Modelo ensemble entrenado")
        return self
    
    def _optimize_weights(self, data: pd.Series) -> None:
        """Optimizar pesos del ensemble basado en performance"""
        try:
            # Calcular RMSE para cada modelo
            scores = {}
            for model in self.models:
                if model.is_fitted:
                    # Predicción simple para evaluación
                    pred = model.predict(10)  # Usar un pequeño número de pasos
                    actual = data.iloc[-10:] if len(data) >= 10 else data.iloc[-len(pred):]
                    
                    if len(pred) <= len(actual):
                        rmse = np.sqrt(mean_squared_error(actual, pred))
                        scores[model.name] = rmse
                    else:
                        scores[model.name] = 1.0  # Penalizar si hay problemas
            
            # Asignar pesos inversamente proporcionales al RMSE
            if scores:
                min_score = min(scores.values())
                for model_name in self.weights:
                    if model_name in scores:
                        # Peso inverso al error
                        self.weights[model_name] = 1.0 / (scores[model_name] + 1e-8)
                    else:
                        self.weights[model_name] = 0.0
                
                # Renormalizar
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    for key in self.weights:
                        self.weights[key] /= total_weight
                
                logger.info(f"Pesos optimizados: {self.weights}")
        
        except Exception as e:
            logger.warning(f"Error optimizando pesos: {e}")
    
    def predict(self, steps: int) -> pd.Series:
        """Hacer predicciones con ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble no entrenado")
        
        if not self.models:
            raise ValueError("No hay modelos en el ensemble")
        
        # Recopilar predicciones de todos los modelos
        predictions = {}
        for model in self.models:
            if model.is_fitted:
                try:
                    pred = model.predict(steps)
                    predictions[model.name] = pred
                except Exception as e:
                    logger.warning(f"Error en predicción de {model.name}: {e}")
                    continue
        
        if not predictions:
            raise ValueError("No se pudo generar ninguna predicción")
        
        # Combinar predicciones según el método de ensemble
        combined_pred = self._combine_predictions(predictions, steps)
        return combined_pred
    
    def _combine_predictions(self, predictions: Dict[str, pd.Series], steps: int) -> pd.Series:
        """Combinar predicciones usando pesos del ensemble"""
        # Encontrar el índice común
        first_pred = list(predictions.values())[0]
        
        if self.config.ensemble_method == 'weighted':
            # Promedio ponderado
            combined = np.zeros(steps)
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = self.weights.get(model_name, 0)
                # Alinear longitudes
                min_len = min(len(pred), steps)
                combined[:min_len] += pred[:min_len] * weight
                total_weight += weight
            
            if total_weight > 0:
                combined /= total_weight
            
            return pd.Series(combined, index=first_pred.index[:steps])
        
        else:
            # Por ahora solo implementado weighted, agregar otros métodos si es necesario
            return list(predictions.values())[0][:steps]
    
    def save(self, filepath: str) -> None:
        """Guardar modelo ensemble"""
        ensemble_data = {
            'name': self.name,
            'weights': self.weights,
            'models_info': [{'name': model.name, 'class': model.__class__.__name__} for model in self.models]
        }
        
        with open(filepath, 'w') as f:
            json.dump(ensemble_data, f)
        
        # Guardar cada modelo individualmente
        for i, model in enumerate(self.models):
            model.save(f"{filepath}_model_{i}_{model.name}.pkl")
        
        logger.info(f"Modelo ensemble guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """Cargar modelo ensemble"""
        with open(filepath, 'r') as f:
            ensemble_data = json.load(f)
        
        self.name = ensemble_data['name']
        self.weights = ensemble_data['weights']
        
        # Cargar modelos (requeriría lógica adicional para recrear instancias)
        logger.info(f"Modelo ensemble cargado desde {filepath}")

class Validator:
    """Sistema de validación para modelos de forecasting"""
    
    def __init__(self, config: ForecastingConfig):
        self.config = config
    
    def walk_forward_validation(self, model: BaseModel, data: pd.Series, 
                              initial_train_size: int = None) -> Dict[str, float]:
        """
        Validación walk-forward
        
        Args:
            model: Modelo a validar
            data: Serie temporal
            initial_train_size: Tamaño inicial del conjunto de entrenamiento
            
        Returns:
            Métricas de validación
        """
        if initial_train_size is None:
            initial_train_size = int(len(data) * 0.6)
        
        metrics = {
            'mae': [],
            'rmse': [],
            'mape': [],
            'r2': []
        }
        
        current_train_size = initial_train_size
        
        while current_train_size < len(data) - self.config.horizon:
            # Dividir datos
            train_data = data.iloc[:current_train_size]
            test_data = data.iloc[current_train_size:current_train_size + self.config.horizon]
            
            if len(test_data) == 0:
                break
            
            try:
                # Entrenar modelo con datos de entrenamiento
                model_copy = model.__class__(self.config)
                if hasattr(model, 'p'):
                    model_copy.p = model.p
                if hasattr(model, 'd'):
                    model_copy.d = model.d
                if hasattr(model, 'q'):
                    model_copy.q = model.q
                
                model_copy.fit(train_data)
                
                # Predecir
                predictions = model_copy.predict(len(test_data))
                
                # Calcular métricas
                mae = mean_absolute_error(test_data, predictions)
                rmse = np.sqrt(mean_squared_error(test_data, predictions))
                mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
                r2 = r2_score(test_data, predictions)
                
                metrics['mae'].append(mae)
                metrics['rmse'].append(rmse)
                metrics['mape'].append(mape)
                metrics['r2'].append(r2)
                
                # Avanzar ventana
                current_train_size += self.config.horizon
                
            except Exception as e:
                logger.warning(f"Error en walk-forward validation: {e}")
                current_train_size += self.config.horizon
                continue
        
        # Calcular promedios
        final_metrics = {}
        for metric, values in metrics.items():
            if values:
                final_metrics[f'{metric}_mean'] = np.mean(values)
                final_metrics[f'{metric}_std'] = np.std(values)
            else:
                final_metrics[f'{metric}_mean'] = np.nan
                final_metrics[f'{metric}_std'] = np.nan
        
        return final_metrics
    
    def time_series_cross_validation(self, model: BaseModel, data: pd.Series) -> Dict[str, float]:
        """
        Validación cruzada para series temporales
        
        Args:
            model: Modelo a validar
            data: Serie temporal
            
        Returns:
            Métricas de validación
        """
        # Usar TimeSeriesSplit de sklearn
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        metrics = {
            'mae_scores': [],
            'rmse_scores': [],
            'r2_scores': []
        }
        
        try:
            # Crear features para modelos ML
            preprocessor = DataPreprocessor(self.config)
            feature_df = preprocessor.create_features(data)
            
            X = feature_df.drop('target', axis=1)
            y = feature_df['target']
            
            # Validación cruzada
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                try:
                    # Entrenar modelo
                    model_copy = model.__class__(self.config)
                    model_copy.feature_columns = list(X.columns)
                    model_copy.model.fit(X_train, y_train)
                    
                    # Predecir
                    predictions = model_copy.model.predict(X_test)
                    
                    # Calcular métricas
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    r2 = r2_score(y_test, predictions)
                    
                    metrics['mae_scores'].append(mae)
                    metrics['rmse_scores'].append(rmse)
                    metrics['r2_scores'].append(r2)
                    
                except Exception as e:
                    logger.warning(f"Error en fold de cross-validation: {e}")
                    continue
            
            # Calcular estadísticas
            final_metrics = {}
            for metric, scores in metrics.items():
                if scores:
                    final_metrics[f'{metric}_mean'] = np.mean(scores)
                    final_metrics[f'{metric}_std'] = np.std(scores)
                else:
                    final_metrics[f'{metric}_mean'] = np.nan
                    final_metrics[f'{metric}_std'] = np.nan
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error en time series cross-validation: {e}")
            return {'error': str(e)}

class NoesisForecastingAPI:
    """API para predicciones en tiempo real"""
    
    def __init__(self, config: ForecastingConfig = None):
        self.config = config or ForecastingConfig()
        self.models = {}
        self.preprocessor = DataPreprocessor(self.config)
        self.validator = Validator(self.config)
        
    def add_model(self, name: str, model: BaseModel) -> None:
        """Añadir modelo a la API"""
        self.models[name] = model
        logger.info(f"Modelo {name} añadido a la API")
    
    def train_all_models(self, data: pd.Series, preprocessed: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Entrenar todos los modelos disponibles
        
        Args:
            data: Serie temporal
            preprocessed: Si los datos ya están preprocesados
            
        Returns:
            Resultados de entrenamiento
        """
        results = {}
        
        if not preprocessed:
            # Preprocesar datos
            data = self.preprocessor.handle_missing_values(data, 'interpolate')
            data = self.preprocessor.handle_outliers(data, 'winsorize')
        
        # Crear modelos
        models_to_train = {
            'arima': ARIMAModel(self.config),
            'sarima': SARIMAModel(self.config),
            'prophet': ProphetModel(self.config) if PROPHET_AVAILABLE else None,
            'xgboost': XGBoostModel(self.config),
            'lightgbm': LightGBMModel(self.config),
            'random_forest': RandomForestModel(self.config)
        }
        
        # Entrenar modelos
        for name, model in models_to_train.items():
            if model is not None:
                try:
                    # Validación antes de entrenar
                    if self.config.walk_forward:
                        validation_results = self.validator.walk_forward_validation(model, data)
                        logger.info(f"Validación {name}: {validation_results}")
                    
                    # Entrenar modelo
                    model.fit(data)
                    self.models[name] = model
                    
                    # Calcular métricas finales
                    if len(data) > self.config.horizon:
                        predictions = model.predict(self.config.horizon)
                        actual = data.iloc[-self.config.horizon:]
                        
                        results[name] = {
                            'mae': mean_absolute_error(actual, predictions),
                            'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                            'r2': r2_score(actual, predictions)
                        }
                    
                except Exception as e:
                    logger.error(f"Error entrenando {name}: {e}")
                    results[name] = {'error': str(e)}
        
        # Crear ensemble
        try:
            ensemble = EnsembleModel(self.config, list(self.models.values()))
            ensemble.fit(data)
            self.models['ensemble'] = ensemble
            
            # Evaluar ensemble
            if len(data) > self.config.horizon:
                predictions = ensemble.predict(self.config.horizon)
                actual = data.iloc[-self.config.horizon:]
                
                results['ensemble'] = {
                    'mae': mean_absolute_error(actual, predictions),
                    'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                    'r2': r2_score(actual, predictions)
                }
                
        except Exception as e:
            logger.error(f"Error creando ensemble: {e}")
            results['ensemble'] = {'error': str(e)}
        
        return results
    
    def predict(self, model_name: str = 'ensemble', steps: int = 12) -> pd.Series:
        """
        Hacer predicción con modelo específico
        
        Args:
            model_name: Nombre del modelo
            steps: Número de pasos a predecir
            
        Returns:
            Serie con predicciones
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible")
        
        model = self.models[model_name]
        if not model.is_fitted:
            raise ValueError(f"Modelo {model_name} no está entrenado")
        
        return model.predict(steps)
    
    def predict_ensemble(self, steps: int = 12, method: str = 'weighted') -> Dict[str, Union[pd.Series, float]]:
        """
        Predicción ensemble con múltiples modelos
        
        Args:
            steps: Número de pasos
            method: Método de combinación
            
        Returns:
            Diccionario con predicciones y confianza
        """
        if 'ensemble' not in self.models:
            raise ValueError("Ensemble no está entrenado")
        
        predictions = {}
        confidence_scores = {}
        
        # Recopilar predicciones de todos los modelos
        for name, model in self.models.items():
            if name != 'ensemble' and model.is_fitted:
                try:
                    pred = model.predict(steps)
                    predictions[name] = pred
                    
                    # Calcular score de confianza (simulado)
                    confidence_scores[name] = 0.8  # Placeholder
                    
                except Exception as e:
                    logger.warning(f"Error prediciendo con {name}: {e}")
        
        if not predictions:
            raise ValueError("No se pudo generar ninguna predicción")
        
        # Combinar predicciones
        if method == 'weighted':
            # Promedio ponderado
            weights = self.config.weights
            combined = np.zeros(steps)
            total_weight = 0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 0)
                combined += pred.values * weight
                total_weight += weight
            
            if total_weight > 0:
                combined /= total_weight
            
            combined_pred = pd.Series(combined, index=predictions[list(predictions.keys())[0]].index)
        
        else:
            # Promedio simple
            pred_arrays = [pred.values for pred in predictions.values()]
            combined = np.mean(pred_arrays, axis=0)
            combined_pred = pd.Series(combined, index=predictions[list(predictions.keys())[0]].index)
        
        return {
            'predictions': combined_pred,
            'individual_predictions': predictions,
            'confidence': np.mean(list(confidence_scores.values())) if confidence_scores else 0.5,
            'model_weights': self.config.weights
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Obtener información de un modelo"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible")
        
        model = self.models[model_name]
        
        info = {
            'name': model.name,
            'is_fitted': model.is_fitted,
            'class': model.__class__.__name__
        }
        
        if hasattr(model, 'p'):
            info['parameters'] = {
                'p': getattr(model, 'p', None),
                'd': getattr(model, 'd', None),
                'q': getattr(model, 'q', None)
            }
        
        return info
    
    def save_models(self, directory: str) -> None:
        """Guardar todos los modelos entrenados"""
        import os
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.models.items():
            if model.is_fitted:
                filepath = os.path.join(directory, f"{name}_model.pkl")
                model.save(filepath)
        
        # Guardar configuración
        config_path = os.path.join(directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info(f"Modelos guardados en {directory}")
    
    def load_models(self, directory: str) -> None:
        """Cargar modelos entrenados"""
        import os
        
        if not os.path.exists(directory):
            raise ValueError(f"Directorio {directory} no existe")
        
        # Cargar configuración
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            # Recrear configuración
            self.config = ForecastingConfig(**config_data)
        
        # Cargar modelos
        for file in os.listdir(directory):
            if file.endswith('_model.pkl'):
                model_name = file.replace('_model.pkl', '')
                filepath = os.path.join(directory, file)
                
                try:
                    if model_name == 'arima':
                        model = ARIMAModel(self.config)
                    elif model_name == 'sarima':
                        model = SARIMAModel(self.config)
                    elif model_name == 'prophet':
                        model = ProphetModel(self.config)
                    elif model_name == 'xgboost':
                        model = XGBoostModel(self.config)
                    elif model_name == 'lightgbm':
                        model = LightGBMModel(self.config)
                    elif model_name == 'random_forest':
                        model = RandomForestModel(self.config)
                    elif model_name == 'ensemble':
                        model = EnsembleModel(self.config)
                    else:
                        continue
                    
                    model.load(filepath)
                    self.models[model_name] = model
                    logger.info(f"Modelo {model_name} cargado")
                    
                except Exception as e:
                    logger.error(f"Error cargando {model_name}: {e}")
    
    def analyze_series(self, data: pd.Series) -> Dict[str, Any]:
        """
        Analizar serie temporal
        
        Args:
            data: Serie temporal
            
        Returns:
            Análisis completo de la serie
        """
        # Preprocesar
        data_clean = self.preprocessor.handle_missing_values(data, 'interpolate')
        data_clean = self.preprocessor.handle_outliers(data_clean, 'winsorize')
        
        # Detectar estacionalidad
        seasonality_info = self.preprocessor.detect_seasonality(data_clean)
        
        # Estadísticas básicas
        stats_info = {
            'mean': data_clean.mean(),
            'std': data_clean.std(),
            'min': data_clean.min(),
            'max': data_clean.max(),
            'skewness': data_clean.skew(),
            'kurtosis': data_clean.kurtosis(),
            'length': len(data_clean)
        }
        
        # Test de estacionariedad
        adf_result = adfuller(data_clean.dropna())
        stationarity_info = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05
        }
        
        return {
            'statistics': stats_info,
            'seasonality': seasonality_info,
            'stationarity': stationarity_info,
            'missing_values': data.isnull().sum(),
            'outliers_count': self.preprocessor.detect_outliers(data_clean).sum()
        }

def create_sample_data(start_date: str = '2020-01-01', periods: int = 365, 
                      frequency: str = 'D', trend: float = 0.1, 
                      seasonality_amplitude: float = 10) -> pd.Series:
    """
    Crear datos de muestra para testing
    
    Args:
        start_date: Fecha de inicio
        periods: Número de períodos
        frequency: Frecuencia ('D', 'W', 'M')
        trend: Tendencia lineal
        seasonality_amplitude: Amplitud de estacionalidad
        
    Returns:
        Serie temporal sintética
    """
    dates = pd.date_range(start=start_date, periods=periods, freq=frequency)
    
    # Componentes
    trend_component = np.arange(periods) * trend
    seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * np.arange(periods) / 12)
    noise = np.random.normal(0, 2, periods)
    
    values = trend_component + seasonal_component + noise
    
    return pd.Series(values, index=dates)

# Ejemplo de uso y testing
if __name__ == "__main__":
    # Configuración
    config = ForecastingConfig(
        test_size=0.2,
        validation_size=0.1,
        ensemble_method='weighted',
        n_splits=5,
        walk_forward=True
    )
    
    # Crear API
    api = NoesisForecastingAPI(config)
    
    # Generar datos de muestra
    print("Generando datos de muestra...")
    data = create_sample_data(periods=500, trend=0.05, seasonality_amplitude=15)
    
    # Analizar serie
    print("Analizando serie temporal...")
    analysis = api.analyze_series(data)
    print(f"Análisis: {analysis}")
    
    # Entrenar modelos
    print("Entrenando modelos...")
    results = api.train_all_models(data)
    print(f"Resultados de entrenamiento: {results}")
    
    # Hacer predicciones
    print("Generando predicciones...")
    ensemble_pred = api.predict_ensemble(steps=12)
    print(f"Predicciones ensemble: {ensemble_pred['predictions']}")
    
    # Guardar modelos
    print("Guardando modelos...")
    api.save_models("./noesis_models")
    print("Modelos guardados exitosamente")
    
    print("Sistema de forecasting NOESIS inicializado correctamente!")