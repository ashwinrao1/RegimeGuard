"""
Advanced Regime Detection - Phase 3 Implementation

This module implements the most sophisticated regime detection using:
- Hidden Markov Models (HMM) for temporal dependencies
- Machine Learning ensemble methods
- Advanced feature engineering with factor models
- Regime prediction with confidence intervals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. Install with: pip install hmmlearn")

from improved_regime_detector import ImprovedRegimeDetector
from logging_config import get_logger


class AdvancedRegimeDetector(ImprovedRegimeDetector):
    """Advanced regime detector with HMM and ML ensemble methods."""
    
    def __init__(self, method: str = "ensemble"):
        """Initialize advanced regime detector."""
        super().__init__(method)
        self.logger = get_logger(__name__)
        
        # Advanced models
        self.hmm_model = None
        self.ml_ensemble = None
        self.factor_model = None
        self.scaler = StandardScaler()
        
        # Model selection
        self.available_methods = ["ensemble", "hmm", "ml_only", "kmeans"]
        if method not in self.available_methods:
            self.logger.warning(f"Method {method} not available, using ensemble")
            method = "ensemble"
        
        self.method = method
        self.logger.info(f"Advanced regime detector initialized with method: {method}")
    
    def create_advanced_features(self, returns: pd.DataFrame, 
                               macro_data: pd.DataFrame,
                               prices: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features with factor models and ML engineering."""
        
        self.logger.info("Creating advanced features (Phase 3)...")
        
        # Start with Phase 1 & 2 features
        base_features = self.create_improved_features(returns, macro_data, prices)
        
        # Add Phase 3 advanced features
        advanced_features = base_features.copy()
        
        # 1. Factor Model Features
        factor_features = self._create_factor_features(returns, prices)
        if not factor_features.empty:
            advanced_features = advanced_features.join(factor_features, how='outer')
        
        # 2. Principal Component Analysis
        pca_features = self._create_pca_features(returns)
        if not pca_features.empty:
            advanced_features = advanced_features.join(pca_features, how='outer')
        
        # 3. Advanced Technical Indicators
        technical_features = self._create_advanced_technical_features(returns, prices)
        if not technical_features.empty:
            advanced_features = advanced_features.join(technical_features, how='outer')
        
        # 4. Regime Persistence Features
        persistence_features = self._create_regime_persistence_features(returns)
        if not persistence_features.empty:
            advanced_features = advanced_features.join(persistence_features, how='outer')
        
        # Clean and prepare for ML
        advanced_features = advanced_features.dropna()
        
        # Feature selection and engineering
        if len(advanced_features) > 20:  # Need sufficient data for ML
            advanced_features = self._engineer_ml_features(advanced_features)
        
        self.logger.info(f"Created {len(advanced_features.columns)} advanced features, "
                        f"{len(advanced_features)} observations")
        
        return advanced_features
    
    def _create_factor_features(self, returns: pd.DataFrame, 
                              prices: pd.DataFrame) -> pd.DataFrame:
        """Create factor model features (Fama-French style)."""
        
        try:
            # Market factor (already have market returns)
            market_returns = returns.mean(axis=1)
            
            # Size factor (Small minus Big)
            # Approximate using small cap vs large cap ETFs
            small_cap_assets = [col for col in returns.columns if col in ['IWM', 'VB']]
            large_cap_assets = [col for col in returns.columns if col in ['SPY', 'VOO', 'VTI']]
            
            smb_factor = pd.Series(index=returns.index, dtype=float)
            if small_cap_assets and large_cap_assets:
                small_returns = returns[small_cap_assets].mean(axis=1)
                large_returns = returns[large_cap_assets].mean(axis=1)
                smb_factor = small_returns - large_returns
            
            # Value factor (High minus Low)
            # Approximate using value vs growth ETFs
            value_assets = [col for col in returns.columns if col in ['VTV']]
            growth_assets = [col for col in returns.columns if col in ['VUG', 'QQQ']]
            
            hml_factor = pd.Series(index=returns.index, dtype=float)
            if value_assets and growth_assets:
                value_returns = returns[value_assets].mean(axis=1)
                growth_returns = returns[growth_assets].mean(axis=1)
                hml_factor = value_returns - growth_returns
            
            # Momentum factor
            momentum_factor = market_returns.rolling(21).mean() - market_returns.rolling(252).mean()
            
            # Quality factor (approximate using low volatility)
            quality_factor = -returns.rolling(60).std().mean(axis=1)  # Negative volatility as quality proxy
            
            # Create monthly factor features
            factor_data = pd.DataFrame({
                'market_factor': market_returns,
                'size_factor': smb_factor,
                'value_factor': hml_factor,
                'momentum_factor': momentum_factor,
                'quality_factor': quality_factor
            }, index=returns.index)
            
            # Resample to monthly and add rolling statistics
            monthly_factors = factor_data.resample('ME').last()
            
            # Add factor momentum and volatility
            for factor in ['market_factor', 'size_factor', 'value_factor', 'momentum_factor']:
                if factor in monthly_factors.columns:
                    monthly_factors[f'{factor}_momentum'] = monthly_factors[factor].pct_change(1)
                    monthly_factors[f'{factor}_volatility'] = monthly_factors[factor].rolling(6).std()
            
            return monthly_factors.dropna()
            
        except Exception as e:
            self.logger.warning(f"Factor feature creation failed: {str(e)}")
            return pd.DataFrame()
    
    def _create_pca_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create principal component features."""
        
        try:
            if len(returns.columns) < 5:
                return pd.DataFrame()
            
            # Monthly returns for PCA
            monthly_returns = returns.resample('ME').last()
            
            # Rolling PCA (6-month windows)
            pca_features = []
            
            for i in range(6, len(monthly_returns)):
                window_data = monthly_returns.iloc[i-6:i].dropna()
                
                if len(window_data) >= 5 and len(window_data.columns) >= 3:
                    # Fit PCA
                    pca = PCA(n_components=min(3, len(window_data.columns)))
                    pca_result = pca.fit_transform(window_data.fillna(0))
                    
                    # Store PCA features
                    date = monthly_returns.index[i]
                    pca_features.append({
                        'date': date,
                        'pc1_explained_var': pca.explained_variance_ratio_[0],
                        'pc2_explained_var': pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0,
                        'pc1_loading_concentration': np.std(pca.components_[0]),
                        'total_explained_var': np.sum(pca.explained_variance_ratio_[:2])
                    })
            
            if pca_features:
                pca_df = pd.DataFrame(pca_features)
                pca_df.set_index('date', inplace=True)
                return pca_df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.warning(f"PCA feature creation failed: {str(e)}")
            return pd.DataFrame()
    
    def _create_advanced_technical_features(self, returns: pd.DataFrame,
                                          prices: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical analysis features."""
        
        try:
            market_prices = prices.mean(axis=1)
            market_returns = returns.mean(axis=1)
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = market_prices.rolling(bb_period).mean()
            bb_std_dev = market_prices.rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            bb_position = (market_prices - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic Oscillator
            high_14 = market_prices.rolling(14).max()
            low_14 = market_prices.rolling(14).min()
            stoch_k = 100 * (market_prices - low_14) / (high_14 - low_14)
            stoch_d = stoch_k.rolling(3).mean()
            
            # Williams %R
            williams_r = -100 * (high_14 - market_prices) / (high_14 - low_14)
            
            # Commodity Channel Index (CCI)
            typical_price = market_prices  # Simplified for single price series
            cci_period = 20
            cci_ma = typical_price.rolling(cci_period).mean()
            cci_mad = typical_price.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - cci_ma) / (0.015 * cci_mad)
            
            # Average True Range (ATR) - simplified
            price_changes = market_prices.pct_change().abs()
            atr = price_changes.rolling(14).mean()
            
            # Aroon Oscillator
            aroon_period = 14
            aroon_up = market_prices.rolling(aroon_period).apply(
                lambda x: (aroon_period - (aroon_period - 1 - x.argmax())) / aroon_period * 100
            )
            aroon_down = market_prices.rolling(aroon_period).apply(
                lambda x: (aroon_period - (aroon_period - 1 - x.argmin())) / aroon_period * 100
            )
            aroon_oscillator = aroon_up - aroon_down
            
            # Create monthly technical features
            technical_data = pd.DataFrame({
                'bollinger_position': bb_position,
                'stochastic_k': stoch_k,
                'stochastic_d': stoch_d,
                'williams_r': williams_r,
                'cci': cci,
                'atr': atr,
                'aroon_oscillator': aroon_oscillator
            }, index=market_prices.index)
            
            return technical_data.resample('ME').last().dropna()
            
        except Exception as e:
            self.logger.warning(f"Advanced technical feature creation failed: {str(e)}")
            return pd.DataFrame()
    
    def _create_regime_persistence_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture regime persistence and transitions."""
        
        try:
            market_returns = returns.mean(axis=1)
            
            # Volatility clustering (GARCH-like features)
            vol_20 = market_returns.rolling(20).std()
            vol_60 = market_returns.rolling(60).std()
            vol_clustering = vol_20 / vol_60
            
            # Return autocorrelation
            return_autocorr_1 = market_returns.rolling(60).apply(
                lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
            )
            return_autocorr_5 = market_returns.rolling(60).apply(
                lambda x: x.autocorr(lag=5) if len(x.dropna()) > 15 else 0
            )
            
            # Volatility autocorrelation
            vol_autocorr = vol_20.rolling(60).apply(
                lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
            )
            
            # Regime stability proxy (rolling correlation with trend)
            trend = market_returns.rolling(60).mean()
            regime_stability = market_returns.rolling(60).corr(trend)
            
            # Market microstructure proxies
            # Up/down day clustering
            up_days = (market_returns > 0).astype(int)
            up_day_clustering = up_days.rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
            )
            
            persistence_data = pd.DataFrame({
                'volatility_clustering': vol_clustering,
                'return_autocorr_1': return_autocorr_1,
                'return_autocorr_5': return_autocorr_5,
                'volatility_autocorr': vol_autocorr,
                'regime_stability': regime_stability,
                'up_day_clustering': up_day_clustering
            }, index=returns.index)
            
            return persistence_data.resample('ME').last().dropna()
            
        except Exception as e:
            self.logger.warning(f"Regime persistence feature creation failed: {str(e)}")
            return pd.DataFrame()
    
    def _engineer_ml_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically for machine learning."""
        
        try:
            ml_features = features.copy()
            
            # Add lagged features (important for regime prediction)
            for lag in [1, 2, 3]:
                for col in ['market_volatility', 'market_return', 'rsi', 'macd']:
                    if col in ml_features.columns:
                        ml_features[f'{col}_lag_{lag}'] = ml_features[col].shift(lag)
            
            # Add rolling statistics
            for window in [3, 6]:
                for col in ['market_volatility', 'rsi']:
                    if col in ml_features.columns:
                        ml_features[f'{col}_ma_{window}'] = ml_features[col].rolling(window).mean()
                        ml_features[f'{col}_std_{window}'] = ml_features[col].rolling(window).std()
            
            # Add interaction features
            if 'market_volatility' in ml_features.columns and 'rsi' in ml_features.columns:
                ml_features['vol_rsi_interaction'] = ml_features['market_volatility'] * ml_features['rsi']
            
            if 'macd' in ml_features.columns and 'market_return' in ml_features.columns:
                ml_features['macd_return_interaction'] = ml_features['macd'] * ml_features['market_return']
            
            return ml_features.dropna()
            
        except Exception as e:
            self.logger.warning(f"ML feature engineering failed: {str(e)}")
            return features
    
    def fit_advanced_regimes(self, returns: pd.DataFrame, 
                           macro_data: pd.DataFrame,
                           prices: pd.DataFrame,
                           n_regimes: int = 3) -> np.ndarray:
        """Fit regimes using advanced methods (HMM, ML ensemble)."""
        
        # Create advanced features
        advanced_features = self.create_advanced_features(returns, macro_data, prices)
        
        if len(advanced_features) < 10:
            self.logger.warning("Insufficient data for advanced methods, falling back to basic")
            return self.fit_regimes(advanced_features, n_regimes)
        
        # Choose method based on initialization
        if self.method == "ensemble":
            return self._fit_ensemble_regimes(advanced_features, n_regimes)
        elif self.method == "hmm" and HMM_AVAILABLE:
            return self._fit_hmm_regimes(advanced_features, n_regimes)
        elif self.method == "ml_only":
            return self._fit_ml_regimes(advanced_features, n_regimes)
        else:
            # Fallback to improved K-means
            return self.fit_regimes(advanced_features, n_regimes)
    
    def _fit_ensemble_regimes(self, features: pd.DataFrame, n_regimes: int) -> np.ndarray:
        """Fit regimes using ensemble of methods."""
        
        self.logger.info("Fitting regimes using ensemble method...")
        
        # Method 1: K-means (baseline)
        kmeans_labels = self.fit_regimes(features, n_regimes)
        
        # Method 2: HMM (if available)
        if HMM_AVAILABLE:
            hmm_labels = self._fit_hmm_regimes(features, n_regimes)
        else:
            hmm_labels = kmeans_labels
        
        # Method 3: ML clustering
        ml_labels = self._fit_ml_regimes(features, n_regimes)
        
        # Ensemble voting
        ensemble_labels = self._ensemble_vote([kmeans_labels, hmm_labels, ml_labels])
        
        self.logger.info("Ensemble regime detection completed")
        return ensemble_labels
    
    def _fit_hmm_regimes(self, features: pd.DataFrame, n_regimes: int) -> np.ndarray:
        """Fit regimes using Hidden Markov Model."""
        
        if not HMM_AVAILABLE:
            self.logger.warning("HMM not available, using K-means")
            return self.fit_regimes(features, n_regimes)
        
        try:
            self.logger.info("Fitting regimes using Hidden Markov Model...")
            
            # Prepare data
            X = self.scaler.fit_transform(features.fillna(features.mean()))
            
            # Fit HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            self.hmm_model.fit(X)
            regime_labels = self.hmm_model.predict(X)
            
            self.logger.info(f"HMM regime detection completed with {n_regimes} regimes")
            return regime_labels
            
        except Exception as e:
            self.logger.warning(f"HMM fitting failed: {str(e)}, using K-means")
            return self.fit_regimes(features, n_regimes)
    
    def _fit_ml_regimes(self, features: pd.DataFrame, n_regimes: int) -> np.ndarray:
        """Fit regimes using machine learning clustering."""
        
        try:
            self.logger.info("Fitting regimes using ML ensemble...")
            
            # Use K-means as initial labels for supervised learning
            initial_labels = self.fit_regimes(features, n_regimes)
            
            # Prepare features
            X = features.fillna(features.mean()).values
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ensemble of classifiers
            rf_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Fit models
            rf_model.fit(X_scaled, initial_labels)
            gb_model.fit(X_scaled, initial_labels)
            
            # Predict with ensemble
            rf_pred = rf_model.predict(X_scaled)
            gb_pred = gb_model.predict(X_scaled)
            
            # Ensemble prediction
            ml_labels = self._ensemble_vote([rf_pred, gb_pred])
            
            # Store models for future prediction
            self.ml_ensemble = {
                'rf_model': rf_model,
                'gb_model': gb_model,
                'scaler': self.scaler
            }
            
            self.logger.info("ML regime detection completed")
            return ml_labels
            
        except Exception as e:
            self.logger.warning(f"ML regime fitting failed: {str(e)}, using K-means")
            return self.fit_regimes(features, n_regimes)
    
    def _ensemble_vote(self, label_arrays: List[np.ndarray]) -> np.ndarray:
        """Combine multiple regime predictions using voting."""
        
        if len(label_arrays) == 1:
            return label_arrays[0]
        
        # Simple majority voting
        n_samples = len(label_arrays[0])
        ensemble_labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = [labels[i] for labels in label_arrays]
            # Most common vote
            ensemble_labels[i] = max(set(votes), key=votes.count)
        
        return ensemble_labels
    
    def predict_regime_with_confidence(self, features: pd.DataFrame) -> Tuple[int, float, Dict]:
        """Predict regime with confidence score and probabilities."""
        
        try:
            if features.empty:
                return 0, 0.0, {}
            
            # Prepare features
            X = features.tail(1).fillna(features.mean()).values
            
            predictions = []
            confidences = []
            
            # K-means prediction
            if hasattr(self, 'clusterer') and self.clusterer is not None:
                kmeans_pred = self.predict_regime(features.tail(1))
                predictions.append(kmeans_pred)
                confidences.append(0.7)  # Default confidence
            
            # HMM prediction
            if self.hmm_model is not None:
                X_scaled = self.scaler.transform(X)
                hmm_pred = self.hmm_model.predict(X_scaled)[0]
                hmm_proba = np.max(self.hmm_model.predict_proba(X_scaled)[0])
                predictions.append(hmm_pred)
                confidences.append(hmm_proba)
            
            # ML ensemble prediction
            if self.ml_ensemble is not None:
                X_scaled = self.ml_ensemble['scaler'].transform(X)
                
                rf_pred = self.ml_ensemble['rf_model'].predict(X_scaled)[0]
                rf_proba = np.max(self.ml_ensemble['rf_model'].predict_proba(X_scaled)[0])
                
                gb_pred = self.ml_ensemble['gb_model'].predict(X_scaled)[0]
                gb_proba = np.max(self.ml_ensemble['gb_model'].predict_proba(X_scaled)[0])
                
                predictions.extend([rf_pred, gb_pred])
                confidences.extend([rf_proba, gb_proba])
            
            if not predictions:
                return 0, 0.0, {}
            
            # Ensemble prediction
            final_prediction = max(set(predictions), key=predictions.count)
            avg_confidence = np.mean(confidences)
            
            # Create detailed results
            results = {
                'regime': final_prediction,
                'confidence': avg_confidence,
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'consensus_strength': predictions.count(final_prediction) / len(predictions)
            }
            
            return final_prediction, avg_confidence, results
            
        except Exception as e:
            self.logger.warning(f"Advanced regime prediction failed: {str(e)}")
            return 0, 0.0, {}