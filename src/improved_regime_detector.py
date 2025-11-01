"""
Improved Regime Detection - Phase 1 & 2 Enhancements

This module implements the Phase 1 and Phase 2 improvements:
- RSI and MACD momentum indicators
- Enhanced volatility features
- Multi-objective optimization support
- Dynamic rebalancing triggers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from regime_detector import RegimeDetector
from logging_config import get_logger


class ImprovedRegimeDetector(RegimeDetector):
    """Improved regime detector with Phase 1 & 2 enhancements."""
    
    def __init__(self, method: str = "kmeans"):
        """Initialize improved regime detector."""
        super().__init__(method)
        self.logger = get_logger(__name__)
        self.logger.info("Improved regime detector initialized with Phase 1 & 2 features")
    
    def create_improved_features(self, returns: pd.DataFrame, 
                               macro_data: pd.DataFrame,
                               prices: pd.DataFrame) -> pd.DataFrame:
        """Create improved regime detection features with Phase 1 & 2 enhancements."""
        
        self.logger.info("Creating improved regime features (Phase 1 & 2)...")
        
        # Start with basic market features
        market_returns = returns.mean(axis=1)
        market_prices = prices.mean(axis=1)
        
        # Create monthly features to match macro data frequency
        monthly_features = pd.DataFrame()
        
        # 1. PHASE 1: Basic market features (monthly)
        monthly_vol = market_returns.rolling(20).std().resample('ME').last() * np.sqrt(252)
        monthly_returns = market_returns.resample('ME').last()
        
        monthly_features['market_volatility'] = monthly_vol
        monthly_features['market_return'] = monthly_returns
        
        # 2. PHASE 1: RSI and MACD (momentum indicators)
        rsi = self._calculate_rsi(market_prices, window=14)
        macd, macd_signal = self._calculate_macd(market_prices)
        
        monthly_features['rsi'] = rsi.resample('ME').last()
        monthly_features['macd'] = macd.resample('ME').last()
        monthly_features['macd_signal'] = macd_signal.resample('ME').last()
        monthly_features['macd_histogram'] = (macd - macd_signal).resample('ME').last()
        
        # 3. PHASE 1: Enhanced volatility features
        # Volatility momentum
        vol_momentum = monthly_vol.pct_change(1)
        monthly_features['volatility_momentum'] = vol_momentum
        
        # Volatility mean reversion
        vol_ma = monthly_vol.rolling(6).mean()
        monthly_features['vol_mean_reversion'] = (monthly_vol - vol_ma) / vol_ma
        
        # 4. PHASE 1: Price momentum features
        momentum_1m = market_prices.pct_change(21).resample('ME').last()
        momentum_3m = market_prices.pct_change(63).resample('ME').last()
        momentum_6m = market_prices.pct_change(126).resample('ME').last()
        
        monthly_features['momentum_1m'] = momentum_1m
        monthly_features['momentum_3m'] = momentum_3m
        monthly_features['momentum_6m'] = momentum_6m
        
        # 5. PHASE 2: Cross-asset momentum
        if len(returns.columns) > 5:  # Ensure we have enough assets
            # Equity vs Bond momentum
            equity_assets = [col for col in returns.columns if col in ['SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF', 'XLV']]
            bond_assets = [col for col in returns.columns if col in ['AGG', 'TLT', 'SHY', 'TIP', 'LQD']]
            
            if equity_assets and bond_assets:
                equity_returns = returns[equity_assets].mean(axis=1)
                bond_returns = returns[bond_assets].mean(axis=1)
                
                # Relative performance
                equity_bond_ratio = (equity_returns.rolling(20).mean() / 
                                   bond_returns.rolling(20).mean()).resample('ME').last()
                monthly_features['equity_bond_momentum'] = equity_bond_ratio.pct_change(1)
                
                # Rolling correlation
                equity_bond_corr = equity_returns.rolling(60).corr(bond_returns).resample('ME').last()
                monthly_features['equity_bond_correlation'] = equity_bond_corr
        
        # 6. Add macro features
        macro_monthly = macro_data.resample('ME').last()
        for col in macro_monthly.columns:
            if col in ['VIXCLS', 'DGS10', 'DGS2', 'UNRATE']:
                monthly_features[col] = macro_monthly[col]
        
        # 7. PHASE 1: Enhanced macro features
        if 'DGS10' in monthly_features.columns and 'DGS2' in monthly_features.columns:
            monthly_features['yield_spread'] = monthly_features['DGS10'] - monthly_features['DGS2']
            monthly_features['yield_slope_momentum'] = monthly_features['yield_spread'].pct_change(1)
        
        if 'VIXCLS' in monthly_features.columns:
            monthly_features['vix_momentum'] = monthly_features['VIXCLS'].pct_change(1)
            monthly_features['vix_mean_reversion'] = (
                monthly_features['VIXCLS'] - monthly_features['VIXCLS'].rolling(6).mean()
            ) / monthly_features['VIXCLS'].rolling(6).mean()
        
        # 8. PHASE 2: Market stress indicators
        # Maximum drawdown (rolling)
        cumulative_returns = (1 + market_returns).cumprod()
        rolling_max = cumulative_returns.rolling(252).max()
        drawdown = ((cumulative_returns - rolling_max) / rolling_max).resample('ME').last()
        monthly_features['max_drawdown'] = drawdown
        
        # Skewness and kurtosis
        rolling_skew = market_returns.rolling(60).skew().resample('ME').last()
        rolling_kurt = market_returns.rolling(60).kurt().resample('ME').last()
        monthly_features['return_skewness'] = rolling_skew
        monthly_features['return_kurtosis'] = rolling_kurt
        
        # Clean and validate features
        monthly_features = monthly_features.dropna()
        
        # Standardize features for better clustering
        feature_cols = monthly_features.select_dtypes(include=[np.number]).columns
        monthly_features[feature_cols] = (
            monthly_features[feature_cols] - monthly_features[feature_cols].mean()
        ) / monthly_features[feature_cols].std()
        
        self.logger.info(f"Created {len(monthly_features.columns)} improved features, "
                        f"{len(monthly_features)} observations")
        
        return monthly_features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return macd, macd_signal
    
    def fit_improved_regimes(self, returns: pd.DataFrame, 
                           macro_data: pd.DataFrame,
                           prices: pd.DataFrame,
                           n_regimes: int = 3) -> np.ndarray:
        """Fit regimes using improved features."""
        
        # Create improved features
        improved_features = self.create_improved_features(returns, macro_data, prices)
        
        # Fit regimes using parent class method
        regime_labels = self.fit_regimes(improved_features, n_regimes)
        
        self.logger.info(f"Improved regime detection completed with {n_regimes} regimes")
        
        return regime_labels
    
    def detect_regime_changes(self, current_features: pd.DataFrame, 
                            previous_regime: int) -> Tuple[int, float]:
        """Detect regime changes for dynamic rebalancing (Phase 2)."""
        
        if not hasattr(self, 'clusterer') or self.clusterer is None:
            return previous_regime, 0.0
        
        try:
            # Predict current regime
            current_regime = self.predict_regime(current_features.tail(1))
            
            # Calculate regime confidence (distance to cluster centers)
            if hasattr(self.clusterer, 'cluster_centers_'):
                features_array = current_features.tail(1).values
                distances = np.linalg.norm(
                    self.clusterer.cluster_centers_ - features_array, axis=1
                )
                confidence = 1.0 / (1.0 + distances[current_regime])
            else:
                confidence = 0.5
            
            return current_regime, confidence
            
        except Exception as e:
            self.logger.warning(f"Regime change detection failed: {str(e)}")
            return previous_regime, 0.0