"""
Enhanced Regime Detection with Advanced Technical and Macro Indicators

This module extends the basic regime detection with:
- Momentum indicators (RSI, MACD)
- Volatility regime indicators
- Credit and yield curve features
- Cross-asset correlations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from regime_detector import RegimeDetector
from logging_config import get_logger


class EnhancedRegimeDetector(RegimeDetector):
    """Enhanced regime detector with comprehensive market indicators."""
    
    def __init__(self, method: str = "kmeans"):
        """Initialize enhanced regime detector."""
        super().__init__(method)
        self.logger = get_logger(__name__)
        
        # Asset classification for cross-asset analysis
        self.equity_assets = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VUG', 'VTV', 'VB', 'VO']
        self.bond_assets = ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG', 'MUB']
        self.commodity_assets = ['GLD', 'SLV', 'DBC', 'USO', 'PDBC', 'IAU']
        
        self.logger.info("Enhanced regime detector initialized with advanced indicators")
    
    def create_enhanced_features(self, returns: pd.DataFrame, 
                               macro_data: pd.DataFrame,
                               prices: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive regime detection features."""
        
        self.logger.info("Creating enhanced regime detection features...")
        
        # Start with basic features
        basic_features = self._create_basic_features(returns, macro_data)
        
        # Add enhanced features
        enhanced_features = basic_features.copy()
        
        # 1. Momentum Indicators
        momentum_features = self._create_momentum_features(returns, prices)
        enhanced_features = enhanced_features.join(momentum_features, how='outer')
        
        # 2. Volatility Regime Indicators  
        volatility_features = self._create_volatility_features(returns, macro_data)
        enhanced_features = enhanced_features.join(volatility_features, how='outer')
        
        # 3. Credit and Yield Indicators
        credit_features = self._create_credit_yield_features(macro_data)
        enhanced_features = enhanced_features.join(credit_features, how='outer')
        
        # 4. Cross-Asset Features
        cross_asset_features = self._create_cross_asset_features(returns)
        enhanced_features = enhanced_features.join(cross_asset_features, how='outer')
        
        # 5. Market Stress Indicators
        stress_features = self._create_stress_indicators(returns, prices)
        enhanced_features = enhanced_features.join(stress_features, how='outer')
        
        # Clean and validate
        enhanced_features = enhanced_features.dropna()
        
        self.logger.info(f"Created {len(enhanced_features.columns)} enhanced features, "
                        f"{len(enhanced_features)} observations")
        
        return enhanced_features
    
    def _create_basic_features(self, returns: pd.DataFrame, 
                             macro_data: pd.DataFrame) -> pd.DataFrame:
        """Create basic regime features (existing functionality)."""
        
        # Market volatility (20-day rolling)
        market_returns = returns.mean(axis=1)
        market_volatility = market_returns.rolling(20).std() * np.sqrt(252)
        
        # Resample to monthly frequency to match macro data
        monthly_vol = market_volatility.resample('M').last()
        monthly_returns = market_returns.resample('M').last()
        
        # Align with macro data
        aligned_macro = macro_data.resample('M').last()
        
        # Create feature DataFrame
        features = pd.DataFrame(index=aligned_macro.index)
        features['market_volatility'] = monthly_vol
        features['market_return'] = monthly_returns
        
        # Add macro features
        for col in aligned_macro.columns:
            features[col] = aligned_macro[col]
        
        # Yield spread
        if 'DGS10' in features.columns and 'DGS2' in features.columns:
            features['yield_spread'] = features['DGS10'] - features['DGS2']
        
        return features.dropna()
    
    def _create_momentum_features(self, returns: pd.DataFrame, 
                                prices: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based regime indicators."""
        
        # Equal-weighted market index
        market_prices = prices.mean(axis=1)
        
        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(market_prices, window=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal = self._calculate_macd(market_prices)
        
        # Price momentum (various timeframes)
        momentum_1m = market_prices.pct_change(21)  # 1-month momentum
        momentum_3m = market_prices.pct_change(63)  # 3-month momentum
        momentum_6m = market_prices.pct_change(126) # 6-month momentum
        
        # Moving average ratios
        ma_50 = market_prices.rolling(50).mean()
        ma_200 = market_prices.rolling(200).mean()
        ma_ratio = ma_50 / ma_200
        price_to_ma200 = market_prices / ma_200
        
        # Combine features and resample to monthly
        momentum_data = pd.DataFrame({
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'ma_ratio_50_200': ma_ratio,
            'price_to_ma200': price_to_ma200
        }, index=market_prices.index)
        
        return momentum_data.resample('M').last().dropna()
    
    def _create_volatility_features(self, returns: pd.DataFrame,
                                  macro_data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility regime indicators."""
        
        # Realized volatility (multiple timeframes)
        market_returns = returns.mean(axis=1)
        
        realized_vol_20d = market_returns.rolling(20).std() * np.sqrt(252)
        realized_vol_60d = market_returns.rolling(60).std() * np.sqrt(252)
        
        # Volatility of volatility
        vol_of_vol = realized_vol_20d.rolling(20).std()
        
        # VIX-based features (if available)
        vix_features = pd.DataFrame()
        if 'VIXCLS' in macro_data.columns:
            vix = macro_data['VIXCLS'].resample('D').ffill()
            
            # Align with returns
            aligned_vix = vix.reindex(returns.index, method='ffill')
            
            # VIX vs realized volatility
            vol_risk_premium = aligned_vix - realized_vol_20d
            
            # VIX momentum
            vix_momentum = aligned_vix.pct_change(20)
            
            vix_features = pd.DataFrame({
                'vol_risk_premium': vol_risk_premium,
                'vix_momentum': vix_momentum,
                'vix_level': aligned_vix
            }, index=returns.index)
        
        # Combine volatility features
        vol_data = pd.DataFrame({
            'realized_vol_20d': realized_vol_20d,
            'realized_vol_60d': realized_vol_60d,
            'vol_of_vol': vol_of_vol
        }, index=returns.index)
        
        if not vix_features.empty:
            vol_data = vol_data.join(vix_features)
        
        return vol_data.resample('M').last().dropna()
    
    def _create_credit_yield_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Create credit and yield curve indicators."""
        
        features = pd.DataFrame(index=macro_data.index)
        
        # Yield curve shape indicators
        if all(col in macro_data.columns for col in ['DGS2', 'DGS5', 'DGS10']):
            # Yield curve slope (10Y - 2Y)
            features['yield_slope'] = macro_data['DGS10'] - macro_data['DGS2']
            
            # Yield curve curvature (butterfly)
            features['yield_curvature'] = (
                macro_data['DGS5'] - 
                (macro_data['DGS2'] + macro_data['DGS10']) / 2
            )
            
            # 5Y-2Y slope
            features['yield_slope_5y2y'] = macro_data['DGS5'] - macro_data['DGS2']
        
        # Interest rate momentum
        if 'DGS10' in macro_data.columns:
            features['rate_momentum_1m'] = macro_data['DGS10'].pct_change(1)
            features['rate_momentum_3m'] = macro_data['DGS10'].pct_change(3)
        
        return features.resample('M').last().dropna()
    
    def _create_cross_asset_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset correlation and momentum features."""
        
        # Get asset class returns
        equity_cols = [col for col in returns.columns if col in self.equity_assets]
        bond_cols = [col for col in returns.columns if col in self.bond_assets]
        commodity_cols = [col for col in returns.columns if col in self.commodity_assets]
        
        features = pd.DataFrame(index=returns.index)
        
        if equity_cols and bond_cols:
            equity_returns = returns[equity_cols].mean(axis=1)
            bond_returns = returns[bond_cols].mean(axis=1)
            
            # Rolling correlations
            features['equity_bond_corr_60d'] = (
                equity_returns.rolling(60).corr(bond_returns)
            )
            features['equity_bond_corr_20d'] = (
                equity_returns.rolling(20).corr(bond_returns)
            )
            
            # Relative performance
            features['equity_bond_ratio'] = (
                (1 + equity_returns).rolling(20).apply(np.prod) /
                (1 + bond_returns).rolling(20).apply(np.prod)
            )
        
        if equity_cols and commodity_cols:
            equity_returns = returns[equity_cols].mean(axis=1)
            commodity_returns = returns[commodity_cols].mean(axis=1)
            
            features['equity_commodity_corr'] = (
                equity_returns.rolling(60).corr(commodity_returns)
            )
        
        return features.resample('M').last().dropna()
    
    def _create_stress_indicators(self, returns: pd.DataFrame,
                                prices: pd.DataFrame) -> pd.DataFrame:
        """Create market stress and tail risk indicators."""
        
        market_returns = returns.mean(axis=1)
        market_prices = prices.mean(axis=1)
        
        # Maximum drawdown (rolling)
        cumulative_returns = (1 + market_returns).cumprod()
        rolling_max = cumulative_returns.rolling(252).max()
        max_drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        # Downside deviation
        downside_returns = market_returns[market_returns < 0]
        downside_vol = market_returns.rolling(60).apply(
            lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2)) * np.sqrt(252)
        )
        
        # Skewness and kurtosis (rolling)
        rolling_skew = market_returns.rolling(60).skew()
        rolling_kurt = market_returns.rolling(60).kurt()
        
        # Up/down capture ratios
        up_days = (market_returns > 0).rolling(60).sum()
        down_days = (market_returns < 0).rolling(60).sum()
        up_down_ratio = up_days / (down_days + 1e-6)  # Avoid division by zero
        
        stress_data = pd.DataFrame({
            'max_drawdown': max_drawdown,
            'downside_volatility': downside_vol,
            'rolling_skewness': rolling_skew,
            'rolling_kurtosis': rolling_kurt,
            'up_down_ratio': up_down_ratio
        }, index=returns.index)
        
        return stress_data.resample('M').last().dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
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
    
    def fit_enhanced_regimes(self, returns: pd.DataFrame, 
                           macro_data: pd.DataFrame,
                           prices: pd.DataFrame,
                           n_regimes: int = 4) -> np.ndarray:
        """Fit regimes using enhanced features."""
        
        # Create enhanced features
        enhanced_features = self.create_enhanced_features(returns, macro_data, prices)
        
        # Fit regimes using parent class method
        regime_labels = self.fit_regimes(enhanced_features, n_regimes)
        
        self.logger.info(f"Enhanced regime detection completed with {n_regimes} regimes")
        
        return regime_labels