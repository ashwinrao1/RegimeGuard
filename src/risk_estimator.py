"""
Risk estimation module for the robust portfolio optimization system.

This module implements regime-specific risk parameter estimation including
covariance matrices, expected returns, and shrinkage estimation techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import linalg
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
import warnings

from interfaces import RiskEstimatorInterface, RegimeParameters
from config import get_config
from logging_config import get_logger


class CovarianceEstimator:
    """Implements various covariance estimation methods for regime-specific risk modeling."""
    
    def __init__(self, method: str = "sample", shrinkage_target: str = "identity"):
        """Initialize covariance estimator.
        
        Args:
            method: Estimation method ("sample", "ledoit_wolf", "oas", "shrunk")
            shrinkage_target: Target for shrinkage ("identity", "diagonal", "market")
        """
        self.method = method.lower()
        self.shrinkage_target = shrinkage_target.lower()
        self.logger = get_logger(__name__)
        
        # Validation parameters
        self.min_observations = 30  # Minimum observations per regime
        self.regularization_factor = 1e-6  # For numerical stability
        
    def estimate_sample_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate sample covariance matrix.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Sample covariance matrix
        """
        if len(returns) < 2:
            raise ValueError("Need at least 2 observations for covariance estimation")
        
        # Calculate sample covariance
        cov_matrix = returns.cov().values
        
        # Add regularization for numerical stability
        cov_matrix += self.regularization_factor * np.eye(cov_matrix.shape[0])
        
        return cov_matrix
    
    def estimate_ledoit_wolf_covariance(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Estimate covariance using Ledoit-Wolf shrinkage.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Tuple of (shrunk covariance matrix, shrinkage intensity)
        """
        if len(returns) < self.min_observations:
            self.logger.warning(f"Few observations ({len(returns)}) for Ledoit-Wolf estimation")
        
        # Fit Ledoit-Wolf estimator
        lw = LedoitWolf()
        lw.fit(returns.values)
        
        shrunk_cov = lw.covariance_
        shrinkage_intensity = lw.shrinkage_
        
        self.logger.debug(f"Ledoit-Wolf shrinkage intensity: {shrinkage_intensity:.4f}")
        
        return shrunk_cov, shrinkage_intensity
    
    def estimate_oas_covariance(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Estimate covariance using Oracle Approximating Shrinkage (OAS).
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Tuple of (shrunk covariance matrix, shrinkage intensity)
        """
        if len(returns) < self.min_observations:
            self.logger.warning(f"Few observations ({len(returns)}) for OAS estimation")
        
        # Fit OAS estimator
        oas = OAS()
        oas.fit(returns.values)
        
        shrunk_cov = oas.covariance_
        shrinkage_intensity = oas.shrinkage_
        
        self.logger.debug(f"OAS shrinkage intensity: {shrinkage_intensity:.4f}")
        
        return shrunk_cov, shrinkage_intensity
    
    def apply_custom_shrinkage(self, sample_cov: np.ndarray, shrinkage_intensity: float,
                              target_type: str = "identity") -> np.ndarray:
        """Apply custom shrinkage to sample covariance matrix.
        
        Args:
            sample_cov: Sample covariance matrix
            shrinkage_intensity: Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage)
            target_type: Type of shrinkage target
            
        Returns:
            Shrunk covariance matrix
        """
        n_assets = sample_cov.shape[0]
        
        # Define shrinkage targets
        if target_type == "identity":
            # Shrink towards identity matrix scaled by average variance
            avg_variance = np.trace(sample_cov) / n_assets
            target = avg_variance * np.eye(n_assets)
        
        elif target_type == "diagonal":
            # Shrink towards diagonal matrix (remove correlations)
            target = np.diag(np.diag(sample_cov))
        
        elif target_type == "market":
            # Shrink towards single-factor (market) model
            market_var = np.mean(np.diag(sample_cov))
            market_corr = 0.3  # Assumed market correlation
            target = market_var * (market_corr * np.ones((n_assets, n_assets)) + 
                                 (1 - market_corr) * np.eye(n_assets))
        
        elif target_type == "constant_correlation":
            # Shrink towards constant correlation matrix
            avg_variance = np.mean(np.diag(sample_cov))
            avg_correlation = 0.2  # Assumed constant correlation
            
            target = np.full((n_assets, n_assets), avg_correlation * avg_variance)
            np.fill_diagonal(target, avg_variance)
        
        else:
            raise ValueError(f"Unknown shrinkage target: {target_type}")
        
        # Apply shrinkage
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target
        
        return shrunk_cov
    
    def estimate_regime_covariance(self, returns: pd.DataFrame, regime_labels: np.ndarray,
                                 regime_id: int) -> Dict[str, Any]:
        """Estimate covariance matrix for a specific regime.
        
        Args:
            returns: DataFrame with asset returns
            regime_labels: Array of regime labels
            regime_id: ID of regime to estimate
            
        Returns:
            Dictionary with covariance matrix and estimation metadata
        """
        # Filter returns for this regime
        regime_mask = regime_labels == regime_id
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) == 0:
            raise ValueError(f"No observations found for regime {regime_id}")
        
        self.logger.info(f"Estimating covariance for regime {regime_id} with {len(regime_returns)} observations")
        
        result = {
            'regime_id': regime_id,
            'n_observations': len(regime_returns),
            'estimation_method': self.method
        }
        
        # Choose estimation method
        if self.method == "sample":
            cov_matrix = self.estimate_sample_covariance(regime_returns)
            result['covariance_matrix'] = cov_matrix
            result['shrinkage_intensity'] = 0.0
        
        elif self.method == "ledoit_wolf":
            cov_matrix, shrinkage = self.estimate_ledoit_wolf_covariance(regime_returns)
            result['covariance_matrix'] = cov_matrix
            result['shrinkage_intensity'] = shrinkage
        
        elif self.method == "oas":
            cov_matrix, shrinkage = self.estimate_oas_covariance(regime_returns)
            result['covariance_matrix'] = cov_matrix
            result['shrinkage_intensity'] = shrinkage
        
        elif self.method == "shrunk":
            # Custom shrinkage with specified target
            sample_cov = self.estimate_sample_covariance(regime_returns)
            
            # Determine shrinkage intensity based on sample size
            n_obs, n_assets = regime_returns.shape
            if n_obs < n_assets:
                shrinkage_intensity = 0.8  # High shrinkage for small samples
            elif n_obs < 2 * n_assets:
                shrinkage_intensity = 0.5  # Medium shrinkage
            else:
                shrinkage_intensity = 0.2  # Low shrinkage for large samples
            
            cov_matrix = self.apply_custom_shrinkage(sample_cov, shrinkage_intensity, self.shrinkage_target)
            result['covariance_matrix'] = cov_matrix
            result['shrinkage_intensity'] = shrinkage_intensity
        
        else:
            raise ValueError(f"Unknown covariance estimation method: {self.method}")
        
        # Validate the resulting matrix
        is_valid = self.validate_covariance_matrix(cov_matrix)
        result['is_valid'] = is_valid
        
        if not is_valid:
            self.logger.warning(f"Invalid covariance matrix for regime {regime_id}, applying regularization")
            cov_matrix = self._regularize_covariance_matrix(cov_matrix)
            result['covariance_matrix'] = cov_matrix
            result['regularized'] = True
        else:
            result['regularized'] = False
        
        return result
    
    def validate_covariance_matrix(self, cov_matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Validate that covariance matrix is positive semi-definite.
        
        Args:
            cov_matrix: Covariance matrix to validate
            tolerance: Numerical tolerance for eigenvalue check
            
        Returns:
            True if matrix is valid (positive semi-definite)
        """
        try:
            # Check if matrix is square
            if cov_matrix.shape[0] != cov_matrix.shape[1]:
                return False
            
            # Check if matrix is symmetric
            if not np.allclose(cov_matrix, cov_matrix.T, atol=tolerance):
                return False
            
            # Check if matrix is positive semi-definite via eigenvalues
            eigenvalues = linalg.eigvals(cov_matrix)
            min_eigenvalue = np.min(eigenvalues)
            
            if min_eigenvalue < -tolerance:
                self.logger.warning(f"Negative eigenvalue detected: {min_eigenvalue}")
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(cov_matrix).all():
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error validating covariance matrix: {str(e)}")
            return False
    
    def _regularize_covariance_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Regularize covariance matrix to ensure positive semi-definiteness.
        
        Args:
            cov_matrix: Potentially invalid covariance matrix
            
        Returns:
            Regularized positive semi-definite covariance matrix
        """
        try:
            # Method 1: Eigenvalue clipping
            eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
            
            # Clip negative eigenvalues
            min_eigenvalue = max(self.regularization_factor, np.max(eigenvalues) * 1e-6)
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            
            # Reconstruct matrix
            regularized_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # Ensure symmetry
            regularized_cov = (regularized_cov + regularized_cov.T) / 2
            
            return regularized_cov
        
        except Exception as e:
            self.logger.error(f"Error in eigenvalue regularization: {str(e)}")
            
            # Fallback: Add ridge regularization
            n_assets = cov_matrix.shape[0]
            ridge_factor = np.trace(cov_matrix) / n_assets * 0.01
            return cov_matrix + ridge_factor * np.eye(n_assets)
    
    def calculate_condition_number(self, cov_matrix: np.ndarray) -> float:
        """Calculate condition number of covariance matrix.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Condition number (ratio of largest to smallest eigenvalue)
        """
        try:
            eigenvalues = linalg.eigvals(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zero/negative eigenvalues
            
            if len(eigenvalues) == 0:
                return float('inf')
            
            condition_number = np.max(eigenvalues) / np.min(eigenvalues)
            return float(condition_number)
        
        except Exception:
            return float('inf')
    
    def calculate_frobenius_norm(self, cov_matrix: np.ndarray) -> float:
        """Calculate Frobenius norm of covariance matrix.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Frobenius norm
        """
        return float(np.linalg.norm(cov_matrix, 'fro'))


class ReturnEstimator:
    """Estimates expected returns for different regimes."""
    
    def __init__(self, method: str = "sample_mean", outlier_threshold: float = 3.0):
        """Initialize return estimator.
        
        Args:
            method: Estimation method ("sample_mean", "robust_mean", "shrunk_mean", "capm", "factor_model")
            outlier_threshold: Threshold for outlier detection (in standard deviations)
        """
        self.method = method.lower()
        self.outlier_threshold = outlier_threshold
        self.logger = get_logger(__name__)
        
        # Parameters for advanced methods
        self.risk_free_rate = 0.0  # Assumed risk-free rate
        self.market_risk_premium = 0.06  # Assumed annual market risk premium
    
    def estimate_sample_mean(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate sample mean returns.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Array of mean returns
        """
        return returns.mean().values
    
    def estimate_robust_mean(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate robust mean returns with outlier handling.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Array of robust mean returns
        """
        robust_means = []
        
        for col in returns.columns:
            asset_returns = returns[col].values
            
            # Remove outliers using z-score
            z_scores = np.abs((asset_returns - np.mean(asset_returns)) / np.std(asset_returns))
            clean_returns = asset_returns[z_scores <= self.outlier_threshold]
            
            if len(clean_returns) > 0:
                robust_means.append(np.mean(clean_returns))
            else:
                # Fallback to median if all returns are outliers
                robust_means.append(np.median(asset_returns))
        
        return np.array(robust_means)
    
    def estimate_shrunk_mean(self, returns: pd.DataFrame, shrinkage_target: float = 0.0,
                           shrinkage_intensity: float = 0.1) -> np.ndarray:
        """Estimate shrunk mean returns towards a target.
        
        Args:
            returns: DataFrame with asset returns
            shrinkage_target: Target mean return for shrinkage
            shrinkage_intensity: Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage)
            
        Returns:
            Array of shrunk mean returns
        """
        sample_means = self.estimate_sample_mean(returns)
        
        # Shrink towards target
        shrunk_means = (1 - shrinkage_intensity) * sample_means + shrinkage_intensity * shrinkage_target
        
        return shrunk_means
    
    def estimate_capm_returns(self, returns: pd.DataFrame, market_returns: Optional[pd.Series] = None) -> np.ndarray:
        """Estimate expected returns using CAPM model.
        
        Args:
            returns: DataFrame with asset returns
            market_returns: Market return series (if None, uses equal-weighted portfolio)
            
        Returns:
            Array of CAPM-based expected returns
        """
        if market_returns is None:
            # Use equal-weighted portfolio as market proxy
            market_returns = returns.mean(axis=1)
        
        # Align market returns with asset returns
        aligned_market = market_returns.reindex(returns.index).dropna()
        aligned_returns = returns.reindex(aligned_market.index).dropna()
        
        capm_returns = []
        
        for col in aligned_returns.columns:
            asset_returns = aligned_returns[col]
            
            # Calculate beta using linear regression
            covariance = np.cov(asset_returns, aligned_market)[0, 1]
            market_variance = np.var(aligned_market)
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            # CAPM expected return: R_f + beta * (R_m - R_f)
            # Assuming daily returns, convert annual risk premium to daily
            daily_risk_premium = self.market_risk_premium / 252
            expected_return = self.risk_free_rate / 252 + beta * daily_risk_premium
            
            capm_returns.append(expected_return)
        
        return np.array(capm_returns)
    
    def estimate_factor_model_returns(self, returns: pd.DataFrame, factors: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Estimate expected returns using multi-factor model.
        
        Args:
            returns: DataFrame with asset returns
            factors: DataFrame with factor returns (if None, uses PCA factors)
            
        Returns:
            Array of factor model expected returns
        """
        if factors is None:
            # Create factors using PCA
            factors = self._create_pca_factors(returns, n_factors=3)
        
        # Align data
        aligned_data = returns.join(factors, how='inner').dropna()
        aligned_returns = aligned_data[returns.columns]
        aligned_factors = aligned_data[factors.columns]
        
        factor_returns = []
        
        for col in aligned_returns.columns:
            asset_returns = aligned_returns[col].values
            
            # Multiple regression: R_i = alpha + beta_1*F_1 + ... + beta_k*F_k + epsilon
            try:
                from sklearn.linear_model import LinearRegression
                
                model = LinearRegression()
                model.fit(aligned_factors.values, asset_returns)
                
                # Expected return is the intercept (alpha) plus factor risk premiums
                factor_premiums = aligned_factors.mean().values
                expected_return = model.intercept_ + np.dot(model.coef_, factor_premiums)
                
                factor_returns.append(expected_return)
            
            except ImportError:
                # Fallback to simple mean if sklearn not available
                factor_returns.append(np.mean(asset_returns))
            except Exception as e:
                self.logger.warning(f"Factor model failed for {col}: {str(e)}")
                factor_returns.append(np.mean(asset_returns))
        
        return np.array(factor_returns)
    
    def _create_pca_factors(self, returns: pd.DataFrame, n_factors: int = 3) -> pd.DataFrame:
        """Create factors using Principal Component Analysis."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize returns
            scaler = StandardScaler()
            scaled_returns = scaler.fit_transform(returns.dropna())
            
            # Apply PCA
            pca = PCA(n_components=n_factors)
            factor_scores = pca.fit_transform(scaled_returns)
            
            # Create factor DataFrame
            factor_names = [f'PC{i+1}' for i in range(n_factors)]
            factors = pd.DataFrame(
                factor_scores, 
                index=returns.dropna().index, 
                columns=factor_names
            )
            
            return factors
        
        except ImportError:
            self.logger.warning("sklearn not available for PCA factors")
            # Return simple factors based on market and size
            market_factor = returns.mean(axis=1)
            size_factor = returns.iloc[:, :len(returns.columns)//2].mean(axis=1) - returns.iloc[:, len(returns.columns)//2:].mean(axis=1)
            
            return pd.DataFrame({
                'Market': market_factor,
                'Size': size_factor,
                'Momentum': market_factor.rolling(20).mean() - market_factor.rolling(60).mean()
            })
    
    def estimate_black_litterman_returns(self, returns: pd.DataFrame, market_cap_weights: Optional[np.ndarray] = None,
                                       views: Optional[Dict[str, float]] = None, 
                                       view_confidence: float = 0.1) -> np.ndarray:
        """Estimate expected returns using Black-Litterman model.
        
        Args:
            returns: DataFrame with asset returns
            market_cap_weights: Market capitalization weights (if None, uses equal weights)
            views: Dictionary of analyst views {asset_name: expected_return}
            view_confidence: Confidence in views (lower = more confident)
            
        Returns:
            Array of Black-Litterman expected returns
        """
        n_assets = len(returns.columns)
        
        # Market capitalization weights (equal if not provided)
        if market_cap_weights is None:
            market_cap_weights = np.ones(n_assets) / n_assets
        
        # Estimate covariance matrix
        cov_matrix = returns.cov().values
        
        # Implied equilibrium returns (reverse optimization)
        risk_aversion = 3.0  # Typical risk aversion parameter
        implied_returns = risk_aversion * cov_matrix @ market_cap_weights
        
        # If no views provided, return implied returns
        if not views:
            return implied_returns
        
        # Incorporate views using Black-Litterman formula
        try:
            # Create picking matrix P and view vector Q
            view_assets = [asset for asset in views.keys() if asset in returns.columns]
            if not view_assets:
                return implied_returns
            
            n_views = len(view_assets)
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)
            
            for i, asset in enumerate(view_assets):
                asset_idx = returns.columns.get_loc(asset)
                P[i, asset_idx] = 1.0
                Q[i] = views[asset]
            
            # View uncertainty matrix (diagonal)
            Omega = view_confidence * np.eye(n_views)
            
            # Black-Litterman formula
            tau = 0.05  # Scaling factor for uncertainty of prior
            
            M1 = linalg.inv(tau * cov_matrix)
            M2 = P.T @ linalg.inv(Omega) @ P
            M3 = linalg.inv(tau * cov_matrix) @ implied_returns
            M4 = P.T @ linalg.inv(Omega) @ Q
            
            bl_returns = linalg.inv(M1 + M2) @ (M3 + M4)
            
            return bl_returns
        
        except Exception as e:
            self.logger.warning(f"Black-Litterman estimation failed: {str(e)}")
            return implied_returns
    
    def estimate_momentum_returns(self, returns: pd.DataFrame, lookback_periods: List[int] = [20, 60, 120]) -> np.ndarray:
        """Estimate expected returns based on momentum signals.
        
        Args:
            returns: DataFrame with asset returns
            lookback_periods: List of lookback periods for momentum calculation
            
        Returns:
            Array of momentum-based expected returns
        """
        momentum_returns = []
        
        for col in returns.columns:
            asset_returns = returns[col]
            
            # Calculate momentum signals for different periods
            momentum_signals = []
            for period in lookback_periods:
                if len(asset_returns) >= period:
                    momentum = asset_returns.rolling(period).mean().iloc[-1]
                    momentum_signals.append(momentum)
            
            # Average momentum signal
            if momentum_signals:
                avg_momentum = np.mean(momentum_signals)
                # Scale momentum to reasonable expected return range
                expected_return = np.clip(avg_momentum * 0.5, -0.05, 0.05)  # Cap at ±5% daily
            else:
                expected_return = 0.0
            
            momentum_returns.append(expected_return)
        
        return np.array(momentum_returns)
    
    def estimate_mean_reversion_returns(self, returns: pd.DataFrame, lookback_period: int = 252) -> np.ndarray:
        """Estimate expected returns based on mean reversion.
        
        Args:
            returns: DataFrame with asset returns
            lookback_period: Lookback period for mean reversion calculation
            
        Returns:
            Array of mean reversion expected returns
        """
        mean_reversion_returns = []
        
        for col in returns.columns:
            asset_returns = returns[col]
            
            if len(asset_returns) >= lookback_period:
                # Calculate long-term mean
                long_term_mean = asset_returns.rolling(lookback_period).mean().iloc[-1]
                
                # Calculate recent performance
                recent_period = min(20, len(asset_returns))
                recent_mean = asset_returns.tail(recent_period).mean()
                
                # Mean reversion signal: expect return towards long-term mean
                reversion_signal = long_term_mean - recent_mean
                
                # Scale to reasonable range
                expected_return = np.clip(reversion_signal * 0.1, -0.02, 0.02)
            else:
                expected_return = 0.0
            
            mean_reversion_returns.append(expected_return)
        
        return np.array(mean_reversion_returns)
    
    def estimate_regime_returns(self, returns: pd.DataFrame, regime_labels: np.ndarray,
                              regime_id: int) -> Dict[str, Any]:
        """Estimate expected returns for a specific regime.
        
        Args:
            returns: DataFrame with asset returns
            regime_labels: Array of regime labels
            regime_id: ID of regime to estimate
            
        Returns:
            Dictionary with return estimates and metadata
        """
        # Filter returns for this regime
        regime_mask = regime_labels == regime_id
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) == 0:
            raise ValueError(f"No observations found for regime {regime_id}")
        
        self.logger.info(f"Estimating returns for regime {regime_id} with {len(regime_returns)} observations")
        
        result = {
            'regime_id': regime_id,
            'n_observations': len(regime_returns),
            'estimation_method': self.method,
            'asset_names': list(returns.columns)
        }
        
        # Choose estimation method
        if self.method == "sample_mean":
            mean_returns = self.estimate_sample_mean(regime_returns)
        
        elif self.method == "robust_mean":
            mean_returns = self.estimate_robust_mean(regime_returns)
        
        elif self.method == "shrunk_mean":
            # Shrink towards zero (risk-free rate assumption)
            shrinkage_intensity = min(0.5, 10.0 / len(regime_returns))  # More shrinkage for smaller samples
            mean_returns = self.estimate_shrunk_mean(regime_returns, 0.0, shrinkage_intensity)
            result['shrinkage_intensity'] = shrinkage_intensity
        
        elif self.method == "capm":
            mean_returns = self.estimate_capm_returns(regime_returns)
        
        elif self.method == "factor_model":
            mean_returns = self.estimate_factor_model_returns(regime_returns)
        
        elif self.method == "black_litterman":
            mean_returns = self.estimate_black_litterman_returns(regime_returns)
        
        elif self.method == "momentum":
            mean_returns = self.estimate_momentum_returns(regime_returns)
        
        elif self.method == "mean_reversion":
            mean_returns = self.estimate_mean_reversion_returns(regime_returns)
        
        else:
            raise ValueError(f"Unknown return estimation method: {self.method}")
        
        result['mean_returns'] = mean_returns
        
        # Calculate additional statistics
        result['return_volatility'] = regime_returns.std().values
        result['return_skewness'] = regime_returns.skew().values
        result['return_kurtosis'] = regime_returns.kurtosis().values
        
        # Calculate confidence intervals (assuming normal distribution)
        n_obs = len(regime_returns)
        std_errors = regime_returns.std().values / np.sqrt(n_obs)
        confidence_level = 0.95
        t_critical = 1.96  # Approximate for large samples
        
        result['confidence_intervals'] = {
            'lower': mean_returns - t_critical * std_errors,
            'upper': mean_returns + t_critical * std_errors,
            'confidence_level': confidence_level
        }
        
        return result
    
    def forecast_returns(self, returns: pd.DataFrame, forecast_horizon: int = 1,
                        method: str = "historical") -> Dict[str, Any]:
        """Forecast future returns using various methods.
        
        Args:
            returns: DataFrame with historical returns
            forecast_horizon: Number of periods to forecast
            method: Forecasting method ("historical", "ar", "garch", "ensemble")
            
        Returns:
            Dictionary with forecasted returns and confidence intervals
        """
        self.logger.info(f"Forecasting returns for {forecast_horizon} periods using {method}")
        
        forecast_result = {
            'method': method,
            'forecast_horizon': forecast_horizon,
            'asset_names': list(returns.columns)
        }
        
        if method == "historical":
            # Simple historical average
            forecasted_returns = self.estimate_sample_mean(returns)
            forecast_std = returns.std().values
        
        elif method == "ar":
            # Autoregressive model
            forecasted_returns, forecast_std = self._forecast_ar_model(returns, forecast_horizon)
        
        elif method == "garch":
            # GARCH model for volatility forecasting
            forecasted_returns, forecast_std = self._forecast_garch_model(returns, forecast_horizon)
        
        elif method == "ensemble":
            # Ensemble of multiple methods
            forecasted_returns, forecast_std = self._forecast_ensemble(returns, forecast_horizon)
        
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
        
        # Calculate confidence intervals
        confidence_levels = [0.68, 0.95, 0.99]  # 1, 2, 3 sigma
        confidence_intervals = {}
        
        for conf_level in confidence_levels:
            z_score = {0.68: 1.0, 0.95: 1.96, 0.99: 2.58}[conf_level]
            
            confidence_intervals[f'{int(conf_level*100)}%'] = {
                'lower': forecasted_returns - z_score * forecast_std,
                'upper': forecasted_returns + z_score * forecast_std
            }
        
        forecast_result.update({
            'forecasted_returns': forecasted_returns,
            'forecast_std': forecast_std,
            'confidence_intervals': confidence_intervals
        })
        
        return forecast_result
    
    def _forecast_ar_model(self, returns: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast using autoregressive model."""
        forecasted_returns = []
        forecast_stds = []
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            
            try:
                # Simple AR(1) model: r_t = c + phi * r_{t-1} + epsilon_t
                if len(asset_returns) >= 20:
                    # Estimate AR(1) parameters
                    y = asset_returns[1:].values
                    x = asset_returns[:-1].values
                    
                    # Add constant term
                    X = np.column_stack([np.ones(len(x)), x])
                    
                    # OLS estimation
                    params = linalg.lstsq(X, y)[0]
                    c, phi = params[0], params[1]
                    
                    # Forecast
                    last_return = asset_returns.iloc[-1]
                    forecast = c + phi * last_return
                    
                    # Estimate residual standard deviation
                    fitted = X @ params
                    residuals = y - fitted
                    residual_std = np.std(residuals)
                    
                    forecasted_returns.append(forecast)
                    forecast_stds.append(residual_std)
                
                else:
                    # Fallback to historical mean
                    forecasted_returns.append(np.mean(asset_returns))
                    forecast_stds.append(np.std(asset_returns))
            
            except Exception as e:
                self.logger.warning(f"AR model failed for {col}: {str(e)}")
                forecasted_returns.append(np.mean(asset_returns))
                forecast_stds.append(np.std(asset_returns))
        
        return np.array(forecasted_returns), np.array(forecast_stds)
    
    def _forecast_garch_model(self, returns: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast using GARCH model (simplified version)."""
        forecasted_returns = []
        forecast_stds = []
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            
            try:
                # Simplified GARCH(1,1): sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
                if len(asset_returns) >= 50:
                    # Estimate unconditional mean
                    mean_return = np.mean(asset_returns)
                    
                    # Estimate volatility using exponential smoothing (EWMA)
                    lambda_ewma = 0.94  # RiskMetrics parameter
                    squared_returns = (asset_returns - mean_return) ** 2
                    
                    # Calculate EWMA variance
                    weights = np.array([(1 - lambda_ewma) * (lambda_ewma ** i) for i in range(len(squared_returns))])
                    weights = weights[::-1]  # Reverse to match time order
                    weights = weights / np.sum(weights)  # Normalize
                    
                    ewma_variance = np.sum(weights * squared_returns.values)
                    forecast_volatility = np.sqrt(ewma_variance)
                    
                    forecasted_returns.append(mean_return)
                    forecast_stds.append(forecast_volatility)
                
                else:
                    # Fallback to historical statistics
                    forecasted_returns.append(np.mean(asset_returns))
                    forecast_stds.append(np.std(asset_returns))
            
            except Exception as e:
                self.logger.warning(f"GARCH model failed for {col}: {str(e)}")
                forecasted_returns.append(np.mean(asset_returns))
                forecast_stds.append(np.std(asset_returns))
        
        return np.array(forecasted_returns), np.array(forecast_stds)
    
    def _forecast_ensemble(self, returns: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast using ensemble of methods."""
        # Get forecasts from different methods
        historical_returns = self.estimate_sample_mean(returns)
        historical_stds = returns.std().values
        
        ar_returns, ar_stds = self._forecast_ar_model(returns, horizon)
        garch_returns, garch_stds = self._forecast_garch_model(returns, horizon)
        
        # Ensemble weights (can be optimized based on historical performance)
        weights = np.array([0.4, 0.3, 0.3])  # Historical, AR, GARCH
        
        # Weighted average
        ensemble_returns = (weights[0] * historical_returns + 
                          weights[1] * ar_returns + 
                          weights[2] * garch_returns)
        
        ensemble_stds = (weights[0] * historical_stds + 
                        weights[1] * ar_stds + 
                        weights[2] * garch_stds)
        
        return ensemble_returns, ensemble_stds
    
    def calculate_return_attribution(self, returns: pd.DataFrame, 
                                   benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Calculate return attribution analysis.
        
        Args:
            returns: DataFrame with asset returns
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary with attribution analysis
        """
        # Align data
        aligned_benchmark = benchmark_returns.reindex(returns.index).dropna()
        aligned_returns = returns.reindex(aligned_benchmark.index).dropna()
        
        attribution_results = {}
        
        for col in aligned_returns.columns:
            asset_returns = aligned_returns[col]
            
            # Calculate beta and alpha
            covariance = np.cov(asset_returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha (Jensen's alpha)
            asset_mean = np.mean(asset_returns)
            benchmark_mean = np.mean(aligned_benchmark)
            alpha = asset_mean - beta * benchmark_mean
            
            # Tracking error
            active_returns = asset_returns - aligned_benchmark
            tracking_error = np.std(active_returns)
            
            # Information ratio
            information_ratio = np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
            
            attribution_results[col] = {
                'alpha': float(alpha),
                'beta': float(beta),
                'tracking_error': float(tracking_error),
                'information_ratio': float(information_ratio),
                'correlation': float(np.corrcoef(asset_returns, aligned_benchmark)[0, 1])
            }
        
        return attribution_results


class RiskValidator:
    """Validates risk parameter estimates and provides quality metrics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_regime_parameters(self, regime_params: RegimeParameters) -> Dict[str, Any]:
        """Validate regime parameters for consistency and quality.
        
        Args:
            regime_params: RegimeParameters object to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate covariance matrix
        cov_validation = self._validate_covariance_properties(regime_params.covariance_matrix)
        validation_results.update(cov_validation)
        
        # Validate return estimates
        return_validation = self._validate_return_estimates(regime_params.mean_returns)
        validation_results['return_validation'] = return_validation
        
        # Check consistency between returns and covariance
        consistency_check = self._check_return_covariance_consistency(
            regime_params.mean_returns, regime_params.covariance_matrix
        )
        validation_results['consistency_check'] = consistency_check
        
        # Overall validation status
        if validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results
    
    def _validate_covariance_properties(self, cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Validate covariance matrix properties."""
        results = {
            'covariance_valid': True,
            'covariance_warnings': [],
            'covariance_errors': []
        }
        
        try:
            # Check dimensions
            if cov_matrix.shape[0] != cov_matrix.shape[1]:
                results['covariance_errors'].append("Covariance matrix is not square")
                results['covariance_valid'] = False
                return results
            
            n_assets = cov_matrix.shape[0]
            
            # Check symmetry
            if not np.allclose(cov_matrix, cov_matrix.T, atol=1e-8):
                results['covariance_errors'].append("Covariance matrix is not symmetric")
                results['covariance_valid'] = False
            
            # Check positive semi-definiteness
            eigenvalues = linalg.eigvals(cov_matrix)
            min_eigenvalue = np.min(eigenvalues)
            
            if min_eigenvalue < -1e-8:
                results['covariance_errors'].append(f"Covariance matrix is not positive semi-definite (min eigenvalue: {min_eigenvalue})")
                results['covariance_valid'] = False
            elif min_eigenvalue < 1e-10:
                results['covariance_warnings'].append("Covariance matrix is nearly singular")
            
            # Check condition number
            condition_number = np.max(eigenvalues) / max(np.min(eigenvalues[eigenvalues > 0]), 1e-12)
            if condition_number > 1e12:
                results['covariance_warnings'].append(f"High condition number: {condition_number:.2e}")
            
            # Check diagonal elements (variances)
            variances = np.diag(cov_matrix)
            if np.any(variances <= 0):
                results['covariance_errors'].append("Non-positive variances detected")
                results['covariance_valid'] = False
            
            # Check for extreme values
            if np.any(variances > 1.0):  # Daily variance > 100%
                results['covariance_warnings'].append("Extremely high variances detected")
            
            # Check correlations
            correlations = cov_matrix / np.sqrt(np.outer(variances, variances))
            np.fill_diagonal(correlations, 1.0)  # Ensure diagonal is exactly 1
            
            if np.any(np.abs(correlations) > 1.0001):  # Allow small numerical errors
                results['covariance_warnings'].append("Correlations outside [-1, 1] range")
            
            # Store additional metrics
            results['condition_number'] = float(condition_number)
            results['min_eigenvalue'] = float(min_eigenvalue)
            results['max_eigenvalue'] = float(np.max(eigenvalues))
            results['trace'] = float(np.trace(cov_matrix))
            results['frobenius_norm'] = float(np.linalg.norm(cov_matrix, 'fro'))
        
        except Exception as e:
            results['covariance_errors'].append(f"Error validating covariance matrix: {str(e)}")
            results['covariance_valid'] = False
        
        return results
    
    def _validate_return_estimates(self, mean_returns: np.ndarray) -> Dict[str, Any]:
        """Validate return estimates."""
        results = {
            'returns_valid': True,
            'returns_warnings': [],
            'returns_errors': []
        }
        
        try:
            # Check for NaN or infinite values
            if not np.isfinite(mean_returns).all():
                results['returns_errors'].append("Non-finite values in return estimates")
                results['returns_valid'] = False
                return results
            
            # Check for extreme values (daily returns > 50%)
            if np.any(np.abs(mean_returns) > 0.5):
                results['returns_warnings'].append("Extreme return estimates detected")
            
            # Check for very high positive returns (daily > 10%)
            if np.any(mean_returns > 0.1):
                results['returns_warnings'].append("Very high positive return estimates")
            
            # Check for very negative returns (daily < -10%)
            if np.any(mean_returns < -0.1):
                results['returns_warnings'].append("Very negative return estimates")
            
            # Store statistics
            results['mean_return'] = float(np.mean(mean_returns))
            results['std_return'] = float(np.std(mean_returns))
            results['min_return'] = float(np.min(mean_returns))
            results['max_return'] = float(np.max(mean_returns))
        
        except Exception as e:
            results['returns_errors'].append(f"Error validating returns: {str(e)}")
            results['returns_valid'] = False
        
        return results
    
    def _check_return_covariance_consistency(self, mean_returns: np.ndarray, 
                                           cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Check consistency between return estimates and covariance matrix."""
        results = {
            'consistent': True,
            'warnings': []
        }
        
        try:
            if len(mean_returns) != cov_matrix.shape[0]:
                results['consistent'] = False
                results['warnings'].append("Dimension mismatch between returns and covariance")
                return results
            
            # Check if high return assets have reasonable risk
            variances = np.diag(cov_matrix)
            volatilities = np.sqrt(variances)
            
            # Calculate Sharpe ratios (assuming zero risk-free rate)
            sharpe_ratios = mean_returns / volatilities
            
            # Check for extreme Sharpe ratios
            if np.any(np.abs(sharpe_ratios) > 5.0):
                results['warnings'].append("Extreme Sharpe ratios detected")
            
            # Store metrics
            results['sharpe_ratios'] = sharpe_ratios.tolist()
            results['mean_sharpe'] = float(np.mean(sharpe_ratios))
            results['std_sharpe'] = float(np.std(sharpe_ratios))
        
        except Exception as e:
            results['warnings'].append(f"Error checking consistency: {str(e)}")
        
        return results


class RiskParameterStorage:
    """Handles storage and retrieval of risk parameters."""
    
    def __init__(self, storage_path: str = "data/risk_parameters.pkl"):
        """Initialize parameter storage.
        
        Args:
            storage_path: Path to storage file
        """
        self.storage_path = storage_path
        self.logger = get_logger(__name__)
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    def save_regime_parameters(self, regime_parameters: Dict[int, RegimeParameters], 
                             metadata: Optional[Dict] = None) -> bool:
        """Save regime parameters to storage.
        
        Args:
            regime_parameters: Dictionary of regime parameters
            metadata: Optional metadata about the estimation
            
        Returns:
            True if successful
        """
        try:
            import pickle
            from datetime import datetime
            
            storage_data = {
                'regime_parameters': regime_parameters,
                'metadata': metadata or {},
                'timestamp': datetime.now(),
                'version': '1.0'
            }
            
            with open(self.storage_path, 'wb') as f:
                pickle.dump(storage_data, f)
            
            self.logger.info(f"Saved regime parameters to {self.storage_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save regime parameters: {str(e)}")
            return False
    
    def load_regime_parameters(self) -> Optional[Dict[int, RegimeParameters]]:
        """Load regime parameters from storage.
        
        Returns:
            Dictionary of regime parameters or None if not found
        """
        try:
            import pickle
            import os
            
            if not os.path.exists(self.storage_path):
                self.logger.info("No stored regime parameters found")
                return None
            
            with open(self.storage_path, 'rb') as f:
                storage_data = pickle.load(f)
            
            regime_parameters = storage_data.get('regime_parameters', {})
            metadata = storage_data.get('metadata', {})
            timestamp = storage_data.get('timestamp')
            
            self.logger.info(f"Loaded regime parameters from {self.storage_path} (saved: {timestamp})")
            return regime_parameters
        
        except Exception as e:
            self.logger.error(f"Failed to load regime parameters: {str(e)}")
            return None
    
    def get_storage_metadata(self) -> Optional[Dict]:
        """Get metadata about stored parameters.
        
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            import pickle
            import os
            
            if not os.path.exists(self.storage_path):
                return None
            
            with open(self.storage_path, 'rb') as f:
                storage_data = pickle.load(f)
            
            return storage_data.get('metadata', {})
        
        except Exception as e:
            self.logger.error(f"Failed to get storage metadata: {str(e)}")
            return None
    
    def export_parameters_to_csv(self, regime_parameters: Dict[int, RegimeParameters], 
                               output_dir: str = "exports") -> Dict[str, str]:
        """Export regime parameters to CSV files.
        
        Args:
            regime_parameters: Dictionary of regime parameters
            output_dir: Output directory for CSV files
            
        Returns:
            Dictionary mapping file types to file paths
        """
        try:
            import os
            import pandas as pd
            
            os.makedirs(output_dir, exist_ok=True)
            exported_files = {}
            
            # Export mean returns
            returns_data = []
            for regime_id, params in regime_parameters.items():
                for i, ret in enumerate(params.mean_returns):
                    returns_data.append({
                        'regime_id': regime_id,
                        'asset_index': i,
                        'mean_return': ret
                    })
            
            if returns_data:
                returns_df = pd.DataFrame(returns_data)
                returns_path = os.path.join(output_dir, 'regime_returns.csv')
                returns_df.to_csv(returns_path, index=False)
                exported_files['returns'] = returns_path
            
            # Export covariance matrices
            for regime_id, params in regime_parameters.items():
                cov_df = pd.DataFrame(params.covariance_matrix)
                cov_path = os.path.join(output_dir, f'regime_{regime_id}_covariance.csv')
                cov_df.to_csv(cov_path, index=False)
                exported_files[f'covariance_regime_{regime_id}'] = cov_path
            
            # Export summary statistics
            summary_data = []
            for regime_id, params in regime_parameters.items():
                summary_data.append({
                    'regime_id': regime_id,
                    'regime_probability': params.regime_probability,
                    'start_date': params.start_date,
                    'end_date': params.end_date,
                    'n_assets': len(params.mean_returns),
                    'avg_return': np.mean(params.mean_returns),
                    'avg_volatility': np.mean(np.sqrt(np.diag(params.covariance_matrix)))
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(output_dir, 'regime_summary.csv')
                summary_df.to_csv(summary_path, index=False)
                exported_files['summary'] = summary_path
            
            self.logger.info(f"Exported regime parameters to {output_dir}")
            return exported_files
        
        except Exception as e:
            self.logger.error(f"Failed to export parameters: {str(e)}")
            return {}


class RiskParameterValidator:
    """Enhanced validator for risk parameters with comprehensive checks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.validation_history = []
    
    def validate_parameter_consistency(self, regime_parameters: Dict[int, RegimeParameters]) -> Dict[str, Any]:
        """Validate consistency across all regime parameters.
        
        Args:
            regime_parameters: Dictionary of all regime parameters
            
        Returns:
            Dictionary with comprehensive validation results
        """
        self.logger.info("Validating parameter consistency across regimes")
        
        validation_results = {
            'overall_valid': True,
            'regime_validations': {},
            'cross_regime_checks': {},
            'warnings': [],
            'errors': []
        }
        
        if not regime_parameters:
            validation_results['overall_valid'] = False
            validation_results['errors'].append("No regime parameters provided")
            return validation_results
        
        # Validate each regime individually
        for regime_id, params in regime_parameters.items():
            regime_validation = self._validate_single_regime(params)
            validation_results['regime_validations'][regime_id] = regime_validation
            
            if not regime_validation['valid']:
                validation_results['overall_valid'] = False
        
        # Cross-regime consistency checks
        cross_checks = self._validate_cross_regime_consistency(regime_parameters)
        validation_results['cross_regime_checks'] = cross_checks
        
        if cross_checks['errors']:
            validation_results['overall_valid'] = False
            validation_results['errors'].extend(cross_checks['errors'])
        
        validation_results['warnings'].extend(cross_checks['warnings'])
        
        # Store validation in history
        self.validation_history.append({
            'timestamp': pd.Timestamp.now(),
            'n_regimes': len(regime_parameters),
            'overall_valid': validation_results['overall_valid'],
            'n_errors': len(validation_results['errors']),
            'n_warnings': len(validation_results['warnings'])
        })
        
        return validation_results
    
    def _validate_single_regime(self, params: RegimeParameters) -> Dict[str, Any]:
        """Validate a single regime's parameters."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Basic structure validation
            if params.mean_returns is None or params.covariance_matrix is None:
                validation['errors'].append("Missing mean returns or covariance matrix")
                validation['valid'] = False
                return validation
            
            n_assets = len(params.mean_returns)
            
            # Dimension consistency
            if params.covariance_matrix.shape != (n_assets, n_assets):
                validation['errors'].append(f"Dimension mismatch: returns({n_assets}) vs covariance{params.covariance_matrix.shape}")
                validation['valid'] = False
            
            # Covariance matrix validation
            cov_validation = self._validate_covariance_matrix_detailed(params.covariance_matrix)
            validation['metrics'].update(cov_validation['metrics'])
            
            if not cov_validation['valid']:
                validation['errors'].extend(cov_validation['errors'])
                validation['valid'] = False
            
            validation['warnings'].extend(cov_validation['warnings'])
            
            # Return validation
            return_validation = self._validate_returns_detailed(params.mean_returns)
            validation['metrics'].update(return_validation['metrics'])
            
            if not return_validation['valid']:
                validation['errors'].extend(return_validation['errors'])
                validation['valid'] = False
            
            validation['warnings'].extend(return_validation['warnings'])
            
            # Probability validation
            if not (0 <= params.regime_probability <= 1):
                validation['errors'].append(f"Invalid regime probability: {params.regime_probability}")
                validation['valid'] = False
            
            # Date validation
            if params.start_date > params.end_date:
                validation['errors'].append("Start date is after end date")
                validation['valid'] = False
        
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            validation['valid'] = False
        
        return validation
    
    def _validate_covariance_matrix_detailed(self, cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Detailed covariance matrix validation."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Basic checks
            if not isinstance(cov_matrix, np.ndarray):
                validation['errors'].append("Covariance matrix is not a numpy array")
                validation['valid'] = False
                return validation
            
            if cov_matrix.ndim != 2:
                validation['errors'].append("Covariance matrix is not 2-dimensional")
                validation['valid'] = False
                return validation
            
            n = cov_matrix.shape[0]
            if cov_matrix.shape[1] != n:
                validation['errors'].append("Covariance matrix is not square")
                validation['valid'] = False
                return validation
            
            # Numerical checks
            if not np.isfinite(cov_matrix).all():
                validation['errors'].append("Covariance matrix contains non-finite values")
                validation['valid'] = False
                return validation
            
            # Symmetry check
            symmetry_error = np.max(np.abs(cov_matrix - cov_matrix.T))
            validation['metrics']['symmetry_error'] = float(symmetry_error)
            
            if symmetry_error > 1e-10:
                validation['errors'].append(f"Matrix not symmetric (max error: {symmetry_error})")
                validation['valid'] = False
            
            # Eigenvalue analysis
            eigenvalues = linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues)  # Take real part (should be real for symmetric matrix)
            
            min_eigenvalue = float(np.min(eigenvalues))
            max_eigenvalue = float(np.max(eigenvalues))
            condition_number = max_eigenvalue / max(min_eigenvalue, 1e-16)
            
            validation['metrics'].update({
                'min_eigenvalue': min_eigenvalue,
                'max_eigenvalue': max_eigenvalue,
                'condition_number': float(condition_number),
                'rank': int(np.linalg.matrix_rank(cov_matrix)),
                'trace': float(np.trace(cov_matrix)),
                'determinant': float(linalg.det(cov_matrix))
            })
            
            # Positive semi-definiteness
            if min_eigenvalue < -1e-10:
                validation['errors'].append(f"Matrix not positive semi-definite (min eigenvalue: {min_eigenvalue})")
                validation['valid'] = False
            elif min_eigenvalue < 1e-12:
                validation['warnings'].append("Matrix is nearly singular")
            
            # Condition number check
            if condition_number > 1e12:
                validation['warnings'].append(f"High condition number: {condition_number:.2e}")
            
            # Variance checks
            variances = np.diag(cov_matrix)
            validation['metrics'].update({
                'min_variance': float(np.min(variances)),
                'max_variance': float(np.max(variances)),
                'mean_variance': float(np.mean(variances))
            })
            
            if np.any(variances <= 0):
                validation['errors'].append("Non-positive variances on diagonal")
                validation['valid'] = False
            
            # Correlation matrix checks
            std_devs = np.sqrt(variances)
            correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
            
            # Check correlation bounds
            off_diagonal_corrs = correlation_matrix[np.triu_indices(n, k=1)]
            max_abs_corr = np.max(np.abs(off_diagonal_corrs))
            
            validation['metrics']['max_abs_correlation'] = float(max_abs_corr)
            
            if max_abs_corr > 1.001:  # Allow small numerical errors
                validation['warnings'].append(f"Correlations exceed bounds: {max_abs_corr}")
        
        except Exception as e:
            validation['errors'].append(f"Covariance validation error: {str(e)}")
            validation['valid'] = False
        
        return validation
    
    def _validate_returns_detailed(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detailed return validation."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Basic checks
            if not isinstance(returns, np.ndarray):
                validation['errors'].append("Returns is not a numpy array")
                validation['valid'] = False
                return validation
            
            if returns.ndim != 1:
                validation['errors'].append("Returns is not 1-dimensional")
                validation['valid'] = False
                return validation
            
            # Numerical checks
            if not np.isfinite(returns).all():
                validation['errors'].append("Returns contain non-finite values")
                validation['valid'] = False
                return validation
            
            # Statistical metrics
            validation['metrics'].update({
                'mean_return': float(np.mean(returns)),
                'std_return': float(np.std(returns)),
                'min_return': float(np.min(returns)),
                'max_return': float(np.max(returns)),
                'skewness': float(self._calculate_skewness(returns)),
                'kurtosis': float(self._calculate_kurtosis(returns))
            })
            
            # Extreme value checks
            if np.any(np.abs(returns) > 0.5):  # 50% daily return
                validation['warnings'].append("Extreme return values detected (>50% daily)")
            
            if np.any(returns > 0.2):  # 20% daily return
                validation['warnings'].append("Very high positive returns detected")
            
            if np.any(returns < -0.2):  # -20% daily return
                validation['warnings'].append("Very negative returns detected")
        
        except Exception as e:
            validation['errors'].append(f"Return validation error: {str(e)}")
            validation['valid'] = False
        
        return validation
    
    def _validate_cross_regime_consistency(self, regime_parameters: Dict[int, RegimeParameters]) -> Dict[str, Any]:
        """Validate consistency across regimes."""
        validation = {
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            regimes = list(regime_parameters.keys())
            n_regimes = len(regimes)
            
            if n_regimes < 2:
                return validation
            
            # Check dimension consistency
            dimensions = [len(params.mean_returns) for params in regime_parameters.values()]
            if len(set(dimensions)) > 1:
                validation['errors'].append(f"Inconsistent dimensions across regimes: {dimensions}")
                return validation
            
            n_assets = dimensions[0]
            
            # Probability consistency
            total_probability = sum(params.regime_probability for params in regime_parameters.values())
            validation['metrics']['total_probability'] = float(total_probability)
            
            if abs(total_probability - 1.0) > 0.01:
                validation['warnings'].append(f"Regime probabilities don't sum to 1: {total_probability}")
            
            # Return consistency across regimes
            all_returns = np.array([params.mean_returns for params in regime_parameters.values()])
            return_ranges = np.max(all_returns, axis=0) - np.min(all_returns, axis=0)
            
            validation['metrics'].update({
                'max_return_range': float(np.max(return_ranges)),
                'mean_return_range': float(np.mean(return_ranges)),
                'return_range_std': float(np.std(return_ranges))
            })
            
            # Covariance consistency
            all_volatilities = []
            for params in regime_parameters.values():
                volatilities = np.sqrt(np.diag(params.covariance_matrix))
                all_volatilities.append(volatilities)
            
            all_volatilities = np.array(all_volatilities)
            volatility_ranges = np.max(all_volatilities, axis=0) - np.min(all_volatilities, axis=0)
            
            validation['metrics'].update({
                'max_volatility_range': float(np.max(volatility_ranges)),
                'mean_volatility_range': float(np.mean(volatility_ranges)),
                'volatility_range_std': float(np.std(volatility_ranges))
            })
            
            # Check for regime differentiation
            if np.max(return_ranges) < 0.001:  # Very similar returns
                validation['warnings'].append("Regimes have very similar return profiles")
            
            if np.max(volatility_ranges) < 0.001:  # Very similar volatilities
                validation['warnings'].append("Regimes have very similar volatility profiles")
        
        except Exception as e:
            validation['errors'].append(f"Cross-regime validation error: {str(e)}")
        
        return validation
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0
    
    def get_validation_history(self) -> List[Dict]:
        """Get validation history."""
        return self.validation_history
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=== Risk Parameter Validation Report ===\n")
        
        # Overall status
        status = "PASSED" if validation_results['overall_valid'] else "FAILED"
        report.append(f"Overall Status: {status}\n")
        
        # Summary statistics
        n_regimes = len(validation_results.get('regime_validations', {}))
        n_errors = len(validation_results.get('errors', []))
        n_warnings = len(validation_results.get('warnings', []))
        
        report.append(f"Number of Regimes: {n_regimes}")
        report.append(f"Total Errors: {n_errors}")
        report.append(f"Total Warnings: {n_warnings}\n")
        
        # Regime-specific results
        if validation_results.get('regime_validations'):
            report.append("=== Regime-Specific Validation ===")
            for regime_id, regime_val in validation_results['regime_validations'].items():
                status = "PASSED" if regime_val['valid'] else "FAILED"
                report.append(f"Regime {regime_id}: {status}")
                
                if regime_val['errors']:
                    report.append(f"  Errors: {', '.join(regime_val['errors'])}")
                
                if regime_val['warnings']:
                    report.append(f"  Warnings: {', '.join(regime_val['warnings'])}")
            
            report.append("")
        
        # Cross-regime checks
        if validation_results.get('cross_regime_checks'):
            cross_checks = validation_results['cross_regime_checks']
            report.append("=== Cross-Regime Consistency ===")
            
            if cross_checks.get('metrics'):
                metrics = cross_checks['metrics']
                if 'total_probability' in metrics:
                    report.append(f"Total Probability: {metrics['total_probability']:.4f}")
                if 'max_return_range' in metrics:
                    report.append(f"Max Return Range: {metrics['max_return_range']:.4f}")
                if 'max_volatility_range' in metrics:
                    report.append(f"Max Volatility Range: {metrics['max_volatility_range']:.4f}")
            
            if cross_checks['errors']:
                report.append(f"Cross-Regime Errors: {', '.join(cross_checks['errors'])}")
            
            if cross_checks['warnings']:
                report.append(f"Cross-Regime Warnings: {', '.join(cross_checks['warnings'])}")
        
        return "\n".join(report)


class RiskEstimator(RiskEstimatorInterface):
    """Main risk estimation class implementing the RiskEstimatorInterface."""
    
    def __init__(self, covariance_method: str = "ledoit_wolf", 
                 return_method: str = "sample_mean",
                 shrinkage_target: str = "identity"):
        """Initialize risk estimator.
        
        Args:
            covariance_method: Method for covariance estimation
            return_method: Method for return estimation
            shrinkage_target: Target for shrinkage estimation
        """
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize estimators
        self.covariance_estimator = CovarianceEstimator(covariance_method, shrinkage_target)
        self.return_estimator = ReturnEstimator(return_method)
        self.validator = RiskValidator()
        
        # Initialize enhanced components
        self.parameter_validator = RiskParameterValidator()
        self.parameter_storage = RiskParameterStorage()
        
        # State
        self.regime_parameters = {}
        self.estimation_metadata = {}
        self.validation_results = {}
        
        self.logger.info(f"RiskEstimator initialized with covariance_method={covariance_method}, return_method={return_method}")
    
    def estimate_regime_covariance(self, returns: pd.DataFrame, regime_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Estimate covariance matrices for each regime."""
        self.logger.info("Estimating regime-specific covariance matrices")
        
        unique_regimes = np.unique(regime_labels[regime_labels != -1])
        regime_covariances = {}
        
        for regime_id in unique_regimes:
            try:
                cov_result = self.covariance_estimator.estimate_regime_covariance(
                    returns, regime_labels, regime_id
                )
                
                regime_covariances[regime_id] = cov_result['covariance_matrix']
                
                # Store metadata
                if regime_id not in self.estimation_metadata:
                    self.estimation_metadata[regime_id] = {}
                self.estimation_metadata[regime_id]['covariance'] = cov_result
                
            except Exception as e:
                self.logger.error(f"Failed to estimate covariance for regime {regime_id}: {str(e)}")
                # Use identity matrix as fallback
                n_assets = returns.shape[1]
                avg_variance = returns.var().mean()
                regime_covariances[regime_id] = avg_variance * np.eye(n_assets)
        
        return regime_covariances
    
    def estimate_regime_returns(self, returns: pd.DataFrame, regime_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Estimate expected returns for each regime."""
        self.logger.info("Estimating regime-specific expected returns")
        
        unique_regimes = np.unique(regime_labels[regime_labels != -1])
        regime_returns = {}
        
        for regime_id in unique_regimes:
            try:
                return_result = self.return_estimator.estimate_regime_returns(
                    returns, regime_labels, regime_id
                )
                
                regime_returns[regime_id] = return_result['mean_returns']
                
                # Store metadata
                if regime_id not in self.estimation_metadata:
                    self.estimation_metadata[regime_id] = {}
                self.estimation_metadata[regime_id]['returns'] = return_result
                
            except Exception as e:
                self.logger.error(f"Failed to estimate returns for regime {regime_id}: {str(e)}")
                # Use zero returns as fallback
                regime_returns[regime_id] = np.zeros(returns.shape[1])
        
        return regime_returns
    
    def apply_shrinkage(self, sample_cov: np.ndarray, shrinkage_target: str = 'identity') -> np.ndarray:
        """Apply shrinkage estimation to covariance matrix."""
        # Use Ledoit-Wolf to determine optimal shrinkage intensity
        try:
            # Create dummy returns data to use LedoitWolf (this is a limitation of the interface)
            # In practice, this method should be called with the original returns data
            self.logger.warning("apply_shrinkage called without returns data, using default shrinkage")
            
            # Apply default shrinkage
            shrinkage_intensity = 0.2
            return self.covariance_estimator.apply_custom_shrinkage(
                sample_cov, shrinkage_intensity, shrinkage_target
            )
        
        except Exception as e:
            self.logger.error(f"Error applying shrinkage: {str(e)}")
            return sample_cov
    
    def validate_covariance(self, cov_matrix: np.ndarray) -> bool:
        """Validate covariance matrix properties."""
        return self.covariance_estimator.validate_covariance_matrix(cov_matrix)
    
    def estimate_all_regime_parameters(self, returns: pd.DataFrame, regime_labels: np.ndarray) -> Dict[int, RegimeParameters]:
        """Estimate all parameters for all regimes.
        
        Args:
            returns: DataFrame with asset returns
            regime_labels: Array of regime labels
            
        Returns:
            Dictionary mapping regime IDs to RegimeParameters objects
        """
        self.logger.info("Estimating all regime parameters")
        
        # Estimate covariances and returns
        regime_covariances = self.estimate_regime_covariance(returns, regime_labels)
        regime_returns = self.estimate_regime_returns(returns, regime_labels)
        
        # Create RegimeParameters objects
        regime_parameters = {}
        
        for regime_id in regime_covariances.keys():
            # Calculate regime probability
            regime_mask = regime_labels == regime_id
            regime_probability = np.mean(regime_mask)
            
            # Get date range for this regime
            regime_dates = returns.index[regime_mask]
            start_date = regime_dates.min() if len(regime_dates) > 0 else returns.index[0]
            end_date = regime_dates.max() if len(regime_dates) > 0 else returns.index[-1]
            
            # Create RegimeParameters object
            regime_params = RegimeParameters(
                regime_id=regime_id,
                mean_returns=regime_returns[regime_id],
                covariance_matrix=regime_covariances[regime_id],
                regime_probability=regime_probability,
                start_date=start_date,
                end_date=end_date
            )
            
            # Validate parameters
            validation_results = self.validator.validate_regime_parameters(regime_params)
            
            if not validation_results['valid']:
                self.logger.warning(f"Validation failed for regime {regime_id}: {validation_results['errors']}")
            
            regime_parameters[regime_id] = regime_params
            
            # Store validation results
            if regime_id not in self.estimation_metadata:
                self.estimation_metadata[regime_id] = {}
            self.estimation_metadata[regime_id]['validation'] = validation_results
        
        # Comprehensive validation
        comprehensive_validation = self.parameter_validator.validate_parameter_consistency(regime_parameters)
        self.validation_results = comprehensive_validation
        
        if not comprehensive_validation['overall_valid']:
            self.logger.warning("Regime parameters failed comprehensive validation")
            for error in comprehensive_validation['errors']:
                self.logger.error(f"Validation error: {error}")
        
        self.regime_parameters = regime_parameters
        return regime_parameters
    
    def get_estimation_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed estimation metadata for all regimes."""
        return self.estimation_metadata
    
    def get_regime_parameters(self) -> Dict[int, RegimeParameters]:
        """Get estimated regime parameters."""
        return self.regime_parameters
    
    def calculate_portfolio_risk(self, weights: np.ndarray, regime_id: int) -> Dict[str, float]:
        """Calculate portfolio risk metrics for a given regime.
        
        Args:
            weights: Portfolio weights
            regime_id: Regime ID
            
        Returns:
            Dictionary with risk metrics
        """
        if regime_id not in self.regime_parameters:
            raise ValueError(f"No parameters available for regime {regime_id}")
        
        regime_params = self.regime_parameters[regime_id]
        
        # Portfolio variance
        portfolio_variance = weights.T @ regime_params.covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Portfolio expected return
        portfolio_return = weights.T @ regime_params.mean_returns
        
        # Sharpe ratio (assuming zero risk-free rate)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(portfolio_volatility),
            'portfolio_return': float(portfolio_return),
            'sharpe_ratio': float(sharpe_ratio),
            'regime_id': regime_id
        }
    
    def save_parameters(self, metadata: Optional[Dict] = None) -> bool:
        """Save current regime parameters to storage.
        
        Args:
            metadata: Optional metadata to save with parameters
            
        Returns:
            True if successful
        """
        if not self.regime_parameters:
            self.logger.warning("No regime parameters to save")
            return False
        
        # Include estimation metadata
        full_metadata = {
            'estimation_metadata': self.estimation_metadata,
            'validation_results': self.validation_results,
            'covariance_method': self.covariance_estimator.method,
            'return_method': self.return_estimator.method,
            'shrinkage_target': self.covariance_estimator.shrinkage_target
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        return self.parameter_storage.save_regime_parameters(self.regime_parameters, full_metadata)
    
    def load_parameters(self) -> bool:
        """Load regime parameters from storage.
        
        Returns:
            True if successful
        """
        loaded_params = self.parameter_storage.load_regime_parameters()
        
        if loaded_params is not None:
            self.regime_parameters = loaded_params
            
            # Load metadata if available
            metadata = self.parameter_storage.get_storage_metadata()
            if metadata:
                self.estimation_metadata = metadata.get('estimation_metadata', {})
                self.validation_results = metadata.get('validation_results', {})
            
            self.logger.info(f"Loaded {len(loaded_params)} regime parameters from storage")
            return True
        
        return False
    
    def export_parameters(self, output_dir: str = "exports") -> Dict[str, str]:
        """Export regime parameters to CSV files.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary mapping file types to file paths
        """
        if not self.regime_parameters:
            self.logger.warning("No regime parameters to export")
            return {}
        
        return self.parameter_storage.export_parameters_to_csv(self.regime_parameters, output_dir)
    
    def validate_current_parameters(self) -> Dict[str, Any]:
        """Validate current regime parameters.
        
        Returns:
            Validation results dictionary
        """
        if not self.regime_parameters:
            return {'overall_valid': False, 'errors': ['No parameters to validate']}
        
        validation_results = self.parameter_validator.validate_parameter_consistency(self.regime_parameters)
        self.validation_results = validation_results
        
        return validation_results
    
    def generate_validation_report(self) -> str:
        """Generate a validation report for current parameters.
        
        Returns:
            Human-readable validation report
        """
        if not self.validation_results:
            self.validate_current_parameters()
        
        return self.parameter_validator.generate_validation_report(self.validation_results)
    
    def update_parameter(self, regime_id: int, parameter_type: str, new_value: np.ndarray) -> bool:
        """Update a specific parameter for a regime.
        
        Args:
            regime_id: Regime ID to update
            parameter_type: Type of parameter ("mean_returns" or "covariance_matrix")
            new_value: New parameter value
            
        Returns:
            True if successful
        """
        if regime_id not in self.regime_parameters:
            self.logger.error(f"Regime {regime_id} not found")
            return False
        
        try:
            regime_params = self.regime_parameters[regime_id]
            
            if parameter_type == "mean_returns":
                if len(new_value) != len(regime_params.mean_returns):
                    self.logger.error("Dimension mismatch for mean returns update")
                    return False
                
                # Create new RegimeParameters object with updated returns
                updated_params = RegimeParameters(
                    regime_id=regime_params.regime_id,
                    mean_returns=new_value,
                    covariance_matrix=regime_params.covariance_matrix,
                    regime_probability=regime_params.regime_probability,
                    start_date=regime_params.start_date,
                    end_date=regime_params.end_date
                )
            
            elif parameter_type == "covariance_matrix":
                expected_shape = regime_params.covariance_matrix.shape
                if new_value.shape != expected_shape:
                    self.logger.error(f"Dimension mismatch for covariance update: expected {expected_shape}, got {new_value.shape}")
                    return False
                
                # Validate new covariance matrix
                if not self.covariance_estimator.validate_covariance_matrix(new_value):
                    self.logger.error("New covariance matrix failed validation")
                    return False
                
                # Create new RegimeParameters object with updated covariance
                updated_params = RegimeParameters(
                    regime_id=regime_params.regime_id,
                    mean_returns=regime_params.mean_returns,
                    covariance_matrix=new_value,
                    regime_probability=regime_params.regime_probability,
                    start_date=regime_params.start_date,
                    end_date=regime_params.end_date
                )
            
            else:
                self.logger.error(f"Unknown parameter type: {parameter_type}")
                return False
            
            # Update the parameter
            self.regime_parameters[regime_id] = updated_params
            
            # Re-validate parameters
            self.validate_current_parameters()
            
            self.logger.info(f"Updated {parameter_type} for regime {regime_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to update parameter: {str(e)}")
            return False
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get a summary of current parameters.
        
        Returns:
            Summary dictionary with key statistics
        """
        if not self.regime_parameters:
            return {'error': 'No parameters available'}
        
        summary = {
            'n_regimes': len(self.regime_parameters),
            'regime_ids': list(self.regime_parameters.keys()),
            'total_probability': sum(p.regime_probability for p in self.regime_parameters.values()),
            'parameter_details': {}
        }
        
        for regime_id, params in self.regime_parameters.items():
            n_assets = len(params.mean_returns)
            avg_return = np.mean(params.mean_returns)
            avg_volatility = np.mean(np.sqrt(np.diag(params.covariance_matrix)))
            
            summary['parameter_details'][regime_id] = {
                'n_assets': n_assets,
                'regime_probability': params.regime_probability,
                'avg_return': float(avg_return),
                'avg_volatility': float(avg_volatility),
                'return_range': [float(np.min(params.mean_returns)), float(np.max(params.mean_returns))],
                'volatility_range': [
                    float(np.min(np.sqrt(np.diag(params.covariance_matrix)))),
                    float(np.max(np.sqrt(np.diag(params.covariance_matrix))))
                ],
                'start_date': str(params.start_date.date()) if hasattr(params.start_date, 'date') else str(params.start_date),
                'end_date': str(params.end_date.date()) if hasattr(params.end_date, 'date') else str(params.end_date)
            }
        
        # Add validation status if available
        if self.validation_results:
            summary['validation_status'] = {
                'overall_valid': self.validation_results.get('overall_valid', False),
                'n_errors': len(self.validation_results.get('errors', [])),
                'n_warnings': len(self.validation_results.get('warnings', []))
            }
        
        return summary