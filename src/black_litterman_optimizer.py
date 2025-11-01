"""
Black-Litterman Portfolio Optimization - Phase 3 Implementation

This module implements the Black-Litterman model for incorporating
market views and regime-based insights into portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

from multi_objective_optimizer import MultiObjectiveOptimizer
from robust_optimizer import ConstraintManager
from logging_config import get_logger


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimizer with regime-based views."""
    
    def __init__(self, risk_aversion: float = 3.0):
        """Initialize Black-Litterman optimizer."""
        self.risk_aversion = risk_aversion
        self.logger = get_logger(__name__)
        
        # Market parameters
        self.market_cap_weights = None
        self.implied_returns = None
        self.views = []
        self.view_uncertainties = []
        
        self.logger.info("Black-Litterman optimizer initialized")
    
    def set_market_equilibrium(self, returns: pd.DataFrame, 
                             market_cap_weights: Optional[np.ndarray] = None) -> None:
        """Set market equilibrium parameters."""
        
        if market_cap_weights is None:
            # Equal weights as proxy for market cap weights
            market_cap_weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        self.market_cap_weights = market_cap_weights
        
        # Calculate historical covariance
        cov_matrix = returns.cov().values * 252  # Annualized
        
        # Implied equilibrium returns (reverse optimization)
        self.implied_returns = self.risk_aversion * np.dot(cov_matrix, market_cap_weights)
        
        self.logger.info(f"Market equilibrium set with {len(returns.columns)} assets")
        self.logger.info(f"Implied returns range: {self.implied_returns.min():.1%} to {self.implied_returns.max():.1%}")
    
    def add_regime_views(self, regime_returns: Dict[int, np.ndarray],
                        regime_probabilities: Dict[int, float],
                        current_regime: int,
                        asset_names: List[str]) -> None:
        """Add views based on regime analysis."""
        
        self.logger.info("Adding regime-based views to Black-Litterman model...")
        
        # Clear existing views
        self.views = []
        self.view_uncertainties = []
        
        # View 1: Current regime expected returns
        if current_regime in regime_returns:
            current_regime_returns = regime_returns[current_regime] * 252  # Annualized
            
            # Add views for assets with strong regime signals
            for i, asset in enumerate(asset_names):
                regime_return = current_regime_returns[i]
                implied_return = self.implied_returns[i]
                
                # Add view if regime return significantly differs from implied
                if abs(regime_return - implied_return) > 0.02:  # 2% threshold
                    view_vector = np.zeros(len(asset_names))
                    view_vector[i] = 1.0
                    
                    self.views.append(view_vector)
                    
                    # View uncertainty based on regime probability
                    regime_prob = regime_probabilities.get(current_regime, 0.33)
                    uncertainty = 0.05 / regime_prob  # Higher uncertainty for less probable regimes
                    self.view_uncertainties.append(uncertainty)
                    
                    self.logger.info(f"Added view for {asset}: {regime_return:.1%} "
                                   f"(vs implied {implied_return:.1%})")
        
        # View 2: Relative performance views
        self._add_relative_performance_views(regime_returns, regime_probabilities, asset_names)
        
        # View 3: Volatility-based views
        self._add_volatility_views(regime_returns, asset_names)
    
    def _add_relative_performance_views(self, regime_returns: Dict[int, np.ndarray],
                                      regime_probabilities: Dict[int, float],
                                      asset_names: List[str]) -> None:
        """Add relative performance views between asset classes."""
        
        try:
            # Define asset classes
            equity_indices = [i for i, name in enumerate(asset_names) 
                            if name in ['SPY', 'QQQ', 'IWM', 'VTI']]
            bond_indices = [i for i, name in enumerate(asset_names) 
                          if name in ['AGG', 'TLT', 'SHY', 'TIP']]
            
            if len(equity_indices) >= 2 and len(bond_indices) >= 2:
                # Calculate regime-weighted expected returns
                expected_returns = np.zeros(len(asset_names))
                for regime_id, returns in regime_returns.items():
                    prob = regime_probabilities.get(regime_id, 0)
                    expected_returns += prob * returns * 252
                
                # Equity vs Bond view
                avg_equity_return = np.mean(expected_returns[equity_indices])
                avg_bond_return = np.mean(expected_returns[bond_indices])
                
                if abs(avg_equity_return - avg_bond_return) > 0.03:  # 3% threshold
                    # Create relative view: Equity - Bonds
                    view_vector = np.zeros(len(asset_names))
                    for i in equity_indices:
                        view_vector[i] = 1.0 / len(equity_indices)
                    for i in bond_indices:
                        view_vector[i] = -1.0 / len(bond_indices)
                    
                    self.views.append(view_vector)
                    self.view_uncertainties.append(0.04)  # 4% uncertainty
                    
                    self.logger.info(f"Added equity vs bond view: "
                                   f"{avg_equity_return - avg_bond_return:.1%} outperformance")
        
        except Exception as e:
            self.logger.warning(f"Failed to add relative performance views: {str(e)}")
    
    def _add_volatility_views(self, regime_returns: Dict[int, np.ndarray],
                            asset_names: List[str]) -> None:
        """Add views based on volatility expectations."""
        
        try:
            # Low volatility assets should outperform in high volatility regimes
            volatility_assets = [i for i, name in enumerate(asset_names) 
                               if name in ['SHY', 'AGG', 'TIP']]  # Low vol assets
            
            if len(volatility_assets) >= 2:
                # Simple view: low volatility assets have positive expected returns
                view_vector = np.zeros(len(asset_names))
                for i in volatility_assets:
                    view_vector[i] = 1.0 / len(volatility_assets)
                
                self.views.append(view_vector)
                self.view_uncertainties.append(0.03)  # 3% uncertainty
                
                self.logger.info("Added low volatility preference view")
        
        except Exception as e:
            self.logger.warning(f"Failed to add volatility views: {str(e)}")
    
    def optimize_black_litterman(self, returns: pd.DataFrame,
                                regime_covariances: Dict[int, np.ndarray],
                                regime_returns: Dict[int, np.ndarray],
                                regime_probabilities: Dict[int, float],
                                current_regime: int,
                                constraints: ConstraintManager,
                                asset_names: List[str]) -> Tuple[np.ndarray, Dict]:
        """Optimize portfolio using Black-Litterman model."""
        
        self.logger.info("Starting Black-Litterman optimization...")
        
        # Set market equilibrium
        self.set_market_equilibrium(returns)
        
        # Add regime-based views
        self.add_regime_views(regime_returns, regime_probabilities, current_regime, asset_names)
        
        if not self.views:
            self.logger.warning("No views added, using market equilibrium")
            return self.market_cap_weights, {'status': 'no_views', 'method': 'market_equilibrium'}
        
        # Calculate Black-Litterman expected returns
        bl_returns = self._calculate_bl_returns(returns)
        
        # Use regime-weighted covariance
        bl_covariance = self._calculate_bl_covariance(regime_covariances, regime_probabilities)
        
        # Optimize portfolio
        optimal_weights = self._optimize_bl_portfolio(bl_returns, bl_covariance, constraints)
        
        # Calculate performance metrics
        bl_info = {
            'status': 'optimal',
            'method': 'black_litterman',
            'n_views': len(self.views),
            'bl_expected_return': np.dot(optimal_weights, bl_returns),
            'implied_expected_return': np.dot(optimal_weights, self.implied_returns),
            'view_impact': np.dot(optimal_weights, bl_returns) - np.dot(optimal_weights, self.implied_returns)
        }
        
        self.logger.info(f"Black-Litterman optimization completed with {len(self.views)} views")
        self.logger.info(f"Expected return: {bl_info['bl_expected_return']:.1%}")
        self.logger.info(f"View impact: {bl_info['view_impact']:.1%}")
        
        return optimal_weights, bl_info
    
    def _calculate_bl_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate Black-Litterman expected returns."""
        
        # Historical covariance
        cov_matrix = returns.cov().values * 252
        
        # Convert views to matrices
        P = np.array(self.views)  # Picking matrix
        Q = np.array([np.dot(view, self.implied_returns) for view in self.views])  # View returns
        
        # View uncertainty matrix (diagonal)
        omega = np.diag(np.array(self.view_uncertainties) ** 2)
        
        # Tau parameter (scales the uncertainty of the prior)
        tau = 1.0 / len(returns)
        
        # Black-Litterman formula
        try:
            # M1 = inv(tau * Sigma)
            M1 = linalg.inv(tau * cov_matrix)
            
            # M2 = P' * inv(Omega) * P
            M2 = np.dot(P.T, np.dot(linalg.inv(omega), P))
            
            # M3 = inv(tau * Sigma) * Pi + P' * inv(Omega) * Q
            M3 = np.dot(M1, self.implied_returns) + np.dot(P.T, np.dot(linalg.inv(omega), Q))
            
            # New expected returns
            bl_returns = np.dot(linalg.inv(M1 + M2), M3)
            
            return bl_returns
            
        except Exception as e:
            self.logger.warning(f"Black-Litterman calculation failed: {str(e)}")
            return self.implied_returns
    
    def _calculate_bl_covariance(self, regime_covariances: Dict[int, np.ndarray],
                               regime_probabilities: Dict[int, float]) -> np.ndarray:
        """Calculate regime-weighted covariance matrix."""
        
        # Probability-weighted covariance
        n_assets = len(list(regime_covariances.values())[0])
        weighted_cov = np.zeros((n_assets, n_assets))
        
        for regime_id, cov_matrix in regime_covariances.items():
            prob = regime_probabilities.get(regime_id, 0)
            weighted_cov += prob * cov_matrix * 252  # Annualized
        
        return weighted_cov
    
    def _optimize_bl_portfolio(self, bl_returns: np.ndarray,
                             bl_covariance: np.ndarray,
                             constraints: ConstraintManager) -> np.ndarray:
        """Optimize portfolio using Black-Litterman inputs."""
        
        try:
            # Mean-variance optimization with Black-Litterman inputs
            inv_cov = linalg.inv(bl_covariance + 1e-6 * np.eye(len(bl_covariance)))
            
            # Unconstrained optimal weights
            unconstrained_weights = np.dot(inv_cov, bl_returns) / self.risk_aversion
            
            # Apply constraints (simplified)
            constraint_dict = constraints.get_constraints()
            
            # Normalize to satisfy budget constraint
            if 'budget' in constraint_dict:
                target = constraint_dict['budget'].get('target', 1.0)
                unconstrained_weights = unconstrained_weights / np.sum(unconstrained_weights) * target
            
            # Apply box constraints
            if 'box' in constraint_dict:
                min_weight = constraint_dict['box'].get('min_weight', 0.0)
                max_weight = constraint_dict['box'].get('max_weight', 1.0)
                
                unconstrained_weights = np.clip(unconstrained_weights, min_weight, max_weight)
                
                # Renormalize after clipping
                if np.sum(unconstrained_weights) > 0:
                    unconstrained_weights = unconstrained_weights / np.sum(unconstrained_weights)
            
            # Ensure long-only if required
            if 'long_only' in constraint_dict:
                unconstrained_weights = np.maximum(unconstrained_weights, 0)
                if np.sum(unconstrained_weights) > 0:
                    unconstrained_weights = unconstrained_weights / np.sum(unconstrained_weights)
            
            return unconstrained_weights
            
        except Exception as e:
            self.logger.warning(f"Black-Litterman portfolio optimization failed: {str(e)}")
            return self.market_cap_weights
    
    def get_view_summary(self) -> pd.DataFrame:
        """Get summary of current views."""
        
        if not self.views:
            return pd.DataFrame()
        
        view_data = []
        for i, (view, uncertainty) in enumerate(zip(self.views, self.view_uncertainties)):
            view_data.append({
                'view_id': i,
                'view_vector': view,
                'uncertainty': uncertainty,
                'view_strength': 1.0 / uncertainty
            })
        
        return pd.DataFrame(view_data)