"""
Multi-Objective Portfolio Optimizer - Phase 1 & 2 Implementation

This module implements multi-objective optimization that balances:
- Expected returns (regime-weighted)
- Worst-case risk (robust optimization)
- Dynamic risk aversion based on market conditions
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
from robust_optimizer import ConstraintManager
from logging_config import get_logger


class MultiObjectiveOptimizer:
    """Multi-objective portfolio optimizer with dynamic risk aversion."""
    
    def __init__(self, solver: str = "cvxpy"):
        """Initialize multi-objective optimizer."""
        self.solver = solver
        self.logger = get_logger(__name__)
        self.logger.info("Multi-objective optimizer initialized")
    
    def optimize_multi_objective(self, 
                               regime_covariances: Dict[int, np.ndarray],
                               regime_returns: Dict[int, np.ndarray],
                               regime_probabilities: Dict[int, float],
                               constraints: ConstraintManager,
                               current_volatility: float = 0.15,
                               base_risk_aversion: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Multi-objective optimization: maximize expected return while minimizing worst-case risk.
        
        Objective: max E[R] - λ(vol) * max_r(w^T * Σ_r * w)
        where λ(vol) is dynamic risk aversion based on market volatility
        
        Args:
            regime_covariances: Covariance matrices for each regime
            regime_returns: Expected returns for each regime
            regime_probabilities: Probability of each regime
            constraints: Portfolio constraints
            current_volatility: Current market volatility for dynamic risk aversion
            base_risk_aversion: Base risk aversion parameter
            
        Returns:
            Tuple of (optimal_weights, optimization_info)
        """
        
        self.logger.info("Starting multi-objective optimization...")
        
        # Calculate dynamic risk aversion based on market conditions
        dynamic_risk_aversion = self._calculate_dynamic_risk_aversion(
            current_volatility, base_risk_aversion
        )
        
        # Calculate regime-weighted expected returns
        expected_returns = self._calculate_expected_returns(
            regime_returns, regime_probabilities
        )
        
        # Solve multi-objective optimization
        optimal_weights, opt_info = self._solve_multi_objective_cvxpy(
            regime_covariances, expected_returns, constraints, dynamic_risk_aversion
        )
        
        # Add optimization metadata
        opt_info.update({
            'dynamic_risk_aversion': dynamic_risk_aversion,
            'current_volatility': current_volatility,
            'expected_portfolio_return': np.dot(optimal_weights, expected_returns) if optimal_weights is not None else 0,
            'regime_probabilities': regime_probabilities
        })
        
        self.logger.info(f"Multi-objective optimization completed. "
                        f"Risk aversion: {dynamic_risk_aversion:.2f}, "
                        f"Expected return: {opt_info.get('expected_portfolio_return', 0)*252:.1%}")
        
        return optimal_weights, opt_info
    
    def _calculate_dynamic_risk_aversion(self, current_volatility: float, 
                                       base_risk_aversion: float) -> float:
        """Calculate dynamic risk aversion based on market conditions."""
        
        # Volatility regimes
        low_vol_threshold = 0.12   # 12% annualized
        high_vol_threshold = 0.25  # 25% annualized
        
        if current_volatility > high_vol_threshold:
            # High volatility: be more risk averse
            multiplier = 2.0
            regime_type = "High Volatility"
        elif current_volatility < low_vol_threshold:
            # Low volatility: be less risk averse (take more risk)
            multiplier = 0.5
            regime_type = "Low Volatility"
        else:
            # Normal volatility: use base risk aversion
            multiplier = 1.0
            regime_type = "Normal Volatility"
        
        dynamic_risk_aversion = base_risk_aversion * multiplier
        
        self.logger.info(f"Dynamic risk aversion: {regime_type} regime "
                        f"(vol: {current_volatility:.1%}) -> λ = {dynamic_risk_aversion:.2f}")
        
        return dynamic_risk_aversion
    
    def _calculate_expected_returns(self, regime_returns: Dict[int, np.ndarray],
                                  regime_probabilities: Dict[int, float]) -> np.ndarray:
        """Calculate regime-weighted expected returns."""
        
        if not regime_returns:
            raise ValueError("No regime returns provided")
        
        # Get dimensions from first regime
        first_regime = list(regime_returns.keys())[0]
        n_assets = len(regime_returns[first_regime])
        
        # Calculate weighted average returns
        expected_returns = np.zeros(n_assets)
        total_prob = 0.0
        
        for regime_id, returns in regime_returns.items():
            prob = regime_probabilities.get(regime_id, 0.0)
            expected_returns += prob * returns
            total_prob += prob
        
        # Normalize if probabilities don't sum to 1
        if total_prob > 0:
            expected_returns = expected_returns / total_prob
        
        self.logger.info(f"Expected returns calculated: "
                        f"mean = {np.mean(expected_returns)*252:.1%}, "
                        f"std = {np.std(expected_returns)*252:.1%}")
        
        return expected_returns
    
    def _solve_multi_objective_cvxpy(self, regime_covariances: Dict[int, np.ndarray],
                                   expected_returns: np.ndarray,
                                   constraints: ConstraintManager,
                                   risk_aversion: float) -> Tuple[np.ndarray, Dict]:
        """Solve multi-objective optimization using CVXPY."""
        
        n_assets = len(expected_returns)
        
        # Decision variables
        w = cp.Variable(n_assets)  # Portfolio weights
        t = cp.Variable()          # Worst-case variance
        
        # Objective: maximize expected return - risk_aversion * worst_case_variance
        # Convert to minimization: minimize -expected_return + risk_aversion * worst_case_variance
        objective = cp.Minimize(-expected_returns.T @ w + risk_aversion * t)
        
        # Constraints
        cvx_constraints = []
        
        # Worst-case variance constraints (one for each regime)
        for regime_id, cov_matrix in regime_covariances.items():
            cvx_constraints.append(cp.quad_form(w, cov_matrix) <= t)
        
        # Portfolio constraints from constraint manager
        portfolio_constraints = self._convert_constraints_to_cvxpy(w, constraints, n_assets)
        cvx_constraints.extend(portfolio_constraints)
        
        # Solve optimization problem
        problem = cp.Problem(objective, cvx_constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                worst_case_variance = t.value
                
                # Ensure weights are valid and normalized
                if optimal_weights is not None:
                    optimal_weights = np.maximum(optimal_weights, 0)  # Ensure non-negative
                    if np.sum(optimal_weights) > 0:
                        optimal_weights = optimal_weights / np.sum(optimal_weights)
                    else:
                        optimal_weights = None
                
                opt_info = {
                    'status': 'optimal',
                    'objective_value': problem.value,
                    'worst_case_variance': worst_case_variance,
                    'solver_time': problem.solver_stats.solve_time if hasattr(problem.solver_stats, 'solve_time') else 0
                }
                
                return optimal_weights, opt_info
            
            else:
                self.logger.warning(f"Optimization failed with status: {problem.status}")
                return None, {'status': problem.status, 'objective_value': np.inf}
                
        except Exception as e:
            self.logger.error(f"Optimization solver error: {str(e)}")
            return None, {'status': 'error', 'error': str(e)}
    
    def _convert_constraints_to_cvxpy(self, w, constraints: ConstraintManager, 
                                    n_assets: int) -> List:
        """Convert constraint manager constraints to CVXPY format."""
        
        cvx_constraints = []
        constraint_dict = constraints.get_constraints()
        
        # Budget constraint (sum of weights = target)
        if 'budget' in constraint_dict:
            target = constraint_dict['budget'].get('target', 1.0)
            cvx_constraints.append(cp.sum(w) == target)
        
        # Long-only constraint
        if 'long_only' in constraint_dict:
            cvx_constraints.append(w >= 0)
        
        # Box constraints (individual weight bounds)
        if 'box' in constraint_dict:
            min_weight = constraint_dict['box'].get('min_weight', 0.0)
            max_weight = constraint_dict['box'].get('max_weight', 1.0)
            
            cvx_constraints.append(w >= min_weight)
            cvx_constraints.append(w <= max_weight)
        
        return cvx_constraints
    
    def calculate_portfolio_metrics(self, weights: np.ndarray,
                                  regime_covariances: Dict[int, np.ndarray],
                                  regime_returns: Dict[int, np.ndarray],
                                  regime_probabilities: Dict[int, float]) -> Dict:
        """Calculate portfolio risk and return metrics."""
        
        if weights is None:
            return {}
        
        # Expected return
        expected_returns = self._calculate_expected_returns(regime_returns, regime_probabilities)
        portfolio_return = np.dot(weights, expected_returns)
        
        # Worst-case variance
        worst_case_var = 0
        for regime_id, cov_matrix in regime_covariances.items():
            regime_var = np.dot(weights, np.dot(cov_matrix, weights))
            worst_case_var = max(worst_case_var, regime_var)
        
        # Average variance (probability-weighted)
        avg_var = 0
        for regime_id, cov_matrix in regime_covariances.items():
            prob = regime_probabilities.get(regime_id, 0)
            regime_var = np.dot(weights, np.dot(cov_matrix, weights))
            avg_var += prob * regime_var
        
        return {
            'expected_return_daily': portfolio_return,
            'expected_return_annual': portfolio_return * 252,
            'worst_case_volatility': np.sqrt(worst_case_var * 252),
            'average_volatility': np.sqrt(avg_var * 252),
            'worst_case_variance': worst_case_var,
            'average_variance': avg_var
        }