"""
Base interfaces and abstract classes for the robust portfolio optimization system.

This module defines the core interfaces that all components must implement,
ensuring consistent APIs and enabling modular design.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RegimeParameters:
    """Data structure for regime-specific statistical parameters."""
    regime_id: int
    mean_returns: np.ndarray
    covariance_matrix: np.ndarray
    regime_probability: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp


@dataclass
class OptimizationResult:
    """Data structure for optimization results."""
    weights: np.ndarray
    objective_value: float
    solver_status: str
    computation_time: float
    constraints_satisfied: bool


@dataclass
class BacktestResult:
    """Data structure for backtesting results."""
    portfolio_returns: pd.Series
    portfolio_weights: pd.DataFrame
    performance_metrics: Dict[str, float]
    benchmark_comparison: pd.DataFrame
    regime_attribution: pd.DataFrame


class DataManagerInterface(ABC):
    """Interface for data acquisition and preprocessing."""
    
    @abstractmethod
    def download_asset_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download asset price data from external APIs."""
        pass
    
    @abstractmethod
    def download_macro_data(self, series_ids: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download macroeconomic data from external APIs."""
        pass
    
    @abstractmethod
    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute returns from price data."""
        pass
    
    @abstractmethod
    def create_regime_features(self, returns: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime detection."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality and completeness."""
        pass


class RegimeDetectorInterface(ABC):
    """Interface for market regime detection."""
    
    @abstractmethod
    def fit_regimes(self, features: pd.DataFrame, n_regimes: int = 3) -> np.ndarray:
        """Fit regime detection model and return regime labels."""
        pass
    
    @abstractmethod
    def predict_regime(self, features: pd.DataFrame) -> int:
        """Predict regime for new data."""
        pass
    
    @abstractmethod
    def get_regime_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistical summary of detected regimes."""
        pass
    
    @abstractmethod
    def validate_regimes(self, regime_labels: np.ndarray) -> bool:
        """Validate regime detection results."""
        pass


class RiskEstimatorInterface(ABC):
    """Interface for risk parameter estimation."""
    
    @abstractmethod
    def estimate_regime_covariance(self, returns: pd.DataFrame, regime_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Estimate covariance matrices for each regime."""
        pass
    
    @abstractmethod
    def estimate_regime_returns(self, returns: pd.DataFrame, regime_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Estimate expected returns for each regime."""
        pass
    
    @abstractmethod
    def apply_shrinkage(self, sample_cov: np.ndarray, shrinkage_target: str = 'identity') -> np.ndarray:
        """Apply shrinkage estimation to covariance matrix."""
        pass
    
    @abstractmethod
    def validate_covariance(self, cov_matrix: np.ndarray) -> bool:
        """Validate covariance matrix properties."""
        pass


class RobustOptimizerInterface(ABC):
    """Interface for robust portfolio optimization."""
    
    @abstractmethod
    def optimize_worst_case(self, regime_covariances: Dict[int, np.ndarray], constraints: Dict) -> np.ndarray:
        """Solve worst-case variance minimization problem."""
        pass
    
    @abstractmethod
    def optimize_cvar(self, regime_returns: Dict[int, np.ndarray], regime_probs: np.ndarray, alpha: float) -> np.ndarray:
        """Solve CVaR optimization problem."""
        pass
    
    @abstractmethod
    def add_constraint(self, constraint_type: str, parameters: Dict) -> None:
        """Add portfolio constraint to optimization problem."""
        pass
    
    @abstractmethod
    def solve_optimization(self) -> OptimizationResult:
        """Solve the configured optimization problem."""
        pass


class BacktestEngineInterface(ABC):
    """Interface for portfolio backtesting."""
    
    @abstractmethod
    def run_backtest(self, start_date: str, end_date: str, rebalance_freq: str) -> BacktestResult:
        """Run backtesting simulation."""
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        pass
    
    @abstractmethod
    def compare_strategies(self, strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compare performance across different strategies."""
        pass
    
    @abstractmethod
    def account_for_transaction_costs(self, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
        """Account for transaction costs in backtest."""
        pass


class VisualizationEngineInterface(ABC):
    """Interface for visualization and reporting."""
    
    @abstractmethod
    def plot_regime_detection(self, regime_labels: np.ndarray, dates: pd.DatetimeIndex) -> Any:
        """Create regime detection visualization."""
        pass
    
    @abstractmethod
    def create_allocation_heatmap(self, weights: pd.DataFrame) -> Any:
        """Create portfolio allocation heatmap."""
        pass
    
    @abstractmethod
    def plot_performance_comparison(self, strategy_returns: Dict[str, pd.Series]) -> Any:
        """Create performance comparison plots."""
        pass
    
    @abstractmethod
    def generate_summary_report(self, backtest_results: Dict) -> str:
        """Generate comprehensive summary report."""
        pass