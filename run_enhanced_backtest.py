#!/usr/bin/env python3
"""
Enhanced Train/Test Backtesting with Improved Regime Detection

This script demonstrates the improved system with:
- Enhanced regime detection (momentum, volatility, credit indicators)
- Multi-objective optimization (return vs risk)
- 4-5 regimes for better granularity
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from enhanced_regime_detector import EnhancedRegimeDetector
from train_test_backtest import TrainTestBacktester
from data_manager import DataManager
from risk_estimator import RiskEstimator
from robust_optimizer import RobustOptimizer, ConstraintManager
from logging_config import setup_logging
import logging
import pandas as pd
import numpy as np
from typing import Dict


class EnhancedTrainTestBacktester(TrainTestBacktester):
    """Enhanced backtester with improved regime detection and optimization."""
    
    def __init__(self):
        """Initialize enhanced backtester."""
        super().__init__()
        
        # Replace regime detector with enhanced version
        self.regime_detector = EnhancedRegimeDetector()
        
        self.logger.info("Enhanced TrainTestBacktester initialized")
    
    def _train_model(self, train_data: Dict) -> Dict:
        """Train model using enhanced regime detection."""
        
        # Get price data for enhanced features
        returns_data = train_data['returns']
        
        # Reconstruct prices from returns (approximate)
        initial_price = 100
        prices_data = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for col in returns_data.columns:
            prices_data[col] = initial_price * (1 + returns_data[col]).cumprod()
        
        # Enhanced regime detection with 3 regimes (due to data constraints)
        self.logger.info("Detecting market regimes with enhanced features...")
        regime_labels = self.regime_detector.fit_enhanced_regimes(
            returns_data, train_data['macro'], prices_data, n_regimes=3
        )
        
        # Map regime labels to daily frequency
        self.logger.info("Mapping regime labels to daily frequency...")
        daily_regime_labels = self._map_regimes_to_daily_frequency(
            regime_labels, train_data['features'], train_data['returns']
        )
        
        # Estimate regime-specific parameters
        self.logger.info("Estimating regime-specific risk parameters...")
        regime_covariances = self.risk_estimator.estimate_regime_covariance(
            train_data['returns'], daily_regime_labels
        )
        regime_returns = self.risk_estimator.estimate_regime_returns(
            train_data['returns'], daily_regime_labels
        )
        
        # Calculate regime probabilities
        unique_regimes, regime_counts = np.unique(daily_regime_labels, return_counts=True)
        regime_probs = regime_counts / len(daily_regime_labels)
        
        trained_model = {
            'regime_detector': self.regime_detector,
            'regime_covariances': regime_covariances,
            'regime_returns': regime_returns,
            'regime_probabilities': dict(zip(unique_regimes, regime_probs)),
            'asset_names': train_data['returns'].columns.tolist()
        }
        
        self.logger.info(f"Enhanced model trained with {len(unique_regimes)} regimes")
        for regime_id, prob in trained_model['regime_probabilities'].items():
            self.logger.info(f"  Regime {regime_id}: {prob:.1%} of training period")
        
        return trained_model
    
    def _get_optimal_weights_multi_objective(self, trained_model: Dict, 
                                           current_regime: int,
                                           risk_aversion: float = 1.0) -> np.ndarray:
        """Get optimal weights using multi-objective optimization."""
        
        try:
            regime_covariances = trained_model['regime_covariances']
            regime_returns = trained_model['regime_returns']
            regime_probs = list(trained_model['regime_probabilities'].values())
            
            # Dynamic risk aversion based on current market volatility
            # (This would be calculated from recent market data in practice)
            base_risk_aversion = risk_aversion
            
            # For demonstration, vary risk aversion by regime
            if current_regime == 0:  # Assume regime 0 is high volatility
                dynamic_risk_aversion = base_risk_aversion * 1.5
            elif current_regime == 3:  # Assume regime 3 is low volatility
                dynamic_risk_aversion = base_risk_aversion * 0.7
            else:
                dynamic_risk_aversion = base_risk_aversion
            
            # Calculate regime-weighted expected returns
            expected_returns = np.zeros(len(trained_model['asset_names']))
            for i, (regime_id, prob) in enumerate(trained_model['regime_probabilities'].items()):
                if regime_id in regime_returns:
                    expected_returns += prob * regime_returns[regime_id]
            
            # Set up constraints
            n_assets = len(trained_model['asset_names'])
            max_individual_weight = min(0.20, 8.0 / n_assets)  # Slightly higher concentration allowed
            
            constraint_manager = ConstraintManager()
            constraint_manager.add_constraint("budget", {"target": 1.0})
            constraint_manager.add_constraint("long_only", {})
            constraint_manager.add_constraint("box", {
                "min_weight": 0.005,  # Slightly higher minimum
                "max_weight": max_individual_weight
            })
            
            # Use multi-objective optimization (simplified version)
            # In practice, this would be a more sophisticated implementation
            result = self.robust_optimizer.worst_case_optimizer.optimize(
                regime_covariances, constraint_manager
            )
            
            optimal_weights = result.weights
            
            # Apply return tilt (simple approach)
            if result.solver_status == "optimal" and optimal_weights is not None:
                # Tilt towards higher expected return assets
                return_tilt = expected_returns / (np.abs(expected_returns).max() + 1e-6)
                tilt_strength = 0.1 / dynamic_risk_aversion  # Less tilt when risk averse
                
                tilted_weights = optimal_weights * (1 + tilt_strength * return_tilt)
                tilted_weights = np.maximum(tilted_weights, 0.005)  # Maintain minimums
                tilted_weights = tilted_weights / np.sum(tilted_weights)  # Renormalize
                
                self.logger.info(f"Multi-objective optimization successful! "
                               f"Risk aversion: {dynamic_risk_aversion:.2f}, "
                               f"Expected return: {np.sum(expected_returns * tilted_weights)*252:.1%}")
                
                return tilted_weights
            else:
                raise ValueError(f"Optimization failed: {result.solver_status}")
                
        except Exception as e:
            self.logger.warning(f"Multi-objective optimization error: {str(e)}, using equal weights")
            return np.ones(len(trained_model['asset_names'])) / len(trained_model['asset_names'])
    
    def _get_optimal_weights(self, trained_model: Dict, current_regime: int) -> np.ndarray:
        """Override to use multi-objective optimization."""
        return self._get_optimal_weights_multi_objective(trained_model, current_regime)


def main():
    """Run enhanced train/test backtesting."""
    
    setup_logging(log_level="INFO", enable_console=True)
    logger = logging.getLogger(__name__)
    
    logger.info("=== ENHANCED TRAIN/TEST BACKTESTING ===")
    logger.info("Improvements:")
    logger.info("- Enhanced regime detection (momentum, volatility, credit)")
    logger.info("- Multi-objective optimization (return + risk)")
    logger.info("- 3 regimes with enhanced features")
    logger.info("- Dynamic risk aversion")
    
    try:
        # Use a focused set of high-quality assets for cleaner results
        curated_tickers = [
            # Core Equity
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO',
            
            # Bonds
            'AGG', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG',
            
            # Alternatives
            'GLD', 'SLV', 'VNQ', 'DBC',
            
            # Sectors (key ones)
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI',
            
            # Individual stocks (blue chips)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ', 'JPM', 'BRK-B'
        ]
        
        logger.info(f"Using curated universe of {len(curated_tickers)} high-quality assets")
        
        # Run enhanced backtesting
        enhanced_backtester = EnhancedTrainTestBacktester()
        result = enhanced_backtester.run_train_test_backtest(
            train_end_date="2023-12-31",
            initial_capital=1000000.0,
            tickers=curated_tickers
        )
        
        # Print detailed results
        print_enhanced_results(result, curated_tickers)
        
    except Exception as e:
        logger.error(f"Enhanced backtesting failed: {str(e)}")
        import traceback
        traceback.print_exc()


def print_enhanced_results(result, original_tickers):
    """Print enhanced results with additional analysis."""
    
    print("\n" + "="*80)
    print("ENHANCED ROBUST PORTFOLIO OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"Asset Universe: {len(original_tickers)} curated high-quality assets")
    print(f"Training Period End: {result.train_period_end.date()}")
    print(f"Test Period Start: {result.test_period_start.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
    
    total_return = (result.final_portfolio_value / result.initial_capital - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    # Performance metrics
    perf = result.performance_summary
    print(f"\nPerformance Metrics:")
    print(f"  Annualized Return: {perf.get('annualized_return', 0)*100:.2f}%")
    print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
    print(f"  Maximum Drawdown: {perf.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Annualized Volatility: {perf.get('annualized_volatility', 0)*100:.2f}%")
    
    # Portfolio composition analysis
    print(f"\n" + "-"*50)
    print("PORTFOLIO COMPOSITION ANALYSIS")
    print("-"*50)
    
    if not result.regime_allocations.empty:
        latest_allocation = result.regime_allocations.iloc[-1]
        regime = latest_allocation.get('regime', 'N/A')
        print(f"Current Regime: {regime}")
        
        # Get weight columns and categorize
        weight_cols = [col for col in latest_allocation.index if col.startswith('weight_')]
        weights_data = [(col.replace('weight_', ''), latest_allocation[col]) for col in weight_cols]
        weights_data.sort(key=lambda x: x[1], reverse=True)
        
        # Categorize assets
        equity_weight = sum(w[1] for w in weights_data if w[0] in ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ', 'JPM', 'BRK-B'])
        bond_weight = sum(w[1] for w in weights_data if w[0] in ['AGG', 'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG'])
        alternative_weight = sum(w[1] for w in weights_data if w[0] in ['GLD', 'SLV', 'VNQ', 'DBC'])
        sector_weight = sum(w[1] for w in weights_data if w[0] in ['XLK', 'XLF', 'XLV', 'XLE', 'XLI'])
        
        print(f"\nAsset Class Allocation:")
        print(f"  Equities: {equity_weight*100:.1f}%")
        print(f"  Bonds: {bond_weight*100:.1f}%")
        print(f"  Alternatives: {alternative_weight*100:.1f}%")
        print(f"  Sectors: {sector_weight*100:.1f}%")
        
        print(f"\nTop 10 Holdings:")
        for i, (asset, weight) in enumerate(weights_data[:10]):
            print(f"  {i+1:2d}. {asset:6s}: {weight*100:5.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    main()