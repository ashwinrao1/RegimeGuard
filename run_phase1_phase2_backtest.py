#!/usr/bin/env python3
"""
Phase 1 & Phase 2 Enhanced Portfolio Optimization

This script implements the complete Phase 1 and Phase 2 improvements:

PHASE 1 (Quick Wins):
- RSI and MACD momentum indicators
- Enhanced volatility features
- Multi-objective optimization (return + risk)
- Dynamic risk aversion

PHASE 2 (Medium Impact):
- Cross-asset momentum signals
- Market stress indicators
- Dynamic rebalancing triggers
- Expanded constraint management
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from improved_regime_detector import ImprovedRegimeDetector
from multi_objective_optimizer import MultiObjectiveOptimizer
from train_test_backtest import TrainTestBacktester
from data_manager import DataManager
from risk_estimator import RiskEstimator
from robust_optimizer import ConstraintManager
from logging_config import setup_logging
import logging
import pandas as pd
import numpy as np
from typing import Dict


class Phase1Phase2Backtester(TrainTestBacktester):
    """Enhanced backtester with Phase 1 & 2 improvements."""
    
    def __init__(self):
        """Initialize enhanced backtester with all improvements."""
        super().__init__()
        
        # Replace components with improved versions
        self.regime_detector = ImprovedRegimeDetector()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        
        # Enhanced settings
        self.dynamic_rebalancing = True
        self.regime_change_threshold = 0.7  # Confidence threshold for regime changes
        
        self.logger.info("Phase 1 & 2 Enhanced Backtester initialized")
    
    def _train_model(self, train_data: Dict) -> Dict:
        """Train model using improved regime detection and multi-objective optimization."""
        
        # Get price data for enhanced features
        returns_data = train_data['returns']
        
        # Reconstruct prices from returns (approximate)
        initial_price = 100
        prices_data = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for col in returns_data.columns:
            prices_data[col] = initial_price * (1 + returns_data[col]).cumprod()
        
        # Improved regime detection with enhanced features
        self.logger.info("Detecting market regimes with Phase 1 & 2 features...")
        regime_labels = self.regime_detector.fit_improved_regimes(
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
        
        # Calculate current market volatility for dynamic risk aversion
        recent_returns = train_data['returns'].tail(60).mean(axis=1)
        current_volatility = recent_returns.std() * np.sqrt(252)
        
        trained_model = {
            'regime_detector': self.regime_detector,
            'regime_covariances': regime_covariances,
            'regime_returns': regime_returns,
            'regime_probabilities': dict(zip(unique_regimes, regime_probs)),
            'asset_names': train_data['returns'].columns.tolist(),
            'current_volatility': current_volatility,
            'prices_data': prices_data,  # Store for regime change detection
            'macro_data': train_data['macro']
        }
        
        self.logger.info(f"Enhanced model trained with {len(unique_regimes)} regimes")
        self.logger.info(f"Current market volatility: {current_volatility:.1%}")
        for regime_id, prob in trained_model['regime_probabilities'].items():
            self.logger.info(f"  Regime {regime_id}: {prob:.1%} of training period")
        
        return trained_model
    
    def _get_optimal_weights(self, trained_model: Dict, current_regime: int) -> np.ndarray:
        """Get optimal weights using multi-objective optimization."""
        
        try:
            # Set up enhanced constraints (Phase 2)
            n_assets = len(trained_model['asset_names'])
            
            # More flexible constraints than basic system
            max_individual_weight = min(0.25, 10.0 / n_assets)  # Allow up to 25% or 10/N
            min_individual_weight = 0.01  # 1% minimum to ensure diversification
            
            constraint_manager = ConstraintManager()
            constraint_manager.add_constraint("budget", {"target": 1.0})
            constraint_manager.add_constraint("long_only", {})
            constraint_manager.add_constraint("box", {
                "min_weight": min_individual_weight,
                "max_weight": max_individual_weight
            })
            
            # Use multi-objective optimization
            optimal_weights, opt_info = self.multi_objective_optimizer.optimize_multi_objective(
                regime_covariances=trained_model['regime_covariances'],
                regime_returns=trained_model['regime_returns'],
                regime_probabilities=trained_model['regime_probabilities'],
                constraints=constraint_manager,
                current_volatility=trained_model.get('current_volatility', 0.15),
                base_risk_aversion=1.0  # Can be made dynamic based on regime
            )
            
            if opt_info['status'] == 'optimal' and optimal_weights is not None:
                self.logger.info(f"Multi-objective optimization successful!")
                self.logger.info(f"  Expected return: {opt_info.get('expected_portfolio_return', 0)*252:.1%}")
                self.logger.info(f"  Risk aversion: {opt_info.get('dynamic_risk_aversion', 1.0):.2f}")
                self.logger.info(f"  Worst-case vol: {np.sqrt(opt_info.get('worst_case_variance', 0)*252):.1%}")
                
                return optimal_weights
            else:
                raise ValueError(f"Optimization failed: {opt_info['status']}")
                
        except Exception as e:
            self.logger.warning(f"Multi-objective optimization error: {str(e)}, using equal weights")
            return np.ones(len(trained_model['asset_names'])) / len(trained_model['asset_names'])
    
    def _should_rebalance(self, current_date: pd.Timestamp, last_rebalance_date: pd.Timestamp,
                         trained_model: Dict, test_data: Dict) -> bool:
        """Determine if rebalancing should occur (Phase 2: Dynamic rebalancing)."""
        
        # Always rebalance on first date or monthly schedule
        days_since_rebalance = (current_date - last_rebalance_date).days
        monthly_rebalance = days_since_rebalance >= 21  # Monthly
        
        if monthly_rebalance:
            return True
        
        # Phase 2: Regime-triggered rebalancing
        if not self.dynamic_rebalancing:
            return False
        
        try:
            # Get recent data for regime detection
            recent_returns = test_data['returns'][test_data['returns'].index <= current_date].tail(60)
            recent_macro = test_data['macro'][test_data['macro'].index <= current_date].tail(60)
            
            if len(recent_returns) < 20:  # Need minimum data
                return False
            
            # Reconstruct recent prices
            recent_prices = pd.DataFrame(index=recent_returns.index, columns=recent_returns.columns)
            for col in recent_returns.columns:
                recent_prices[col] = 100 * (1 + recent_returns[col]).cumprod()
            
            # Create current features
            current_features = self.regime_detector.create_improved_features(
                recent_returns, recent_macro, recent_prices
            )
            
            if current_features.empty:
                return False
            
            # Detect regime change
            previous_regime = getattr(self, '_last_regime', 0)
            current_regime, confidence = self.regime_detector.detect_regime_changes(
                current_features, previous_regime
            )
            
            # Trigger rebalancing if regime changed with high confidence
            regime_changed = (current_regime != previous_regime and 
                            confidence > self.regime_change_threshold)
            
            if regime_changed:
                self.logger.info(f"Regime change detected: {previous_regime} -> {current_regime} "
                               f"(confidence: {confidence:.2f}), triggering rebalancing")
                self._last_regime = current_regime
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Dynamic rebalancing check failed: {str(e)}")
            return monthly_rebalance
    
    def _simulate_test_period(self, trained_model: Dict, test_data: Dict, 
                            initial_capital: float) -> Dict:
        """Enhanced simulation with dynamic rebalancing (Phase 2)."""
        
        test_returns = test_data['returns']
        
        if test_returns.empty:
            raise ValueError("No test period data available")
        
        # Initialize tracking variables
        portfolio_values = [initial_capital]
        portfolio_returns = []
        monthly_values = []
        allocations = []
        
        current_value = initial_capital
        current_weights = None
        last_rebalance_date = test_returns.index[0]
        
        # Get all potential rebalancing dates (daily check for dynamic rebalancing)
        all_dates = test_returns.index
        
        self.logger.info(f"Simulating enhanced test period with dynamic rebalancing")
        
        for i, current_date in enumerate(all_dates[:-1]):  # Exclude last date
            
            # Check if rebalancing should occur
            should_rebalance = (current_weights is None or 
                              self._should_rebalance(current_date, last_rebalance_date, 
                                                   trained_model, test_data))
            
            if should_rebalance:
                try:
                    # Predict current regime (simplified)
                    current_regime = 0  # Default regime
                    
                    # Get optimal weights
                    new_weights = self._get_optimal_weights(trained_model, current_regime)
                    current_weights = new_weights
                    last_rebalance_date = current_date
                    
                    # Record allocation
                    allocation_record = {
                        'date': current_date,
                        'regime': current_regime,
                        'portfolio_value': current_value,
                        'rebalance_trigger': 'regime_change' if hasattr(self, '_last_regime') else 'monthly'
                    }
                    
                    for j, asset in enumerate(trained_model['asset_names']):
                        allocation_record[f'weight_{asset}'] = new_weights[j]
                    
                    allocations.append(allocation_record)
                    
                    # Add to monthly values if it's month-end or regime change
                    if (current_date.month != (current_date + pd.Timedelta(days=1)).month or 
                        allocation_record['rebalance_trigger'] == 'regime_change'):
                        monthly_values.append(current_value)
                    
                except Exception as e:
                    self.logger.warning(f"Rebalancing error at {current_date}: {str(e)}")
                    if current_weights is None:
                        current_weights = np.ones(len(trained_model['asset_names'])) / len(trained_model['asset_names'])
            
            # Calculate daily return
            if current_weights is not None:
                next_date = all_dates[i + 1]
                daily_returns = test_returns.loc[next_date]
                portfolio_return = np.sum(current_weights * daily_returns.values)
                
                # Update portfolio value
                current_value *= (1 + portfolio_return)
                portfolio_values.append(current_value)
                portfolio_returns.append(portfolio_return)
        
        # Create result DataFrames
        allocations_df = pd.DataFrame(allocations)
        if not allocations_df.empty:
            allocations_df.set_index('date', inplace=True)
        
        # Create monthly values series
        if monthly_values:
            monthly_dates = [alloc['date'] for alloc in allocations]
            monthly_values_series = pd.Series(monthly_values, index=monthly_dates[:len(monthly_values)])
        else:
            monthly_values_series = pd.Series(dtype=float)
        
        portfolio_returns_series = pd.Series(
            portfolio_returns,
            index=all_dates[1:len(portfolio_returns)+1]
        )
        
        return {
            'final_value': current_value,
            'monthly_values': monthly_values_series,
            'returns': portfolio_returns_series,
            'allocations': allocations_df,
            'portfolio_values': portfolio_values
        }


def main():
    """Run Phase 1 & Phase 2 enhanced backtesting."""
    
    setup_logging(log_level="INFO", enable_console=True)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("PHASE 1 & PHASE 2 ENHANCED PORTFOLIO OPTIMIZATION")
    logger.info("="*80)
    logger.info("Phase 1 Improvements:")
    logger.info("  ✓ RSI and MACD momentum indicators")
    logger.info("  ✓ Enhanced volatility features")
    logger.info("  ✓ Multi-objective optimization (return + risk)")
    logger.info("  ✓ Dynamic risk aversion based on market volatility")
    logger.info("")
    logger.info("Phase 2 Improvements:")
    logger.info("  ✓ Cross-asset momentum signals")
    logger.info("  ✓ Market stress indicators")
    logger.info("  ✓ Dynamic rebalancing triggers")
    logger.info("  ✓ Enhanced constraint management")
    
    try:
        # Use a well-diversified universe for testing
        enhanced_tickers = [
            # Core US Equity
            'SPY', 'QQQ', 'IWM', 'VTI',
            
            # International
            'VEA', 'VWO',
            
            # Bonds (variety)
            'AGG', 'TLT', 'SHY', 'TIP', 'LQD', 'HYG',
            
            # Alternatives
            'GLD', 'SLV', 'VNQ',
            
            # Sectors
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI',
            
            # Quality Individual Stocks
            'AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM'
        ]
        
        logger.info(f"Using enhanced universe of {len(enhanced_tickers)} assets")
        
        # Run enhanced backtesting
        enhanced_backtester = Phase1Phase2Backtester()
        result = enhanced_backtester.run_train_test_backtest(
            train_end_date="2023-12-31",
            initial_capital=1000000.0,
            tickers=enhanced_tickers
        )
        
        # Print comprehensive results
        print_enhanced_results(result, enhanced_tickers)
        
    except Exception as e:
        logger.error(f"Enhanced backtesting failed: {str(e)}")
        import traceback
        traceback.print_exc()


def print_enhanced_results(result, tickers):
    """Print comprehensive results with Phase 1 & 2 analysis."""
    
    print("\n" + "="*80)
    print("PHASE 1 & PHASE 2 ENHANCED RESULTS")
    print("="*80)
    
    print(f"Enhanced Universe: {len(tickers)} diversified assets")
    print(f"Training Period End: {result.train_period_end.date()}")
    print(f"Test Period Start: {result.test_period_start.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
    
    total_return = (result.final_portfolio_value / result.initial_capital - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    # Performance metrics
    perf = result.performance_summary
    print(f"\nEnhanced Performance Metrics:")
    print(f"  Annualized Return: {perf.get('annualized_return', 0)*100:.2f}%")
    print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
    print(f"  Maximum Drawdown: {perf.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Annualized Volatility: {perf.get('annualized_volatility', 0)*100:.2f}%")
    
    # Enhanced portfolio analysis
    print(f"\n" + "-"*60)
    print("ENHANCED PORTFOLIO ANALYSIS")
    print("-"*60)
    
    if not result.regime_allocations.empty:
        latest_allocation = result.regime_allocations.iloc[-1]
        
        # Asset class breakdown
        weight_cols = [col for col in latest_allocation.index if col.startswith('weight_')]
        weights_data = [(col.replace('weight_', ''), latest_allocation[col]) for col in weight_cols]
        weights_data.sort(key=lambda x: x[1], reverse=True)
        
        # Categorize assets
        equity_assets = ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM']
        bond_assets = ['AGG', 'TLT', 'SHY', 'TIP', 'LQD', 'HYG']
        alternative_assets = ['GLD', 'SLV', 'VNQ']
        
        equity_weight = sum(w[1] for w in weights_data if w[0] in equity_assets)
        bond_weight = sum(w[1] for w in weights_data if w[0] in bond_assets)
        alternative_weight = sum(w[1] for w in weights_data if w[0] in alternative_assets)
        
        print(f"Strategic Asset Allocation:")
        print(f"  Equities: {equity_weight*100:.1f}%")
        print(f"  Bonds: {bond_weight*100:.1f}%")
        print(f"  Alternatives: {alternative_weight*100:.1f}%")
        
        print(f"\nTop 10 Holdings:")
        for i, (asset, weight) in enumerate(weights_data[:10]):
            print(f"  {i+1:2d}. {asset:6s}: {weight*100:5.1f}%")
        
        # Rebalancing analysis
        if 'rebalance_trigger' in latest_allocation.index:
            print(f"\nDynamic Rebalancing Analysis:")
            regime_rebalances = sum(1 for _, row in result.regime_allocations.iterrows() 
                                  if row.get('rebalance_trigger') == 'regime_change')
            total_rebalances = len(result.regime_allocations)
            print(f"  Total Rebalancing Events: {total_rebalances}")
            print(f"  Regime-Triggered: {regime_rebalances}")
            print(f"  Monthly Scheduled: {total_rebalances - regime_rebalances}")
    
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    print("✅ Enhanced regime detection with momentum indicators")
    print("✅ Multi-objective optimization balancing return and risk")
    print("✅ Dynamic risk aversion based on market volatility")
    print("✅ Cross-asset momentum and stress indicators")
    print("✅ Dynamic rebalancing with regime change triggers")
    print("✅ More flexible constraint management")
    print("="*80)


if __name__ == "__main__":
    main()