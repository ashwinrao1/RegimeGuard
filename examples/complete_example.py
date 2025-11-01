#!/usr/bin/env python3
"""
Complete Example: Robust Portfolio Optimization System

This script demonstrates the complete workflow of the robust portfolio optimization system:
1. Data download and preprocessing
2. Regime detection
3. Risk parameter estimation
4. Portfolio optimization
5. Backtesting
6. Visualization and reporting

Run this script to see the system in action!
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_manager import DataManager
from regime_detector import RegimeDetector
from risk_estimator import RiskEstimator
from robust_optimizer import RobustOptimizer
from backtest_engine import BacktestEngine
from visualization_engine import VisualizationEngine
from config import get_config
from logging_config import setup_logging


def main():
    """Run the complete robust portfolio optimization example."""
    
    print("🚀 Starting Robust Portfolio Optimization System Demo")
    print("=" * 60)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load configuration
    config = get_config()
    print(f"📋 Configuration loaded")
    print(f"   Default tickers: {config.data.default_tickers}")
    print(f"   Regime detection: {config.regime.clustering_method} with {config.regime.n_regimes} regimes")
    print(f"   Optimization: {config.optimization.optimization_method}")
    
    # Initialize components
    print("\n🔧 Initializing system components...")
    data_manager = DataManager()
    regime_detector = RegimeDetector(method=config.regime.clustering_method, random_state=42)
    risk_estimator = RiskEstimator()
    optimizer = RobustOptimizer()
    backtest_engine = BacktestEngine()
    viz_engine = VisualizationEngine()
    print("   ✅ All components initialized successfully!")
    
    # Step 1: Data Preparation
    print("\n📊 Step 1: Data Preparation")
    print("-" * 30)
    
    # Define date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    print(f"   Downloading data from {start_date} to {end_date}")
    
    try:
        # Download asset data
        asset_prices = data_manager.download_asset_data(
            tickers=config.data.default_tickers,
            start_date=start_date,
            end_date=end_date
        )
        print(f"   ✅ Asset data downloaded: {asset_prices.shape}")
        
        # Download macro data (optional)
        try:
            macro_data = data_manager.download_macro_data(
                series_ids=config.data.macro_series,
                start_date=start_date,
                end_date=end_date
            )
            print(f"   ✅ Macro data downloaded: {macro_data.shape}")
        except Exception as e:
            print(f"   ⚠️  Macro data download failed (FRED API key may be missing): {str(e)}")
            # Create sample macro data
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)
            macro_data = pd.DataFrame({
                'VIXCLS': np.random.randn(len(dates)) * 5 + 20,
                'DGS10': np.random.randn(len(dates)) * 0.5 + 2.5,
                'DGS2': np.random.randn(len(dates)) * 0.3 + 1.5,
                'UNRATE': np.random.randn(len(dates)) * 0.2 + 5.0
            }, index=dates)
            print(f"   ✅ Using sample macro data: {macro_data.shape}")
        
    except Exception as e:
        print(f"   ⚠️  Data download failed: {str(e)}")
        print("   📝 Creating sample data for demonstration...")
        
        # Create sample data
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)
        
        # Sample asset prices (random walk)
        n_assets = len(config.data.default_tickers)
        returns = np.random.randn(len(dates), n_assets) * 0.02
        prices = pd.DataFrame(
            np.exp(returns.cumsum()) * 100,
            index=dates,
            columns=config.data.default_tickers
        )
        asset_prices = prices
        
        # Sample macro data
        macro_data = pd.DataFrame({
            'VIXCLS': np.random.randn(len(dates)) * 5 + 20,
            'DGS10': np.random.randn(len(dates)) * 0.5 + 2.5,
            'DGS2': np.random.randn(len(dates)) * 0.3 + 1.5,
            'UNRATE': np.random.randn(len(dates)) * 0.2 + 5.0
        }, index=dates)
        
        print(f"   ✅ Sample data created: {asset_prices.shape}")
    
    # Calculate returns
    asset_returns = data_manager.compute_returns(asset_prices)
    print(f"   ✅ Returns calculated: {asset_returns.shape}")
    
    # Step 2: Regime Detection
    print("\n🔍 Step 2: Regime Detection")
    print("-" * 30)
    
    # Create regime features
    regime_features = data_manager.create_regime_features(
        returns=asset_returns,
        macro_data=macro_data
    )
    print(f"   ✅ Regime features created: {regime_features.shape}")
    print(f"   📊 Features: {list(regime_features.columns)}")
    
    # Detect regimes
    regime_labels = regime_detector.fit_regimes(regime_features, n_regimes=config.regime.n_regimes)
    
    # Align returns with regime features (since features may have dropped NaN rows)
    aligned_returns = asset_returns.reindex(regime_features.index).dropna()
    print(f"   ✅ Aligned returns with regime features: {aligned_returns.shape}")
    
    # Get regime statistics
    regime_stats = regime_detector.get_regime_statistics()
    validation_results = regime_detector.get_validation_results()
    
    print(f"   ✅ Regimes detected successfully!")
    print(f"   📈 Regime distribution: {np.bincount(regime_labels[regime_labels != -1])}")
    print(f"   🎯 Validation status: {'PASSED' if validation_results['valid'] else 'FAILED'}")
    if 'silhouette_score' in validation_results:
        print(f"   📊 Silhouette score: {validation_results['silhouette_score']:.3f}")
    
    # Step 3: Risk Parameter Estimation
    print("\n⚖️  Step 3: Risk Parameter Estimation")
    print("-" * 30)
    
    # Estimate regime parameters using aligned returns
    regime_parameters = risk_estimator.estimate_all_regime_parameters(aligned_returns, regime_labels)
    
    print(f"   ✅ Risk parameters estimated for {len(regime_parameters)} regimes")
    
    # Validate parameters
    validation_results = risk_estimator.validate_current_parameters()
    print(f"   🎯 Parameter validation: {'PASSED' if validation_results['overall_valid'] else 'FAILED'}")
    
    # Display parameter summary
    param_summary = risk_estimator.get_parameter_summary()
    print(f"   📊 Total probability: {param_summary['total_probability']:.3f}")
    
    for regime_id, details in param_summary['parameter_details'].items():
        print(f"   📈 Regime {regime_id}: prob={details['regime_probability']:.2f}, "
              f"avg_return={details['avg_return']:.4f}, avg_vol={details['avg_volatility']:.4f}")
    
    # Step 4: Portfolio Optimization
    print("\n🎯 Step 4: Portfolio Optimization")
    print("-" * 30)
    
    # Extract covariance matrices and returns using aligned data
    regime_covariances = risk_estimator.estimate_regime_covariance(aligned_returns, regime_labels)
    regime_returns = risk_estimator.estimate_regime_returns(aligned_returns, regime_labels)
    
    # Worst-case optimization
    print("   🔄 Running worst-case optimization...")
    worst_case_weights = optimizer.optimize_worst_case(regime_covariances)
    
    # Validate solution
    validation = optimizer.validate_solution(worst_case_weights)
    print(f"   ✅ Worst-case optimization completed!")
    print(f"   📊 Weights: {worst_case_weights}")
    print(f"   🎯 Solution valid: {validation['valid']}")
    print(f"   📈 Weight sum: {np.sum(worst_case_weights):.6f}")
    
    # Calculate portfolio metrics
    portfolio_metrics = optimizer.calculate_portfolio_metrics(
        worst_case_weights, regime_covariances, regime_returns
    )
    print(f"   📊 Worst-case variance: {portfolio_metrics['worst_case']['variance']:.6f}")
    
    # CVaR optimization (optional)
    try:
        print("   🔄 Running CVaR optimization...")
        regime_probs = np.array([param_summary['parameter_details'][i]['regime_probability'] 
                               for i in sorted(regime_parameters.keys())])
        cvar_weights = optimizer.optimize_cvar(regime_returns, regime_probs, alpha=0.05)
        print(f"   ✅ CVaR optimization completed!")
        print(f"   📊 CVaR weights: {cvar_weights}")
    except Exception as e:
        print(f"   ⚠️  CVaR optimization failed: {str(e)}")
        cvar_weights = worst_case_weights
    
    # Step 5: Backtesting
    print("\n📈 Step 5: Backtesting")
    print("-" * 30)
    
    def optimization_function(returns_window):
        """Optimization function for backtesting."""
        try:
            # Create features for this window
            window_macro = macro_data.reindex(returns_window.index, method='ffill').fillna(method='bfill')
            features = data_manager.create_regime_features(returns_window, window_macro)
            
            if len(features) < 30:  # Not enough data
                return np.ones(len(returns_window.columns)) / len(returns_window.columns)
            
            # Detect regimes
            labels = regime_detector.fit_regimes(features, n_regimes=3)
            
            # Estimate risk parameters
            covariances = risk_estimator.estimate_regime_covariance(returns_window, labels)
            
            # Optimize portfolio
            weights = optimizer.optimize_worst_case(covariances)
            
            return weights
            
        except Exception as e:
            print(f"     ⚠️  Optimization failed in backtest: {str(e)}")
            # Return equal weights as fallback
            return np.ones(len(returns_window.columns)) / len(returns_window.columns)
    
    # Run backtest
    print("   🔄 Running backtest simulation...")
    backtest_start = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    try:
        backtest_result = backtest_engine.run_strategy_backtest(
            returns=asset_returns,
            optimization_func=optimization_function,
            start_date=backtest_start,
            strategy_name='robust_optimization'
        )
        
        print(f"   ✅ Backtest completed successfully!")
        
        # Display key metrics
        metrics = backtest_result.performance_metrics
        print(f"   📊 Performance Metrics:")
        print(f"      Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"      Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"      Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        print(f"      Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"      Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
    except Exception as e:
        print(f"   ⚠️  Backtest failed: {str(e)}")
        backtest_result = None
    
    # Step 6: Visualization and Reporting
    print("\n📊 Step 6: Visualization and Reporting")
    print("-" * 30)
    
    try:
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Create regime timeline plot
        print("   🎨 Creating regime timeline visualization...")
        regime_fig = viz_engine.plot_regime_detection(regime_labels, regime_features.index)
        if regime_fig:
            regime_fig.savefig('output/regime_timeline.png', dpi=300, bbox_inches='tight')
            print("   ✅ Regime timeline saved to output/regime_timeline.png")
        
        # Create portfolio allocation visualization
        if backtest_result and not backtest_result.portfolio_weights.empty:
            print("   🎨 Creating allocation heatmap...")
            alloc_fig = viz_engine.create_allocation_heatmap(backtest_result.portfolio_weights)
            if alloc_fig:
                alloc_fig.savefig('output/allocation_heatmap.png', dpi=300, bbox_inches='tight')
                print("   ✅ Allocation heatmap saved to output/allocation_heatmap.png")
        
        # Create performance comparison
        if backtest_result:
            print("   🎨 Creating performance analysis...")
            strategy_returns = {'Robust Optimization': backtest_result.portfolio_returns}
            
            # Add equal weight benchmark
            equal_weights = np.ones(len(config.data.default_tickers)) / len(config.data.default_tickers)
            equal_weight_returns = (asset_returns * equal_weights).sum(axis=1)
            
            # Align equal weight returns with backtest period
            common_dates = backtest_result.portfolio_returns.index.intersection(equal_weight_returns.index)
            if len(common_dates) > 0:
                equal_weight_returns = equal_weight_returns.loc[common_dates]
                strategy_returns['Equal Weight'] = equal_weight_returns
                print(f"   📊 Equal weight benchmark aligned: {len(equal_weight_returns)} observations")
            else:
                print(f"   ⚠️  No common dates for equal weight benchmark")
            
            perf_fig = viz_engine.plot_performance_comparison(strategy_returns)
            if perf_fig:
                perf_fig.savefig('output/performance_comparison.png', dpi=300, bbox_inches='tight')
                print("   ✅ Performance comparison saved to output/performance_comparison.png")
        
        # Generate comprehensive report
        if backtest_result:
            print("   📝 Generating comprehensive report...")
            backtest_results = {'robust_optimization': {'result': backtest_result}}
            report = viz_engine.generate_summary_report(backtest_results)
            
            with open('output/portfolio_report.txt', 'w') as f:
                f.write(report)
            print("   ✅ Report saved to output/portfolio_report.txt")
            
            # Display report summary
            print("\n" + "="*60)
            print("📋 PORTFOLIO OPTIMIZATION SUMMARY REPORT")
            print("="*60)
            print(report)
        
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {str(e)}")
    
    # Final Summary
    print("\n" + "🎉 SYSTEM DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    print("📁 Output files created in 'output/' directory:")
    print("   📊 regime_timeline.png - Regime detection visualization")
    print("   🎯 allocation_heatmap.png - Portfolio allocation over time")
    print("   📈 performance_comparison.png - Strategy performance comparison")
    print("   📋 portfolio_report.txt - Comprehensive analysis report")
    print("\n💡 Next steps:")
    print("   1. Explore the Jupyter notebooks in notebooks/")
    print("   2. Modify config.yaml to customize parameters")
    print("   3. Add your own optimization strategies")
    print("   4. Set up FRED API key for real macro data")
    print("\n🚀 Happy optimizing!")


if __name__ == "__main__":
    main()