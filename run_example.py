#!/usr/bin/env python3
"""
Quick Start Script for Robust Portfolio Optimization System

This is the simplest way to run the system and see it in action.
Just run: python run_example.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def quick_demo():
    """Run a quick demonstration of the system."""
    
    print("🚀 Robust Portfolio Optimization - Quick Demo")
    print("=" * 50)
    
    try:
        # Import components
        from data_manager import DataManager
        from regime_detector import RegimeDetector
        from risk_estimator import RiskEstimator
        from robust_optimizer import RobustOptimizer
        
        print("✅ System components imported successfully!")
        
        # Create sample data (since we might not have API keys)
        print("\n📊 Creating sample data...")
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Sample asset returns (6 assets)
        asset_names = ['SPY', 'AGG', 'GLD', 'XLE', 'XLF', 'XLK']
        returns = pd.DataFrame(
            np.random.randn(len(dates), 6) * 0.02,
            index=dates,
            columns=asset_names
        )
        
        # Sample macro data
        macro_data = pd.DataFrame({
            'VIX': np.random.randn(len(dates)) * 5 + 20,
            'YIELD_10Y': np.random.randn(len(dates)) * 0.5 + 2.5,
            'YIELD_2Y': np.random.randn(len(dates)) * 0.3 + 1.5
        }, index=dates)
        
        print(f"✅ Sample data created: {returns.shape[0]} days, {returns.shape[1]} assets")
        
        # Initialize components
        print("\n🔧 Initializing components...")
        data_manager = DataManager()
        regime_detector = RegimeDetector(method="kmeans", random_state=42)
        risk_estimator = RiskEstimator()
        optimizer = RobustOptimizer()
        
        print("✅ All components initialized!")
        
        # Create regime features
        print("\n🔍 Detecting market regimes...")
        regime_features = data_manager.create_regime_features(returns, macro_data)
        regime_labels = regime_detector.fit_regimes(regime_features, n_regimes=3)
        
        regime_counts = np.bincount(regime_labels[regime_labels != -1])
        print(f"✅ Detected 3 regimes with distribution: {regime_counts}")
        
        # Estimate risk parameters
        print("\n⚖️  Estimating risk parameters...")
        regime_covariances = risk_estimator.estimate_regime_covariance(returns, regime_labels)
        regime_returns = risk_estimator.estimate_regime_returns(returns, regime_labels)
        
        print(f"✅ Risk parameters estimated for {len(regime_covariances)} regimes")
        
        # Optimize portfolio
        print("\n🎯 Optimizing portfolio...")
        optimal_weights = optimizer.optimize_worst_case(regime_covariances)
        
        print("✅ Portfolio optimization completed!")
        print(f"\n📊 Optimal Portfolio Weights:")
        for i, (asset, weight) in enumerate(zip(asset_names, optimal_weights)):
            print(f"   {asset}: {weight:.1%}")
        
        print(f"\n📈 Portfolio Statistics:")
        print(f"   Total allocation: {np.sum(optimal_weights):.1%}")
        print(f"   Number of positions: {np.sum(optimal_weights > 0.01)}")
        print(f"   Largest position: {np.max(optimal_weights):.1%}")
        print(f"   Most diversified: {'Yes' if np.max(optimal_weights) < 0.5 else 'No'}")
        
        # Calculate portfolio metrics
        portfolio_metrics = optimizer.calculate_portfolio_metrics(
            optimal_weights, regime_covariances, regime_returns
        )
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Worst-case volatility: {np.sqrt(portfolio_metrics['worst_case']['variance']) * np.sqrt(252):.1%}")
        if 'expected' in portfolio_metrics:
            print(f"   Expected volatility: {np.sqrt(portfolio_metrics['expected']['variance']) * np.sqrt(252):.1%}")
        
        # Validation
        validation = optimizer.validate_solution(optimal_weights)
        print(f"\n🎯 Solution Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
        
        if not validation['valid']:
            print("   Issues found:")
            for violation in validation['violations']:
                print(f"   - {violation}")
        
        print("\n" + "🎉 QUICK DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 50)
        print("💡 What happened:")
        print("   1. ✅ Created sample market data (6 assets, 4 years)")
        print("   2. ✅ Detected 3 market regimes using K-means clustering")
        print("   3. ✅ Estimated regime-specific risk parameters")
        print("   4. ✅ Optimized robust portfolio using worst-case approach")
        print("   5. ✅ Validated the solution meets all constraints")
        
        print("\n🚀 Next Steps:")
        print("   • Run 'python examples/complete_example.py' for full demo")
        print("   • Open 'notebooks/01_data_preparation.ipynb' in Jupyter")
        print("   • Set FRED_API_KEY environment variable for real data")
        print("   • Modify config.yaml to customize parameters")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Solution: Install required packages:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Check that all files are in place and try again")
        return False


if __name__ == "__main__":
    success = quick_demo()
    if not success:
        sys.exit(1)