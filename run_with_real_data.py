#!/usr/bin/env python3
"""
Run Robust Portfolio Optimization with REAL MARKET DATA

This script uses your FRED API key to download real macroeconomic data
and Yahoo Finance for real asset prices.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def run_with_real_data():
    """Run the complete system with real market data."""
    
    print("🚀 ROBUST PORTFOLIO OPTIMIZATION WITH REAL DATA")
    print("=" * 55)
    
    # Check for FRED API key
    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        print("❌ FRED_API_KEY not found!")
        print("💡 Set it with: export FRED_API_KEY='your_api_key'")
        print("🔗 Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return False
    
    print(f"✅ FRED API key found: {fred_key[:8]}...")
    
    try:
        # Import components
        from data_manager import DataManager
        from regime_detector import RegimeDetector
        from risk_estimator import RiskEstimator
        from robust_optimizer import RobustOptimizer
        from backtest_engine import BacktestEngine
        from visualization_engine import VisualizationEngine
        from config import get_config
        from logging_config import setup_logging
        
        # Setup logging
        setup_logging(log_level="INFO")
        
        print("✅ System components imported successfully!")
        
        # Load configuration
        config = get_config()
        print(f"✅ Configuration loaded")
        
        # Initialize components
        print("\n🔧 Initializing components...")
        data_manager = DataManager(fred_api_key=fred_key)
        regime_detector = RegimeDetector(method="kmeans", random_state=42)
        risk_estimator = RiskEstimator()
        optimizer = RobustOptimizer()
        backtest_engine = BacktestEngine()
        viz_engine = VisualizationEngine()
        
        print("✅ All components initialized with FRED API!")
        
        # Define date range for real data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 years
        
        print(f"\n📊 DOWNLOADING REAL MARKET DATA")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Assets: {config.data.default_tickers}")
        print(f"   Macro indicators: {config.data.macro_series}")
        
        # Download real asset data from Yahoo Finance
        print("\n📈 Downloading asset prices from Yahoo Finance...")
        try:
            asset_prices = data_manager.download_asset_data(
                tickers=config.data.default_tickers,
                start_date=start_date,
                end_date=end_date
            )
            print(f"✅ Asset data downloaded: {asset_prices.shape}")
            print(f"   Assets: {list(asset_prices.columns)}")
            print(f"   Date range: {asset_prices.index[0].date()} to {asset_prices.index[-1].date()}")
            
        except Exception as e:
            print(f"❌ Asset data download failed: {str(e)}")
            print("💡 Check your internet connection and try again")
            return False
        
        # Download real macro data from FRED
        print("\n📊 Downloading macroeconomic data from FRED...")
        try:
            macro_data = data_manager.download_macro_data(
                series_ids=config.data.macro_series,
                start_date=start_date,
                end_date=end_date
            )
            print(f"✅ Macro data downloaded: {macro_data.shape}")
            print(f"   Series: {list(macro_data.columns)}")
            
            # Show latest macro values
            print(f"\n📊 Latest Macro Indicators:")
            latest_macro = macro_data.dropna().tail(1)
            if not latest_macro.empty:
                for col in latest_macro.columns:
                    value = latest_macro[col].iloc[0]
                    print(f"   {col}: {value:.2f}")
            
        except Exception as e:
            print(f"❌ Macro data download failed: {str(e)}")
            print("💡 Check your FRED API key and try again")
            return False
        
        # Calculate returns
        print(f"\n🔢 Processing market data...")
        asset_returns = data_manager.compute_returns(asset_prices)
        print(f"✅ Returns calculated: {asset_returns.shape}")
        
        # Show recent market performance
        print(f"\n📈 Recent Market Performance (Last 30 days):")
        recent_returns = asset_returns.tail(30)
        for asset in recent_returns.columns:
            total_return = (1 + recent_returns[asset]).prod() - 1
            annualized_vol = recent_returns[asset].std() * np.sqrt(252)
            print(f"   {asset}: {total_return:+.1%} return, {annualized_vol:.1%} volatility")
        
        # Create regime features with real data
        print(f"\n🔍 REGIME DETECTION WITH REAL DATA")
        print("-" * 40)
        
        regime_features = data_manager.create_regime_features(
            returns=asset_returns,
            macro_data=macro_data
        )
        print(f"✅ Regime features created: {regime_features.shape}")
        print(f"   Features: {list(regime_features.columns)}")
        
        # Detect regimes using real market data
        regime_labels = regime_detector.fit_regimes(regime_features, n_regimes=config.regime.n_regimes)
        
        # Align returns with regime features (since features may have dropped NaN rows)
        aligned_returns = asset_returns.reindex(regime_features.index).dropna()
        print(f"✅ Aligned returns with regime features: {aligned_returns.shape}")
        
        # Analyze regime detection results
        regime_stats = regime_detector.get_regime_statistics()
        validation_results = regime_detector.get_validation_results()
        
        print(f"✅ Market regimes detected from real data!")
        
        unique_regimes = np.unique(regime_labels[regime_labels != -1])
        regime_counts = np.bincount(regime_labels[regime_labels != -1])
        
        print(f"\n📊 REAL MARKET REGIME ANALYSIS:")
        for i, (regime_id, count) in enumerate(zip(unique_regimes, regime_counts)):
            pct = count / len(regime_labels[regime_labels != -1]) * 100
            stats = regime_stats.get(regime_id, {})
            avg_return = stats.get('mean_return', 0)
            avg_vol = stats.get('mean_volatility', 0)
            
            # Interpret regime based on characteristics
            if avg_return > 0.001 and avg_vol < 0.15:
                regime_type = "🟢 BULL MARKET"
            elif avg_return < -0.0005 or avg_vol > 0.25:
                regime_type = "🔴 BEAR MARKET"
            else:
                regime_type = "🟡 NEUTRAL MARKET"
            
            print(f"   Regime {regime_id}: {regime_type}")
            print(f"      Duration: {count} days ({pct:.1f}%)")
            print(f"      Avg Return: {avg_return:.4f} daily")
            print(f"      Avg Volatility: {avg_vol:.4f} daily")
        
        print(f"\n🎯 Validation: {'✅ PASSED' if validation_results['valid'] else '❌ FAILED'}")
        if 'silhouette_score' in validation_results:
            print(f"   Silhouette Score: {validation_results['silhouette_score']:.3f}")
        
        # Risk parameter estimation with real data
        print(f"\n⚖️  RISK ESTIMATION WITH REAL DATA")
        print("-" * 40)
        
        regime_parameters = risk_estimator.estimate_all_regime_parameters(aligned_returns, regime_labels)
        
        print(f"✅ Risk parameters estimated from real market data")
        
        # Validate parameters
        validation_results = risk_estimator.validate_current_parameters()
        print(f"🎯 Parameter validation: {'✅ PASSED' if validation_results['overall_valid'] else '❌ FAILED'}")
        
        # Portfolio optimization with real data
        print(f"\n🎯 PORTFOLIO OPTIMIZATION WITH REAL DATA")
        print("-" * 45)
        
        regime_covariances = risk_estimator.estimate_regime_covariance(aligned_returns, regime_labels)
        regime_returns = risk_estimator.estimate_regime_returns(aligned_returns, regime_labels)
        
        print("🔄 Optimizing portfolio for worst-case scenario...")
        optimal_weights = optimizer.optimize_worst_case(regime_covariances)
        
        print("✅ Portfolio optimization completed with real data!")
        
        # Display results
        print(f"\n📊 OPTIMAL PORTFOLIO (Based on Real Market Data):")
        print("=" * 50)
        
        for i, (asset, weight) in enumerate(zip(config.data.default_tickers, optimal_weights)):
            print(f"   {asset}: {weight:>6.1%}")
        
        # Calculate portfolio metrics
        portfolio_metrics = optimizer.calculate_portfolio_metrics(
            optimal_weights, regime_covariances, regime_returns
        )
        
        print(f"\n📈 PORTFOLIO RISK METRICS:")
        worst_case_vol = np.sqrt(portfolio_metrics['worst_case']['variance']) * np.sqrt(252)
        print(f"   Worst-case Annual Volatility: {worst_case_vol:.1%}")
        
        if 'expected' in portfolio_metrics:
            expected_vol = np.sqrt(portfolio_metrics['expected']['variance']) * np.sqrt(252)
            expected_ret = portfolio_metrics['expected'].get('return', 0) * 252
            print(f"   Expected Annual Volatility: {expected_vol:.1%}")
            print(f"   Expected Annual Return: {expected_ret:.1%}")
            if expected_vol > 0:
                sharpe = expected_ret / expected_vol
                print(f"   Expected Sharpe Ratio: {sharpe:.2f}")
        
        # Validation
        validation = optimizer.validate_solution(optimal_weights)
        print(f"\n🎯 Solution Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
        print(f"   Total Allocation: {np.sum(optimal_weights):.1%}")
        print(f"   Largest Position: {np.max(optimal_weights):.1%}")
        print(f"   Number of Positions: {np.sum(optimal_weights > 0.01)}")
        
        # Save results
        print(f"\n💾 SAVING RESULTS")
        print("-" * 20)
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Save portfolio weights
        portfolio_df = pd.DataFrame({
            'Asset': config.data.default_tickers,
            'Weight': optimal_weights,
            'Weight_Pct': [f"{w:.1%}" for w in optimal_weights]
        })
        portfolio_df.to_csv('output/optimal_portfolio_real_data.csv', index=False)
        print("✅ Portfolio saved to: output/optimal_portfolio_real_data.csv")
        
        # Save regime analysis
        regime_analysis = []
        for regime_id in unique_regimes:
            stats = regime_stats.get(regime_id, {})
            regime_analysis.append({
                'Regime_ID': regime_id,
                'Days': regime_counts[regime_id],
                'Percentage': f"{regime_counts[regime_id] / len(regime_labels[regime_labels != -1]) * 100:.1f}%",
                'Avg_Daily_Return': stats.get('mean_return', 0),
                'Avg_Daily_Volatility': stats.get('mean_volatility', 0)
            })
        
        regime_df = pd.DataFrame(regime_analysis)
        regime_df.to_csv('output/regime_analysis_real_data.csv', index=False)
        print("✅ Regime analysis saved to: output/regime_analysis_real_data.csv")
        
        # Create visualizations
        print("\n📊 CREATING VISUALIZATIONS")
        print("-" * 30)
        
        try:
            # Regime timeline
            regime_fig = viz_engine.plot_regime_detection(regime_labels, regime_features.index)
            if regime_fig:
                regime_fig.savefig('output/regime_timeline_real_data.png', dpi=300, bbox_inches='tight')
                print("✅ Regime timeline: output/regime_timeline_real_data.png")
            
            # Portfolio allocation pie chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Only show assets with >1% allocation
            significant_weights = optimal_weights > 0.01
            plot_assets = [asset for i, asset in enumerate(config.data.default_tickers) if significant_weights[i]]
            plot_weights = optimal_weights[significant_weights]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(plot_assets)))
            wedges, texts, autotexts = ax.pie(plot_weights, labels=plot_assets, autopct='%1.1f%%', 
                                            colors=colors, startangle=90)
            
            ax.set_title('Optimal Portfolio Allocation\n(Based on Real Market Data)', fontsize=14, fontweight='bold')
            
            # Add current date
            ax.text(0.02, 0.98, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig('output/portfolio_allocation_real_data.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Portfolio allocation: output/portfolio_allocation_real_data.png")
            
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {str(e)}")
        
        # Generate summary report
        report_lines = [
            "ROBUST PORTFOLIO OPTIMIZATION REPORT",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Source: Real market data via Yahoo Finance & FRED",
            f"Analysis Period: {start_date} to {end_date}",
            "",
            "PORTFOLIO ALLOCATION:",
            "-" * 20
        ]
        
        for asset, weight in zip(config.data.default_tickers, optimal_weights):
            report_lines.append(f"{asset}: {weight:>6.1%}")
        
        report_lines.extend([
            "",
            "RISK METRICS:",
            "-" * 15,
            f"Worst-case Annual Volatility: {worst_case_vol:.1%}",
            f"Total Allocation: {np.sum(optimal_weights):.1%}",
            f"Largest Position: {np.max(optimal_weights):.1%}",
            f"Number of Positions: {np.sum(optimal_weights > 0.01)}",
            "",
            "MARKET REGIME ANALYSIS:",
            "-" * 25
        ])
        
        for regime_id in unique_regimes:
            count = regime_counts[regime_id]
            pct = count / len(regime_labels[regime_labels != -1]) * 100
            report_lines.append(f"Regime {regime_id}: {count} days ({pct:.1f}%)")
        
        report_text = "\n".join(report_lines)
        
        with open('output/portfolio_report_real_data.txt', 'w') as f:
            f.write(report_text)
        print("✅ Full report: output/portfolio_report_real_data.txt")
        
        # Final summary
        print(f"\n🎉 SUCCESS! REAL DATA ANALYSIS COMPLETED!")
        print("=" * 45)
        print("📁 Files created in output/ directory:")
        print("   📊 optimal_portfolio_real_data.csv")
        print("   📈 regime_analysis_real_data.csv") 
        print("   🎨 regime_timeline_real_data.png")
        print("   🥧 portfolio_allocation_real_data.png")
        print("   📋 portfolio_report_real_data.txt")
        
        print(f"\n💡 KEY INSIGHTS FROM REAL DATA:")
        print(f"   🎯 Detected {len(unique_regimes)} distinct market regimes")
        print(f"   📊 Portfolio diversified across {np.sum(optimal_weights > 0.01)} assets")
        print(f"   ⚖️  Worst-case volatility: {worst_case_vol:.1%} annually")
        print(f"   🛡️  Robust against regime uncertainty")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install packages: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your internet connection and API key")
        return False


if __name__ == "__main__":
    success = run_with_real_data()
    if success:
        print(f"\n🚀 Next steps:")
        print(f"   • Check the output/ directory for all results")
        print(f"   • Run backtesting: python examples/complete_example.py")
        print(f"   • Explore notebooks: jupyter notebook notebooks/")
    else:
        print(f"\n💡 Troubleshooting:")
        print(f"   • Verify FRED API key: echo $FRED_API_KEY")
        print(f"   • Check internet connection")
        print(f"   • Try: python run_example.py (uses sample data)")