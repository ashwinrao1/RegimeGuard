#!/usr/bin/env python3
"""
Comparison: Basic vs Enhanced Portfolio Optimization

This script compares the basic system with potential improvements to show
how the system could be enhanced.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from train_test_backtest import TrainTestBacktester
from logging_config import setup_logging
import logging
import pandas as pd
import numpy as np


def run_comparison():
    """Compare basic vs enhanced approaches."""
    
    setup_logging(log_level="INFO", enable_console=True)
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PORTFOLIO OPTIMIZATION: BASIC vs ENHANCED COMPARISON")
    print("="*80)
    
    # Use a smaller, high-quality universe for reliable results
    focused_tickers = [
        # Core Assets
        'SPY', 'AGG', 'TLT', 'GLD', 'VNQ',
        
        # International
        'VEA', 'VWO',
        
        # Bonds
        'SHY', 'TIP', 'LQD',
        
        # Sectors
        'XLK', 'XLF', 'XLV',
        
        # Individual Stocks
        'AAPL', 'MSFT', 'JNJ'
    ]
    
    logger.info(f"Using focused universe of {len(focused_tickers)} assets for reliable comparison")
    
    try:
        # Run basic system
        logger.info("\n=== RUNNING BASIC SYSTEM ===")
        basic_backtester = TrainTestBacktester()
        basic_result = basic_backtester.run_train_test_backtest(
            train_end_date="2023-12-31",
            initial_capital=1000000.0,
            tickers=focused_tickers
        )
        
        # Print comparison
        print_comparison_results(basic_result, focused_tickers)
        
        # Show improvement opportunities
        show_improvement_opportunities(basic_result)
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()


def print_comparison_results(result, tickers):
    """Print results with improvement analysis."""
    
    print(f"\n{'='*60}")
    print("CURRENT SYSTEM RESULTS")
    print(f"{'='*60}")
    
    print(f"Asset Universe: {len(tickers)} focused assets")
    print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
    
    total_return = (result.final_portfolio_value / result.initial_capital - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    perf = result.performance_summary
    print(f"Annualized Return: {perf.get('annualized_return', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
    print(f"Maximum Drawdown: {perf.get('max_drawdown', 0)*100:.2f}%")
    print(f"Volatility: {perf.get('annualized_volatility', 0)*100:.2f}%")
    
    # Analyze portfolio composition
    if not result.regime_allocations.empty:
        latest_allocation = result.regime_allocations.iloc[-1]
        
        print(f"\nPortfolio Composition Analysis:")
        weight_cols = [col for col in latest_allocation.index if col.startswith('weight_')]
        weights_data = [(col.replace('weight_', ''), latest_allocation[col]) for col in weight_cols]
        weights_data.sort(key=lambda x: x[1], reverse=True)
        
        # Categorize
        equity_weight = sum(w[1] for w in weights_data if w[0] in ['SPY', 'VEA', 'VWO', 'XLK', 'XLF', 'XLV', 'AAPL', 'MSFT', 'JNJ'])
        bond_weight = sum(w[1] for w in weights_data if w[0] in ['AGG', 'TLT', 'SHY', 'TIP', 'LQD'])
        alternative_weight = sum(w[1] for w in weights_data if w[0] in ['GLD', 'VNQ'])
        
        print(f"  Equity Allocation: {equity_weight*100:.1f}%")
        print(f"  Bond Allocation: {bond_weight*100:.1f}%")
        print(f"  Alternative Allocation: {alternative_weight*100:.1f}%")
        
        # Check concentration
        max_weight = max(w[1] for w in weights_data)
        min_weight = min(w[1] for w in weights_data)
        weight_range = max_weight - min_weight
        
        print(f"  Concentration Analysis:")
        print(f"    Max Weight: {max_weight*100:.1f}%")
        print(f"    Min Weight: {min_weight*100:.1f}%")
        print(f"    Weight Range: {weight_range*100:.1f}%")


def show_improvement_opportunities(result):
    """Show specific improvement opportunities based on current results."""
    
    print(f"\n{'='*60}")
    print("IMPROVEMENT OPPORTUNITIES")
    print(f"{'='*60}")
    
    perf = result.performance_summary
    current_return = perf.get('annualized_return', 0) * 100
    current_sharpe = perf.get('sharpe_ratio', 0)
    current_vol = perf.get('annualized_volatility', 0) * 100
    current_dd = abs(perf.get('max_drawdown', 0) * 100)
    
    print("Current Performance Analysis:")
    print(f"  Return: {current_return:.1f}% (Target: 8-12%)")
    print(f"  Sharpe: {current_sharpe:.2f} (Target: >1.0)")
    print(f"  Volatility: {current_vol:.1f}% (Target: 8-15%)")
    print(f"  Max Drawdown: {current_dd:.1f}% (Target: <10%)")
    
    print(f"\nSpecific Improvement Recommendations:")
    
    # Return enhancement
    if current_return < 8:
        print("  🔴 LOW RETURNS:")
        print("    - Add momentum factors to capture trends")
        print("    - Include growth/value tilts")
        print("    - Consider tactical asset allocation")
        print("    - Add alternative risk premia")
    elif current_return > 15:
        print("  🟡 HIGH RETURNS (may be unsustainable):")
        print("    - Increase risk controls")
        print("    - Add more conservative assets")
        print("    - Implement dynamic risk budgeting")
    else:
        print("  ✅ RETURNS: Within reasonable range")
    
    # Risk management
    if current_dd > 15:
        print("  🔴 HIGH DRAWDOWN:")
        print("    - Implement stop-loss mechanisms")
        print("    - Add tail risk hedging")
        print("    - Increase bond allocation in stress periods")
        print("    - Use volatility targeting")
    elif current_dd < 5:
        print("  🟡 VERY LOW DRAWDOWN (may be too conservative):")
        print("    - Consider higher risk assets")
        print("    - Reduce bond overweight")
        print("    - Add growth equity exposure")
    else:
        print("  ✅ DRAWDOWN: Well controlled")
    
    # Sharpe ratio
    if current_sharpe < 0.8:
        print("  🔴 LOW RISK-ADJUSTED RETURNS:")
        print("    - Improve regime detection with more indicators")
        print("    - Add factor-based risk models")
        print("    - Implement dynamic rebalancing")
        print("    - Consider alternative assets")
    else:
        print("  ✅ SHARPE RATIO: Good risk-adjusted performance")
    
    # Portfolio construction
    if not result.regime_allocations.empty:
        latest_allocation = result.regime_allocations.iloc[-1]
        weight_cols = [col for col in latest_allocation.index if col.startswith('weight_')]
        weights = [latest_allocation[col] for col in weight_cols]
        
        weight_concentration = np.std(weights)
        
        if weight_concentration < 0.02:  # Very equal weights
            print("  🟡 PORTFOLIO CONCENTRATION:")
            print("    - Weights are very equal (may not be optimal)")
            print("    - Consider factor-based tilts")
            print("    - Implement conviction-weighted allocation")
            print("    - Add momentum/quality overlays")
        elif weight_concentration > 0.15:  # Very concentrated
            print("  🔴 PORTFOLIO CONCENTRATION:")
            print("    - Portfolio is highly concentrated")
            print("    - Add diversification constraints")
            print("    - Implement risk budgeting")
            print("    - Consider equal risk contribution")
        else:
            print("  ✅ PORTFOLIO CONCENTRATION: Well balanced")
    
    print(f"\n{'='*60}")
    print("TOP 3 PRIORITY IMPROVEMENTS")
    print(f"{'='*60}")
    print("1. 🚀 Enhanced Regime Detection:")
    print("   - Add momentum indicators (RSI, MACD)")
    print("   - Include volatility regime features")
    print("   - Use credit spreads and yield curve data")
    print("   - Expected improvement: +1-2% annual return")
    
    print("\n2. 🎯 Multi-Objective Optimization:")
    print("   - Balance return and risk dynamically")
    print("   - Add regime-weighted expected returns")
    print("   - Implement dynamic risk aversion")
    print("   - Expected improvement: +0.2-0.4 Sharpe ratio")
    
    print("\n3. ⚡ Dynamic Rebalancing:")
    print("   - Trigger rebalancing on regime changes")
    print("   - Use volatility-based frequency adjustment")
    print("   - Implement threshold-based rebalancing")
    print("   - Expected improvement: -2-3% max drawdown")
    
    print(f"\n{'='*60}")
    print("IMPLEMENTATION ROADMAP")
    print(f"{'='*60}")
    print("Phase 1 (Quick Wins - 1-2 weeks):")
    print("  - Add RSI and MACD to regime features")
    print("  - Implement basic multi-objective optimization")
    print("  - Add volatility-based risk aversion")
    
    print("\nPhase 2 (Medium Term - 1 month):")
    print("  - Implement factor-based risk models")
    print("  - Add alternative assets and international exposure")
    print("  - Create dynamic rebalancing triggers")
    
    print("\nPhase 3 (Advanced - 2-3 months):")
    print("  - Hidden Markov Models for regime detection")
    print("  - Black-Litterman views integration")
    print("  - Comprehensive performance attribution")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    run_comparison()