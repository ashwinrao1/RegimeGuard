#!/usr/bin/env python3
"""
Example script to run the train/test split backtesting functionality.

This script demonstrates how to use the new train/test backtesting feature
that trains on data through 2023 and tests on 2024-current with $1M capital.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from backtest_engine import BacktestEngine
from train_test_backtest import TrainTestBacktester
from logging_config import setup_logging
import logging


def main():
    """Run the train/test backtesting example."""
    # Setup logging
    setup_logging(log_level="INFO", enable_console=True)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Train/Test Split Backtesting Example ===")
    logger.info("Training period: Up to 2023-12-31")
    logger.info("Testing period: 2024-01-01 to current")
    logger.info("Initial capital: $1,000,000")
    logger.info("Using comprehensive stock universe (200+ assets)")
    
    try:
        # Method 1: Using BacktestEngine
        logger.info("\n--- Using BacktestEngine ---")
        backtest_engine = BacktestEngine()
        result = backtest_engine.run_train_test_backtest(
            train_end_date="2023-12-31",
            initial_capital=1000000.0
        )
        
        # Print results
        print_results(result)
        
    except Exception as e:
        logger.error(f"Backtesting failed: {str(e)}")
        
        # Method 2: Direct usage (fallback)
        logger.info("\n--- Using TrainTestBacktester directly ---")
        try:
            backtester = TrainTestBacktester()
            result = backtester.run_train_test_backtest(
                train_end_date="2023-12-31",
                initial_capital=1000000.0
            )
            
            print_results(result)
            
        except Exception as e2:
            logger.error(f"Direct backtesting also failed: {str(e2)}")
            logger.error("This might be due to missing API keys or data access issues")
            logger.info("Please ensure you have:")
            logger.info("1. Internet connection for data download")
            logger.info("2. FRED API key set in environment (optional)")
            logger.info("3. Required Python packages installed")


def print_results(result):
    """Print formatted results."""
    print("\n" + "="*60)
    print("TRAIN/TEST BACKTESTING RESULTS")
    print("="*60)
    
    print(f"Training Period End: {result.train_period_end.date()}")
    print(f"Test Period Start: {result.test_period_start.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
    
    total_return = (result.final_portfolio_value / result.initial_capital - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    if 'annualized_return' in result.performance_summary:
        ann_return = result.performance_summary['annualized_return'] * 100
        print(f"Annualized Return: {ann_return:.2f}%")
    
    if 'sharpe_ratio' in result.performance_summary:
        sharpe = result.performance_summary['sharpe_ratio']
        print(f"Sharpe Ratio: {sharpe:.3f}")
    
    if 'max_drawdown' in result.performance_summary:
        max_dd = result.performance_summary['max_drawdown'] * 100
        print(f"Maximum Drawdown: {max_dd:.2f}%")
    
    print("\nMonthly Portfolio Values:")
    if not result.monthly_portfolio_values.empty:
        for date, value in result.monthly_portfolio_values.tail(6).items():
            print(f"  {date.date()}: ${value:,.2f}")
    
    print("\nRegime Allocations (Last 3 periods):")
    if not result.regime_allocations.empty:
        for idx, row in result.regime_allocations.tail(3).iterrows():
            print(f"  {idx.date()}: Regime {row.get('regime', 'N/A')}")
            # Print asset weights
            weight_cols = [col for col in row.index if col.startswith('weight_')]
            for col in weight_cols:
                asset = col.replace('weight_', '')
                weight = row[col] * 100
                print(f"    {asset}: {weight:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()