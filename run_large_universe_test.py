#!/usr/bin/env python3
"""
Test script for the train/test backtesting with a large, diversified universe.

This script tests the system with a carefully selected universe of ~50-80 high-quality assets
across multiple asset classes and sectors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from train_test_backtest import TrainTestBacktester
from logging_config import setup_logging
import logging


def get_curated_large_universe():
    """Get a curated universe of high-quality, liquid assets."""
    
    # Core US Equity ETFs
    us_equity = [
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'IWM',   # Russell 2000
        'VTI',   # Total Stock Market
        'VOO',   # S&P 500 (Vanguard)
        'VUG',   # Growth
        'VTV',   # Value
        'VB',    # Small Cap
        'VO',    # Mid Cap
    ]
    
    # International Equity
    international = [
        'VEA',   # Developed Markets
        'VWO',   # Emerging Markets
        'EFA',   # EAFE
        'EEM',   # Emerging Markets
        'VGK',   # Europe
        'VPL',   # Pacific
        'IEFA',  # Core MSCI EAFE
        'IEMG',  # Core MSCI Emerging Markets
    ]
    
    # Sector ETFs
    sectors = [
        'XLK',   # Technology
        'XLF',   # Financials
        'XLV',   # Healthcare
        'XLE',   # Energy
        'XLI',   # Industrials
        'XLY',   # Consumer Discretionary
        'XLP',   # Consumer Staples
        'XLU',   # Utilities
        'XLB',   # Materials
        'XLRE',  # Real Estate
    ]
    
    # Fixed Income
    bonds = [
        'AGG',   # Aggregate Bond
        'BND',   # Total Bond Market
        'TLT',   # 20+ Year Treasury
        'IEF',   # 7-10 Year Treasury
        'SHY',   # 1-3 Year Treasury
        'TIP',   # TIPS
        'LQD',   # Investment Grade Corporate
        'HYG',   # High Yield Corporate
        'EMB',   # Emerging Market Bonds
        'MUB',   # Municipal Bonds
        'VCIT',  # Intermediate Corporate
        'VGIT',  # Intermediate Treasury
    ]
    
    # Commodities & Alternatives
    commodities = [
        'GLD',   # Gold
        'SLV',   # Silver
        'DBC',   # Commodities
        'USO',   # Oil
        'VNQ',   # REITs
        'VTEB',  # Tax-Exempt Bonds
        'PDBC',  # Commodities (Invesco)
        'IAU',   # Gold (iShares)
    ]
    
    # Top Individual Stocks (most liquid and stable)
    individual_stocks = [
        # Mega Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV',
        
        # Financials
        'BRK-B', 'JPM', 'BAC', 'WFC',
        
        # Consumer
        'PG', 'KO', 'WMT', 'HD', 'MCD',
        
        # Industrial
        'BA', 'CAT', 'GE',
        
        # Energy
        'XOM', 'CVX'
    ]
    
    # Combine all
    all_tickers = us_equity + international + sectors + bonds + commodities + individual_stocks
    
    # Remove duplicates and sort
    unique_tickers = sorted(list(set(all_tickers)))
    
    return unique_tickers


def main():
    """Run the large universe train/test backtesting."""
    # Setup logging
    setup_logging(log_level="INFO", enable_console=True)
    logger = logging.getLogger(__name__)
    
    # Get curated universe
    tickers = get_curated_large_universe()
    
    logger.info("=== Large Universe Train/Test Backtesting ===")
    logger.info(f"Using {len(tickers)} carefully selected assets")
    logger.info("Training period: Up to 2023-12-31")
    logger.info("Testing period: 2024-01-01 to current")
    logger.info("Initial capital: $1,000,000")
    
    try:
        # Run backtesting
        backtester = TrainTestBacktester()
        result = backtester.run_train_test_backtest(
            train_end_date="2023-12-31",
            initial_capital=1000000.0,
            tickers=tickers
        )
        
        # Print results
        print_detailed_results(result, tickers)
        
    except Exception as e:
        logger.error(f"Backtesting failed: {str(e)}")
        import traceback
        traceback.print_exc()


def print_detailed_results(result, original_tickers):
    """Print detailed formatted results."""
    print("\n" + "="*80)
    print("LARGE UNIVERSE TRAIN/TEST BACKTESTING RESULTS")
    print("="*80)
    
    print(f"Original Universe Size: {len(original_tickers)} assets")
    print(f"Training Period End: {result.train_period_end.date()}")
    print(f"Test Period Start: {result.test_period_start.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
    
    total_return = (result.final_portfolio_value / result.initial_capital - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    # Performance metrics
    perf = result.performance_summary
    if 'annualized_return' in perf:
        print(f"Annualized Return: {perf['annualized_return']*100:.2f}%")
    if 'sharpe_ratio' in perf:
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
    if 'max_drawdown' in perf:
        print(f"Maximum Drawdown: {perf['max_drawdown']*100:.2f}%")
    if 'annualized_volatility' in perf:
        print(f"Annualized Volatility: {perf['annualized_volatility']*100:.2f}%")
    
    # Test period details
    if 'test_period_days' in perf:
        print(f"Test Period Duration: {perf['test_period_days']} days ({perf.get('test_period_years', 0):.1f} years)")
    
    print("\n" + "-"*50)
    print("MONTHLY PORTFOLIO VALUES (Last 12 months)")
    print("-"*50)
    if not result.monthly_portfolio_values.empty:
        for date, value in result.monthly_portfolio_values.tail(12).items():
            monthly_return = 0
            if len(result.monthly_portfolio_values) > 1:
                prev_values = result.monthly_portfolio_values[result.monthly_portfolio_values.index < date]
                if not prev_values.empty:
                    prev_value = prev_values.iloc[-1]
                    monthly_return = (value / prev_value - 1) * 100
            
            print(f"  {date.strftime('%Y-%m-%d')}: ${value:,.2f} ({monthly_return:+.1f}%)")
    
    print("\n" + "-"*50)
    print("PORTFOLIO COMPOSITION (Most Recent)")
    print("-"*50)
    if not result.regime_allocations.empty:
        latest_allocation = result.regime_allocations.iloc[-1]
        regime = latest_allocation.get('regime', 'N/A')
        print(f"Current Regime: {regime}")
        
        # Get weight columns and sort by weight
        weight_cols = [col for col in latest_allocation.index if col.startswith('weight_')]
        weights_data = [(col.replace('weight_', ''), latest_allocation[col]) for col in weight_cols]
        weights_data.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 15 Holdings:")
        for i, (asset, weight) in enumerate(weights_data[:15]):
            print(f"  {i+1:2d}. {asset:6s}: {weight*100:5.1f}%")
        
        if len(weights_data) > 15:
            remaining_weight = sum(w[1] for w in weights_data[15:])
            print(f"      Others: {remaining_weight*100:5.1f}% ({len(weights_data)-15} assets)")
    
    print("="*80)


if __name__ == "__main__":
    main()