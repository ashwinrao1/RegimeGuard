"""
Backtesting engine module for the robust portfolio optimization system.

This module implements comprehensive backtesting functionality including
rolling window simulation, performance metrics, and benchmark comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import warnings

from interfaces import BacktestEngineInterface, BacktestResult, TrainTestResult
from config import get_config
from logging_config import get_logger


class PerformanceCalculator:
    """Calculates portfolio performance metrics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        if returns.empty:
            return {'error': 'Empty returns series'}
        
        # Remove any NaN values
        clean_returns = returns.dropna()
        if clean_returns.empty:
            return {'error': 'No valid returns after cleaning'}
        
        metrics = {}
        
        # Basic return metrics
        total_return = (1 + clean_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(clean_returns)) - 1
        
        metrics.update({
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'mean_daily_return': float(clean_returns.mean()),
            'median_daily_return': float(clean_returns.median())
        })
        
        # Risk metrics
        daily_volatility = clean_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        metrics.update({
            'daily_volatility': float(daily_volatility),
            'annualized_volatility': float(annualized_volatility),
            'skewness': float(clean_returns.skew()),
            'kurtosis': float(clean_returns.kurtosis())
        })
        
        # Risk-adjusted metrics
        if annualized_volatility > 0:
            sharpe_ratio = annualized_return / annualized_volatility
            metrics['sharpe_ratio'] = float(sharpe_ratio)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(clean_returns)
        metrics.update(drawdown_metrics)
        
        # Value at Risk and Expected Shortfall
        var_es_metrics = self._calculate_var_es(clean_returns)
        metrics.update(var_es_metrics)
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(clean_returns, benchmark_returns)
            metrics.update(benchmark_metrics)
        
        return metrics
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        if in_drawdown.any():
            # Find drawdown periods
            drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
            
            start_indices = drawdown_starts[drawdown_starts].index
            end_indices = drawdown_ends[drawdown_ends].index
            
            # Handle case where drawdown continues to end
            if len(start_indices) > len(end_indices):
                end_indices = end_indices.append(pd.Index([returns.index[-1]]))
            
            for start, end in zip(start_indices, end_indices):
                duration = (returns.index.get_loc(end) - returns.index.get_loc(start) + 1)
                drawdown_periods.append(duration)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Calmar ratio (annualized return / max drawdown)
        annualized_return = (1 + returns.mean()) ** 252 - 1
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'max_drawdown': float(max_drawdown),
            'max_drawdown_duration': int(max_drawdown_duration),
            'avg_drawdown_duration': float(avg_drawdown_duration),
            'calmar_ratio': float(calmar_ratio)
        }
    
    def _calculate_var_es(self, returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """Calculate Value at Risk and Expected Shortfall."""
        metrics = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # Value at Risk (negative of quantile for losses)
            var = -returns.quantile(alpha)
            
            # Expected Shortfall (conditional expectation beyond VaR)
            tail_returns = returns[returns <= -var]
            expected_shortfall = -tail_returns.mean() if not tail_returns.empty else var
            
            conf_str = f"{int(conf_level*100)}"
            metrics[f'var_{conf_str}'] = float(var)
            metrics[f'expected_shortfall_{conf_str}'] = float(expected_shortfall)
        
        return metrics
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate benchmark-relative metrics."""
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if aligned_returns.empty:
            return {'tracking_error': 0.0, 'information_ratio': 0.0, 'beta': 0.0, 'alpha': 0.0}
        
        # Active returns
        active_returns = aligned_returns - aligned_benchmark
        
        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Beta and Alpha (CAPM)
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha (Jensen's alpha)
        portfolio_mean = aligned_returns.mean() * 252
        benchmark_mean = aligned_benchmark.mean() * 252
        alpha = portfolio_mean - beta * benchmark_mean
        
        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        
        return {
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'beta': float(beta),
            'alpha': float(alpha),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0
        }


class BenchmarkComparator:
    """Compares portfolio performance against benchmarks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def create_benchmark_portfolios(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create standard benchmark portfolios.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Dictionary mapping benchmark names to return series
        """
        benchmarks = {}
        
        # Equal-weight benchmark
        equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
        benchmarks['equal_weight'] = (returns * equal_weights).sum(axis=1)
        
        # Market cap weighted (approximated by inverse volatility)
        volatilities = returns.std()
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
        benchmarks['inverse_volatility'] = (returns * inv_vol_weights).sum(axis=1)
        
        # Minimum variance (simplified)
        try:
            cov_matrix = returns.cov().values
            inv_cov = np.linalg.inv(cov_matrix + 1e-6 * np.eye(len(cov_matrix)))
            ones = np.ones(len(cov_matrix))
            min_var_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
            benchmarks['minimum_variance'] = (returns * min_var_weights).sum(axis=1)
        except Exception as e:
            self.logger.warning(f"Could not create minimum variance benchmark: {str(e)}")
        
        # Maximum diversification
        try:
            volatilities = returns.std().values
            correlation_matrix = returns.corr().values
            
            # Weights proportional to volatility / portfolio volatility
            weights = volatilities / np.sum(volatilities)
            portfolio_vol = np.sqrt(weights.T @ correlation_matrix @ weights)
            max_div_weights = weights / portfolio_vol
            max_div_weights = max_div_weights / np.sum(max_div_weights)
            
            benchmarks['maximum_diversification'] = (returns * max_div_weights).sum(axis=1)
        except Exception as e:
            self.logger.warning(f"Could not create maximum diversification benchmark: {str(e)}")
        
        return benchmarks
    
    def compare_strategies(self, strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compare multiple strategies.
        
        Args:
            strategy_returns: Dictionary mapping strategy names to return series
            
        Returns:
            DataFrame with comparison metrics
        """
        if not strategy_returns:
            return pd.DataFrame()
        
        calculator = PerformanceCalculator()
        comparison_data = []
        
        for strategy_name, returns in strategy_returns.items():
            if returns.empty:
                continue
            
            metrics = calculator.calculate_performance_metrics(returns)
            metrics['strategy'] = strategy_name
            comparison_data.append(metrics)
        
        if not comparison_data:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('strategy', inplace=True)
        
        # Add rankings
        ranking_metrics = ['annualized_return', 'sharpe_ratio', 'calmar_ratio']
        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df


class TransactionCostModel:
    """Models transaction costs for backtesting."""
    
    def __init__(self, cost_bps: float = 5.0, cost_model: str = "proportional"):
        """Initialize transaction cost model.
        
        Args:
            cost_bps: Transaction cost in basis points
            cost_model: Cost model type ("proportional", "fixed", "market_impact")
        """
        self.cost_bps = cost_bps
        self.cost_model = cost_model
        self.logger = get_logger(__name__)
    
    def calculate_transaction_costs(self, old_weights: np.ndarray, new_weights: np.ndarray,
                                  portfolio_value: float = 1.0) -> float:
        """Calculate transaction costs for portfolio rebalancing.
        
        Args:
            old_weights: Previous portfolio weights
            new_weights: New portfolio weights
            portfolio_value: Portfolio value
            
        Returns:
            Transaction cost as fraction of portfolio value
        """
        if len(old_weights) != len(new_weights):
            raise ValueError("Weight arrays must have same length")
        
        # Calculate turnover (sum of absolute weight changes)
        turnover = np.sum(np.abs(new_weights - old_weights))
        
        if self.cost_model == "proportional":
            # Proportional cost model
            cost = turnover * (self.cost_bps / 10000.0)
        
        elif self.cost_model == "fixed":
            # Fixed cost per transaction
            n_trades = np.sum(np.abs(new_weights - old_weights) > 1e-6)
            cost = n_trades * (self.cost_bps / 10000.0)
        
        elif self.cost_model == "market_impact":
            # Square-root market impact model
            cost = turnover * (self.cost_bps / 10000.0) * np.sqrt(turnover)
        
        else:
            raise ValueError(f"Unknown cost model: {self.cost_model}")
        
        return cost
    
    def apply_transaction_costs(self, returns: pd.Series, weights_df: pd.DataFrame) -> pd.Series:
        """Apply transaction costs to return series.
        
        Args:
            returns: Portfolio returns before costs
            weights_df: DataFrame with portfolio weights over time
            
        Returns:
            Portfolio returns after transaction costs
        """
        if weights_df.empty or returns.empty:
            return returns
        
        # Align data
        aligned_weights = weights_df.reindex(returns.index, method='ffill').fillna(0)
        
        cost_adjusted_returns = returns.copy()
        
        for i in range(1, len(aligned_weights)):
            old_weights = aligned_weights.iloc[i-1].values
            new_weights = aligned_weights.iloc[i].values
            
            # Calculate transaction cost
            transaction_cost = self.calculate_transaction_costs(old_weights, new_weights)
            
            # Subtract cost from return
            cost_adjusted_returns.iloc[i] -= transaction_cost
        
        return cost_adjusted_returns


class BacktestSimulator:
    """Orchestrates rolling window backtesting simulation."""
    
    def __init__(self, estimation_window: int = 252, rebalance_frequency: str = "monthly",
                 min_history: int = 504):
        """Initialize backtest simulator.
        
        Args:
            estimation_window: Number of days for parameter estimation
            rebalance_frequency: Rebalancing frequency ("daily", "weekly", "monthly")
            min_history: Minimum history required before starting backtest
        """
        self.estimation_window = estimation_window
        self.rebalance_frequency = rebalance_frequency
        self.min_history = min_history
        self.logger = get_logger(__name__)
        
        # Set rebalancing frequency
        self.rebalance_freq_map = {
            "daily": 1,
            "weekly": 5,
            "monthly": 21,
            "quarterly": 63
        }
        
        if rebalance_frequency not in self.rebalance_freq_map:
            raise ValueError(f"Unknown rebalance frequency: {rebalance_frequency}")
        
        self.rebalance_days = self.rebalance_freq_map[rebalance_frequency]
    
    def run_backtest(self, returns: pd.DataFrame, 
                    optimization_func: Callable,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    transaction_cost_bps: float = 5.0) -> Dict[str, Any]:
        """Run rolling window backtest.
        
        Args:
            returns: DataFrame with asset returns
            optimization_func: Function that takes returns and returns optimal weights
            start_date: Backtest start date (optional)
            end_date: Backtest end date (optional)
            transaction_cost_bps: Transaction costs in basis points
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest with {self.rebalance_frequency} rebalancing")
        
        # Determine backtest period
        if start_date:
            start_idx = returns.index.get_loc(pd.to_datetime(start_date))
        else:
            start_idx = self.min_history
        
        if end_date:
            end_idx = returns.index.get_loc(pd.to_datetime(end_date))
        else:
            end_idx = len(returns) - 1
        
        if start_idx >= end_idx:
            raise ValueError("Invalid backtest period")
        
        # Initialize results storage
        portfolio_weights = []
        portfolio_returns = []
        rebalance_dates = []
        optimization_results = []
        
        # Initialize transaction cost model
        cost_model = TransactionCostModel(transaction_cost_bps)
        
        current_weights = None
        
        # Rolling window backtest
        for i in range(start_idx, end_idx + 1):
            current_date = returns.index[i]
            
            # Check if it's a rebalancing day
            days_since_start = i - start_idx
            is_rebalance_day = (days_since_start % self.rebalance_days == 0) or (current_weights is None)
            
            if is_rebalance_day:
                # Get estimation window
                est_start = max(0, i - self.estimation_window)
                est_end = i
                
                estimation_returns = returns.iloc[est_start:est_end]
                
                try:
                    # Run optimization
                    new_weights = optimization_func(estimation_returns)
                    
                    # Ensure weights are valid
                    if new_weights is None or len(new_weights) != len(returns.columns):
                        self.logger.warning(f"Invalid weights at {current_date}, using equal weights")
                        new_weights = np.ones(len(returns.columns)) / len(returns.columns)
                    
                    # Normalize weights
                    if np.sum(new_weights) > 0:
                        new_weights = new_weights / np.sum(new_weights)
                    else:
                        new_weights = np.ones(len(returns.columns)) / len(returns.columns)
                    
                    current_weights = new_weights
                    rebalance_dates.append(current_date)
                    
                    optimization_results.append({
                        'date': current_date,
                        'weights': new_weights.copy(),
                        'estimation_start': returns.index[est_start],
                        'estimation_end': returns.index[est_end-1]
                    })
                
                except Exception as e:
                    self.logger.error(f"Optimization failed at {current_date}: {str(e)}")
                    if current_weights is None:
                        current_weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            # Record current weights
            portfolio_weights.append({
                'date': current_date,
                'weights': current_weights.copy()
            })
            
            # Calculate portfolio return
            if i < len(returns) - 1:  # Don't calculate return for last day
                daily_returns = returns.iloc[i + 1]  # Next day's returns
                portfolio_return = np.sum(current_weights * daily_returns.values)
                portfolio_returns.append({
                    'date': returns.index[i + 1],
                    'return': portfolio_return
                })
        
        # Convert results to DataFrames
        weights_df = pd.DataFrame(portfolio_weights)
        weights_df.set_index('date', inplace=True)
        
        # Expand weights to separate columns
        weight_columns = [f'weight_{col}' for col in returns.columns]
        weights_expanded = pd.DataFrame(
            weights_df['weights'].tolist(),
            index=weights_df.index,
            columns=weight_columns
        )
        
        returns_df = pd.DataFrame(portfolio_returns)
        if not returns_df.empty:
            returns_df.set_index('date', inplace=True)
            portfolio_return_series = returns_df['return']
        else:
            portfolio_return_series = pd.Series(dtype=float)
        
        # Apply transaction costs
        if transaction_cost_bps > 0 and not weights_expanded.empty:
            cost_adjusted_returns = cost_model.apply_transaction_costs(
                portfolio_return_series, weights_expanded
            )
        else:
            cost_adjusted_returns = portfolio_return_series
        
        return {
            'portfolio_returns': cost_adjusted_returns,
            'portfolio_weights': weights_expanded,
            'rebalance_dates': rebalance_dates,
            'optimization_results': optimization_results,
            'backtest_period': {
                'start': returns.index[start_idx],
                'end': returns.index[end_idx]
            },
            'settings': {
                'estimation_window': self.estimation_window,
                'rebalance_frequency': self.rebalance_frequency,
                'transaction_cost_bps': transaction_cost_bps,
                'min_history': self.min_history
            }
        }


class BacktestEngine(BacktestEngineInterface):
    """Main backtesting engine implementing the BacktestEngineInterface."""
    
    def __init__(self):
        """Initialize backtest engine."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.simulator = BacktestSimulator(
            estimation_window=self.config.backtest.estimation_window,
            rebalance_frequency=self.config.optimization.rebalance_frequency,
            min_history=self.config.backtest.min_history
        )
        
        self.performance_calculator = PerformanceCalculator()
        self.benchmark_comparator = BenchmarkComparator()
        
        # State
        self.backtest_results = {}
        
        self.logger.info("BacktestEngine initialized")
    
    def run_backtest(self, start_date: str, end_date: str, rebalance_freq: str) -> BacktestResult:
        """Run backtesting simulation."""
        raise NotImplementedError("This method requires optimization function and returns data")
    
    def run_train_test_backtest(self, train_end_date: str = "2023-12-31", 
                               initial_capital: float = 1000000.0) -> TrainTestResult:
        """Run train/test split backtesting with simulated capital.
        
        This method trains the model on data up to train_end_date and tests
        performance from the next day to the current date with simulated capital.
        
        Args:
            train_end_date: End date for training period (default: "2023-12-31")
            initial_capital: Initial capital amount (default: $1,000,000)
            
        Returns:
            TrainTestResult with performance tracking and portfolio values
        """
        from train_test_backtest import TrainTestBacktester
        
        self.logger.info(f"Running train/test backtest with ${initial_capital:,.0f} initial capital")
        
        backtester = TrainTestBacktester()
        return backtester.run_train_test_backtest(train_end_date, initial_capital)
    
    def run_strategy_backtest(self, returns: pd.DataFrame,
                            optimization_func: Callable,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            strategy_name: str = "strategy") -> BacktestResult:
        """Run backtest for a specific strategy.
        
        Args:
            returns: Asset returns DataFrame
            optimization_func: Optimization function
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_name: Name of the strategy
            
        Returns:
            BacktestResult object
        """
        self.logger.info(f"Running backtest for strategy: {strategy_name}")
        
        # Run simulation
        simulation_results = self.simulator.run_backtest(
            returns=returns,
            optimization_func=optimization_func,
            start_date=start_date,
            end_date=end_date,
            transaction_cost_bps=self.config.optimization.transaction_cost_bps
        )
        
        # Calculate performance metrics
        portfolio_returns = simulation_results['portfolio_returns']
        performance_metrics = self.performance_calculator.calculate_performance_metrics(portfolio_returns)
        
        # Create benchmark comparisons
        benchmark_returns = self.benchmark_comparator.create_benchmark_portfolios(returns)
        
        # Align benchmark returns with backtest period
        backtest_start = simulation_results['backtest_period']['start']
        backtest_end = simulation_results['backtest_period']['end']
        
        aligned_benchmarks = {}
        for bench_name, bench_returns in benchmark_returns.items():
            aligned_bench = bench_returns.loc[backtest_start:backtest_end]
            if not aligned_bench.empty:
                aligned_benchmarks[bench_name] = aligned_bench
        
        # Compare with benchmarks
        all_strategies = {strategy_name: portfolio_returns}
        all_strategies.update(aligned_benchmarks)
        
        benchmark_comparison = self.benchmark_comparator.compare_strategies(all_strategies)
        
        # Create regime attribution (placeholder - would need regime data)
        regime_attribution = pd.DataFrame()
        
        # Store results
        backtest_result = BacktestResult(
            portfolio_returns=portfolio_returns,
            portfolio_weights=simulation_results['portfolio_weights'],
            performance_metrics=performance_metrics,
            benchmark_comparison=benchmark_comparison,
            regime_attribution=regime_attribution
        )
        
        self.backtest_results[strategy_name] = {
            'result': backtest_result,
            'simulation_details': simulation_results
        }
        
        return backtest_result
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        return self.performance_calculator.calculate_performance_metrics(returns)
    
    def compare_strategies(self, strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compare performance across different strategies."""
        return self.benchmark_comparator.compare_strategies(strategy_returns)
    
    def account_for_transaction_costs(self, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
        """Account for transaction costs in backtest."""
        if weights.empty:
            return pd.Series(dtype=float)
        
        cost_model = TransactionCostModel(cost_bps)
        
        # Calculate costs for each rebalancing
        costs = []
        for i in range(1, len(weights)):
            old_weights = weights.iloc[i-1].values
            new_weights = weights.iloc[i].values
            
            cost = cost_model.calculate_transaction_costs(old_weights, new_weights)
            costs.append(cost)
        
        # Create cost series
        cost_dates = weights.index[1:]
        cost_series = pd.Series(costs, index=cost_dates)
        
        return cost_series
    
    def get_backtest_results(self, strategy_name: str) -> Optional[Dict]:
        """Get detailed backtest results for a strategy."""
        return self.backtest_results.get(strategy_name)
    
    def get_all_results(self) -> Dict[str, Dict]:
        """Get all backtest results."""
        return self.backtest_results.copy()
    
    def generate_backtest_report(self, strategy_name: str) -> str:
        """Generate a comprehensive backtest report.
        
        Args:
            strategy_name: Name of the strategy to report on
            
        Returns:
            Formatted report string
        """
        if strategy_name not in self.backtest_results:
            return f"No results found for strategy: {strategy_name}"
        
        result_data = self.backtest_results[strategy_name]
        backtest_result = result_data['result']
        simulation_details = result_data['simulation_details']
        
        report = []
        report.append(f"=== Backtest Report: {strategy_name} ===\n")
        
        # Backtest settings
        settings = simulation_details['settings']
        report.append("Backtest Settings:")
        report.append(f"  Estimation Window: {settings['estimation_window']} days")
        report.append(f"  Rebalance Frequency: {settings['rebalance_frequency']}")
        report.append(f"  Transaction Costs: {settings['transaction_cost_bps']} bps")
        
        # Backtest period
        period = simulation_details['backtest_period']
        report.append(f"  Period: {period['start'].date()} to {period['end'].date()}")
        report.append("")
        
        # Performance metrics
        metrics = backtest_result.performance_metrics
        report.append("Performance Metrics:")
        
        key_metrics = [
            ('Total Return', 'total_return', '{:.2%}'),
            ('Annualized Return', 'annualized_return', '{:.2%}'),
            ('Annualized Volatility', 'annualized_volatility', '{:.2%}'),
            ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
            ('Maximum Drawdown', 'max_drawdown', '{:.2%}'),
            ('Calmar Ratio', 'calmar_ratio', '{:.3f}')
        ]
        
        for label, key, fmt in key_metrics:
            if key in metrics:
                value = metrics[key]
                report.append(f"  {label}: {fmt.format(value)}")
        
        report.append("")
        
        # Benchmark comparison
        if not backtest_result.benchmark_comparison.empty:
            report.append("Benchmark Comparison (Annualized Return):")
            
            bench_comp = backtest_result.benchmark_comparison
            if 'annualized_return' in bench_comp.columns:
                sorted_strategies = bench_comp.sort_values('annualized_return', ascending=False)
                
                for strategy, row in sorted_strategies.iterrows():
                    return_val = row['annualized_return']
                    report.append(f"  {strategy}: {return_val:.2%}")
        
        return "\n".join(report)