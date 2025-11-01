"""
Visualization engine module for the robust portfolio optimization system.

This module provides comprehensive visualization capabilities for regime analysis,
portfolio allocations, and performance comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from interfaces import VisualizationEngineInterface, BacktestResult
from config import get_config
from logging_config import get_logger


class PortfolioVisualizer:
    """Creates portfolio allocation and performance visualizations."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """Initialize portfolio visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.logger = get_logger(__name__)
        self.default_figsize = figsize
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            self.logger.warning(f"Style '{style}' not available, using default")
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_allocation_heatmap(self, weights: pd.DataFrame, 
                                title: str = "Portfolio Allocation Over Time",
                                figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create portfolio allocation heatmap."""
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        if weights.empty:
            ax.text(0.5, 0.5, 'No allocation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create heatmap
        sns.heatmap(weights.T, 
                   cmap='RdYlBu_r',
                   center=0,
                   annot=False,
                   fmt='.2f',
                   cbar_kws={'label': 'Weight'},
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Assets')
        
        # Format x-axis dates
        if len(weights) > 20:
            # Show fewer date labels for readability
            n_ticks = min(10, len(weights))
            tick_positions = np.linspace(0, len(weights)-1, n_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([weights.index[i].strftime('%Y-%m') for i in tick_positions], 
                              rotation=45)
        
        plt.tight_layout()
        return fig   
 
    def plot_allocation_evolution(self, weights: pd.DataFrame,
                                asset_names: Optional[List[str]] = None,
                                title: str = "Portfolio Allocation Evolution",
                                figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create stacked area plot of portfolio allocation evolution."""
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        if weights.empty:
            ax.text(0.5, 0.5, 'No allocation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Use provided asset names or column names
        if asset_names is None:
            asset_names = weights.columns.tolist()
        
        # Create stacked area plot
        ax.stackplot(weights.index, *[weights[col] for col in weights.columns],
                    labels=asset_names, alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Weight')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_comparison(self, strategy_returns: Dict[str, pd.Series],
                                  title: str = "Strategy Performance Comparison",
                                  figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create performance comparison plot."""
        figsize = figsize or self.default_figsize
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.2))
        
        if not strategy_returns:
            ax1.text(0.5, 0.5, 'No performance data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        # Plot 1: Cumulative returns
        for strategy_name, returns in strategy_returns.items():
            if not returns.empty:
                cumulative_returns = (1 + returns).cumprod()
                ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                        label=strategy_name, linewidth=2)
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling volatility
        for strategy_name, returns in strategy_returns.items():
            if not returns.empty and len(returns) > 20:
                rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
                ax2.plot(rolling_vol.index, rolling_vol.values, 
                        label=strategy_name, linewidth=1, alpha=0.7)
        
        ax2.set_title('Rolling Volatility (20-day)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Annualized Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_risk_return_scatter(self, performance_metrics: Dict[str, Dict[str, float]],
                                 title: str = "Risk-Return Analysis",
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create risk-return scatter plot."""
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        if not performance_metrics:
            ax.text(0.5, 0.5, 'No performance metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract risk and return data
        strategies = []
        returns = []
        risks = []
        sharpe_ratios = []
        
        for strategy, metrics in performance_metrics.items():
            if 'annualized_return' in metrics and 'annualized_volatility' in metrics:
                strategies.append(strategy)
                returns.append(metrics['annualized_return'])
                risks.append(metrics['annualized_volatility'])
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
        
        if not strategies:
            ax.text(0.5, 0.5, 'Insufficient data for risk-return plot', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create scatter plot with color coding by Sharpe ratio
        scatter = ax.scatter(risks, returns, c=sharpe_ratios, 
                           cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (risks[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio')
        
        ax.set_xlabel('Annualized Volatility')
        ax.set_ylabel('Annualized Return')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        return fig


class PerformanceAnalyzer:
    """Creates performance analysis and attribution plots."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def plot_drawdown_analysis(self, returns: pd.Series,
                             title: str = "Drawdown Analysis",
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create drawdown analysis plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        if returns.empty:
            ax1.text(0.5, 0.5, 'No return data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        # Calculate cumulative returns and drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Plot 1: Cumulative returns with running maximum
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                label='Portfolio', color='blue', linewidth=2)
        ax1.plot(running_max.index, running_max.values, 
                label='Running Maximum', color='green', linestyle='--', alpha=0.7)
        ax1.fill_between(cumulative_returns.index, cumulative_returns.values, running_max.values,
                        alpha=0.3, color='red', label='Drawdown')
        
        ax1.set_title('Cumulative Returns and Drawdown')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown percentage
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.7, color='red', label='Drawdown %')
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        
        ax2.set_title('Drawdown Percentage')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        return fig
    
    def create_performance_attribution(self, backtest_result: BacktestResult,
                                     figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """Create performance attribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Portfolio weights over time
        if not backtest_result.portfolio_weights.empty:
            weights_df = backtest_result.portfolio_weights
            
            # Select top assets by average weight
            avg_weights = weights_df.mean().sort_values(ascending=False)
            top_assets = avg_weights.head(5).index
            
            for asset in top_assets:
                if asset in weights_df.columns:
                    axes[0, 0].plot(weights_df.index, weights_df[asset], 
                                   label=asset.replace('weight_', ''), linewidth=2)
            
            axes[0, 0].set_title('Portfolio Weights Over Time')
            axes[0, 0].set_ylabel('Weight')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Rolling returns
        if not backtest_result.portfolio_returns.empty:
            returns = backtest_result.portfolio_returns
            rolling_returns = returns.rolling(window=21).mean() * 252  # Monthly rolling, annualized
            
            axes[0, 1].plot(rolling_returns.index, rolling_returns.values, 
                           color='blue', linewidth=2)
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Rolling Returns (21-day)')
            axes[0, 1].set_ylabel('Annualized Return')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Plot 3: Return distribution
        if not backtest_result.portfolio_returns.empty:
            returns = backtest_result.portfolio_returns
            axes[1, 0].hist(returns.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.4f}')
            axes[1, 0].set_title('Return Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics comparison
        if backtest_result.benchmark_comparison is not None and not backtest_result.benchmark_comparison.empty:
            bench_comp = backtest_result.benchmark_comparison
            
            if 'sharpe_ratio' in bench_comp.columns:
                strategies = bench_comp.index.tolist()
                sharpe_ratios = bench_comp['sharpe_ratio'].values
                
                bars = axes[1, 1].bar(range(len(strategies)), sharpe_ratios, 
                                     color=['red' if 'strategy' in s.lower() else 'blue' for s in strategies])
                axes[1, 1].set_title('Sharpe Ratio Comparison')
                axes[1, 1].set_ylabel('Sharpe Ratio')
                axes[1, 1].set_xticks(range(len(strategies)))
                axes[1, 1].set_xticklabels(strategies, rotation=45, ha='right')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, sharpe_ratios):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig


class VisualizationEngine(VisualizationEngineInterface):
    """Main visualization engine implementing the VisualizationEngineInterface."""
    
    def __init__(self):
        """Initialize visualization engine."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize visualizers
        self.portfolio_visualizer = PortfolioVisualizer()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Import regime visualizer if available
        try:
            from regime_visualization import RegimeVisualizer
            self.regime_visualizer = RegimeVisualizer()
        except ImportError:
            self.regime_visualizer = None
            self.logger.warning("Regime visualization not available")
        
        self.logger.info("VisualizationEngine initialized")
    
    def plot_regime_detection(self, regime_labels: np.ndarray, dates: pd.DatetimeIndex) -> Any:
        """Create regime detection visualization."""
        if self.regime_visualizer is None:
            self.logger.error("Regime visualizer not available")
            return None
        
        return self.regime_visualizer.plot_regime_timeline(regime_labels, dates)
    
    def create_allocation_heatmap(self, weights: pd.DataFrame) -> Any:
        """Create portfolio allocation heatmap."""
        return self.portfolio_visualizer.create_allocation_heatmap(weights)
    
    def plot_performance_comparison(self, strategy_returns: Dict[str, pd.Series]) -> Any:
        """Create performance comparison plots."""
        return self.portfolio_visualizer.plot_performance_comparison(strategy_returns)
    
    def generate_summary_report(self, backtest_results: Dict) -> str:
        """Generate comprehensive summary report."""
        report = []
        report.append("=== Portfolio Optimization Summary Report ===\n")
        
        if not backtest_results:
            return "No backtest results available for report generation."
        
        # Overall summary
        report.append("Backtest Summary:")
        report.append(f"Number of strategies tested: {len(backtest_results)}")
        
        # Strategy performance summary
        strategy_metrics = {}
        for strategy_name, result_data in backtest_results.items():
            if 'result' in result_data:
                backtest_result = result_data['result']
                metrics = backtest_result.performance_metrics
                strategy_metrics[strategy_name] = metrics
        
        if strategy_metrics:
            report.append("\nStrategy Performance:")
            
            # Find best performing strategies
            best_return = max(strategy_metrics.items(), 
                            key=lambda x: x[1].get('annualized_return', -float('inf')))
            best_sharpe = max(strategy_metrics.items(), 
                            key=lambda x: x[1].get('sharpe_ratio', -float('inf')))
            
            report.append(f"Best Return: {best_return[0]} ({best_return[1].get('annualized_return', 0):.2%})")
            report.append(f"Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].get('sharpe_ratio', 0):.3f})")
        
        # Detailed metrics table
        if strategy_metrics:
            report.append("\nDetailed Performance Metrics:")
            report.append("-" * 80)
            report.append(f"{'Strategy':<20} {'Return':<10} {'Vol':<10} {'Sharpe':<10} {'MaxDD':<10}")
            report.append("-" * 80)
            
            for strategy, metrics in strategy_metrics.items():
                ret = metrics.get('annualized_return', 0)
                vol = metrics.get('annualized_volatility', 0)
                sharpe = metrics.get('sharpe_ratio', 0)
                dd = metrics.get('max_drawdown', 0)
                
                report.append(f"{strategy:<20} {ret:<10.2%} {vol:<10.2%} {sharpe:<10.3f} {dd:<10.2%}")
        
        return "\n".join(report)
    
    def create_comprehensive_dashboard(self, backtest_results: Dict[str, Any],
                                    regime_data: Optional[Dict] = None,
                                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create a comprehensive visualization dashboard.
        
        Args:
            backtest_results: Dictionary with backtest results
            regime_data: Optional regime detection data
            save_path: Optional path to save plots
            
        Returns:
            Dictionary with created plots and their file paths
        """
        self.logger.info("Creating comprehensive visualization dashboard")
        
        dashboard = {
            'plots': {},
            'saved_files': {}
        }
        
        try:
            # Extract strategy returns for comparison
            strategy_returns = {}
            performance_metrics = {}
            
            for strategy_name, result_data in backtest_results.items():
                if 'result' in result_data:
                    backtest_result = result_data['result']
                    strategy_returns[strategy_name] = backtest_result.portfolio_returns
                    performance_metrics[strategy_name] = backtest_result.performance_metrics
            
            # 1. Performance comparison plot
            if strategy_returns:
                perf_fig = self.portfolio_visualizer.plot_performance_comparison(strategy_returns)
                dashboard['plots']['performance_comparison'] = perf_fig
                
                if save_path:
                    perf_path = f"{save_path}/performance_comparison.png"
                    perf_fig.savefig(perf_path, dpi=300, bbox_inches='tight')
                    dashboard['saved_files']['performance_comparison'] = perf_path
            
            # 2. Risk-return scatter plot
            if performance_metrics:
                risk_return_fig = self.portfolio_visualizer.create_risk_return_scatter(performance_metrics)
                dashboard['plots']['risk_return'] = risk_return_fig
                
                if save_path:
                    rr_path = f"{save_path}/risk_return_analysis.png"
                    risk_return_fig.savefig(rr_path, dpi=300, bbox_inches='tight')
                    dashboard['saved_files']['risk_return'] = rr_path
            
            # 3. Allocation heatmaps for each strategy
            for strategy_name, result_data in backtest_results.items():
                if 'result' in result_data:
                    backtest_result = result_data['result']
                    if not backtest_result.portfolio_weights.empty:
                        alloc_fig = self.portfolio_visualizer.create_allocation_heatmap(
                            backtest_result.portfolio_weights,
                            title=f"Allocation Heatmap: {strategy_name}"
                        )
                        dashboard['plots'][f'allocation_{strategy_name}'] = alloc_fig
                        
                        if save_path:
                            alloc_path = f"{save_path}/allocation_{strategy_name}.png"
                            alloc_fig.savefig(alloc_path, dpi=300, bbox_inches='tight')
                            dashboard['saved_files'][f'allocation_{strategy_name}'] = alloc_path
            
            # 4. Drawdown analysis for main strategy
            main_strategy = list(backtest_results.keys())[0] if backtest_results else None
            if main_strategy and 'result' in backtest_results[main_strategy]:
                main_result = backtest_results[main_strategy]['result']
                if not main_result.portfolio_returns.empty:
                    dd_fig = self.performance_analyzer.plot_drawdown_analysis(
                        main_result.portfolio_returns,
                        title=f"Drawdown Analysis: {main_strategy}"
                    )
                    dashboard['plots']['drawdown_analysis'] = dd_fig
                    
                    if save_path:
                        dd_path = f"{save_path}/drawdown_analysis.png"
                        dd_fig.savefig(dd_path, dpi=300, bbox_inches='tight')
                        dashboard['saved_files']['drawdown_analysis'] = dd_path
            
            # 5. Performance attribution
            if main_strategy and 'result' in backtest_results[main_strategy]:
                main_result = backtest_results[main_strategy]['result']
                attr_fig = self.performance_analyzer.create_performance_attribution(main_result)
                dashboard['plots']['performance_attribution'] = attr_fig
                
                if save_path:
                    attr_path = f"{save_path}/performance_attribution.png"
                    attr_fig.savefig(attr_path, dpi=300, bbox_inches='tight')
                    dashboard['saved_files']['performance_attribution'] = attr_path
            
            # 6. Regime analysis (if available)
            if regime_data and self.regime_visualizer:
                regime_labels = regime_data.get('regime_labels')
                dates = regime_data.get('dates')
                
                if regime_labels is not None and dates is not None:
                    regime_fig = self.regime_visualizer.plot_regime_timeline(regime_labels, dates)
                    dashboard['plots']['regime_analysis'] = regime_fig
                    
                    if save_path:
                        regime_path = f"{save_path}/regime_analysis.png"
                        regime_fig.savefig(regime_path, dpi=300, bbox_inches='tight')
                        dashboard['saved_files']['regime_analysis'] = regime_path
            
            self.logger.info(f"Created {len(dashboard['plots'])} visualization plots")
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            dashboard['error'] = str(e)
        
        return dashboard
    
    def save_all_plots(self, plots: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Save all plots to specified directory.
        
        Args:
            plots: Dictionary of plot objects
            output_dir: Output directory
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        for plot_name, plot_obj in plots.items():
            try:
                file_path = os.path.join(output_dir, f"{plot_name}.png")
                plot_obj.savefig(file_path, dpi=300, bbox_inches='tight')
                saved_files[plot_name] = file_path
                self.logger.debug(f"Saved plot: {plot_name} -> {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save plot {plot_name}: {str(e)}")
        
        return saved_files