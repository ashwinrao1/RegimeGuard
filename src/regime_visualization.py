"""
Regime visualization module for the robust portfolio optimization system.

This module provides visualization tools for regime detection results,
including time series plots, transition matrices, and statistical summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings

from logging_config import get_logger


class RegimeVisualizer:
    """Creates visualizations for regime detection analysis."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """Initialize regime visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.logger = get_logger(__name__)
        self.default_figsize = figsize
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if seaborn not available
            plt.style.use('default')
            self.logger.warning(f"Style '{style}' not available, using default")
        
        # Define regime colors
        self.regime_colors = {
            0: '#1f77b4',  # Blue - Bull market
            1: '#ff7f0e',  # Orange - Neutral market  
            2: '#d62728',  # Red - Bear market
            3: '#2ca02c',  # Green
            4: '#9467bd',  # Purple
            -1: '#7f7f7f'  # Gray - Missing data
        }
        
        # Define regime labels
        self.regime_labels = {
            0: 'Regime 0',
            1: 'Regime 1', 
            2: 'Regime 2',
            3: 'Regime 3',
            4: 'Regime 4',
            -1: 'Missing'
        }
    
    def plot_regime_timeline(self, regime_labels: np.ndarray, dates: pd.DatetimeIndex, 
                           market_data: Optional[pd.Series] = None, 
                           title: str = "Market Regime Detection Over Time",
                           figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create a timeline plot showing regime assignments over time.
        
        Args:
            regime_labels: Array of regime labels
            dates: DatetimeIndex corresponding to regime labels
            market_data: Optional market data to overlay (e.g., price series)
            title: Plot title
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating regime timeline plot")
        
        figsize = figsize or self.default_figsize
        fig, axes = plt.subplots(2 if market_data is not None else 1, 1, 
                                figsize=figsize, sharex=True)
        
        if market_data is not None:
            ax_regimes, ax_market = axes
        else:
            ax_regimes = axes if isinstance(axes, plt.Axes) else axes[0]
            ax_market = None
        
        # Plot regime timeline
        unique_regimes = np.unique(regime_labels[regime_labels != -1])
        
        # Create regime blocks
        for i in range(len(regime_labels)):
            regime = regime_labels[i]
            color = self.regime_colors.get(regime, '#7f7f7f')
            
            # Find the end of current regime block
            end_idx = i
            while end_idx < len(regime_labels) - 1 and regime_labels[end_idx + 1] == regime:
                end_idx += 1
            
            if i == 0 or regime_labels[i-1] != regime:  # Start of new regime block
                start_date = dates[i]
                end_date = dates[end_idx]
                
                ax_regimes.axvspan(start_date, end_date, alpha=0.7, color=color, 
                                 label=self.regime_labels.get(regime, f'Regime {regime}'))
        
        # Remove duplicate labels
        handles, labels = ax_regimes.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_regimes.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax_regimes.set_ylabel('Market Regimes')
        ax_regimes.set_title(title)
        ax_regimes.grid(True, alpha=0.3)
        
        # Format x-axis
        ax_regimes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_regimes.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Plot market data if provided
        if market_data is not None and ax_market is not None:
            ax_market.plot(dates, market_data, color='black', linewidth=1, alpha=0.8)
            ax_market.set_ylabel('Market Level')
            ax_market.set_xlabel('Date')
            ax_market.grid(True, alpha=0.3)
            
            # Format x-axis for market plot
            ax_market.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_market.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax_regimes.set_xlabel('Date')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_transition_matrix(self, transition_matrix: np.ndarray, 
                             regime_names: Optional[List[str]] = None,
                             title: str = "Regime Transition Matrix",
                             figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create a heatmap of the regime transition matrix.
        
        Args:
            transition_matrix: Square matrix of transition probabilities
            regime_names: Optional list of regime names
            title: Plot title
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating transition matrix heatmap")
        
        figsize = figsize or (8, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create regime names if not provided
        if regime_names is None:
            n_regimes = transition_matrix.shape[0]
            regime_names = [f'Regime {i}' for i in range(n_regimes)]
        
        # Create heatmap
        sns.heatmap(transition_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   xticklabels=regime_names,
                   yticklabels=regime_names,
                   ax=ax,
                   cbar_kws={'label': 'Transition Probability'})
        
        ax.set_title(title)
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        
        plt.tight_layout()
        return fig
    
    def plot_regime_statistics(self, regime_stats: Dict[int, Dict[str, Any]], 
                             metrics: List[str] = None,
                             title: str = "Regime Statistics Comparison",
                             figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create bar plots comparing statistics across regimes.
        
        Args:
            regime_stats: Dictionary of regime statistics
            metrics: List of metrics to plot
            title: Plot title
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating regime statistics comparison plot")
        
        if metrics is None:
            # Default metrics to plot
            metrics = ['mean_return', 'mean_volatility', 'count', 'frequency']
        
        # Filter metrics that exist in the data
        available_metrics = []
        for metric in metrics:
            if any(metric in stats for stats in regime_stats.values()):
                available_metrics.append(metric)
        
        if not available_metrics:
            self.logger.warning("No valid metrics found for plotting")
            return plt.figure()
        
        figsize = figsize or (12, 8)
        n_metrics = len(available_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        regimes = sorted(regime_stats.keys())
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Extract values for this metric
            values = []
            regime_labels = []
            
            for regime_id in regimes:
                if metric in regime_stats[regime_id]:
                    values.append(regime_stats[regime_id][metric])
                    regime_labels.append(f'Regime {regime_id}')
            
            if values:
                colors = [self.regime_colors.get(regime_id, '#7f7f7f') for regime_id in regimes[:len(values)]]
                bars = ax.bar(regime_labels, values, color=colors, alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}' if isinstance(value, float) else str(value),
                           ha='center', va='bottom')
                
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_regime_duration_analysis(self, episodes: List[Dict[str, Any]], 
                                    title: str = "Regime Duration Analysis",
                                    figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create plots analyzing regime duration patterns.
        
        Args:
            episodes: List of regime episodes from stability analysis
            title: Plot title
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating regime duration analysis plot")
        
        figsize = figsize or (15, 10)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data
        regime_durations = {}
        all_durations = []
        
        for episode in episodes:
            regime_id = episode['regime_id']
            duration = episode['duration_days']
            
            if regime_id not in regime_durations:
                regime_durations[regime_id] = []
            regime_durations[regime_id].append(duration)
            all_durations.append(duration)
        
        # Plot 1: Duration distribution by regime (box plot)
        ax1 = axes[0, 0]
        if regime_durations:
            regime_ids = sorted(regime_durations.keys())
            duration_data = [regime_durations[rid] for rid in regime_ids]
            regime_names = [f'Regime {rid}' for rid in regime_ids]
            
            bp = ax1.boxplot(duration_data, labels=regime_names, patch_artist=True)
            
            # Color the boxes
            for patch, regime_id in zip(bp['boxes'], regime_ids):
                patch.set_facecolor(self.regime_colors.get(regime_id, '#7f7f7f'))
                patch.set_alpha(0.7)
        
        ax1.set_title('Duration Distribution by Regime')
        ax1.set_ylabel('Duration (Days)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overall duration histogram
        ax2 = axes[0, 1]
        if all_durations:
            ax2.hist(all_durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(all_durations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_durations):.1f} days')
            ax2.axvline(np.median(all_durations), color='orange', linestyle='--',
                       label=f'Median: {np.median(all_durations):.1f} days')
        
        ax2.set_title('Overall Duration Distribution')
        ax2.set_xlabel('Duration (Days)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Duration over time
        ax3 = axes[1, 0]
        if episodes:
            episode_dates = [pd.to_datetime(ep['start_date']) for ep in episodes]
            episode_durations = [ep['duration_days'] for ep in episodes]
            episode_regimes = [ep['regime_id'] for ep in episodes]
            
            # Color points by regime
            for regime_id in set(episode_regimes):
                regime_mask = np.array(episode_regimes) == regime_id
                regime_dates = np.array(episode_dates)[regime_mask]
                regime_durs = np.array(episode_durations)[regime_mask]
                
                ax3.scatter(regime_dates, regime_durs, 
                           color=self.regime_colors.get(regime_id, '#7f7f7f'),
                           label=f'Regime {regime_id}', alpha=0.7, s=50)
        
        ax3.set_title('Episode Duration Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Duration (Days)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Average duration by regime
        ax4 = axes[1, 1]
        if regime_durations:
            regime_ids = sorted(regime_durations.keys())
            avg_durations = [np.mean(regime_durations[rid]) for rid in regime_ids]
            regime_names = [f'Regime {rid}' for rid in regime_ids]
            colors = [self.regime_colors.get(rid, '#7f7f7f') for rid in regime_ids]
            
            bars = ax4.bar(regime_names, avg_durations, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, duration in zip(bars, avg_durations):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{duration:.1f}', ha='center', va='bottom')
        
        ax4.set_title('Average Duration by Regime')
        ax4.set_ylabel('Average Duration (Days)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, Dict[str, float]], 
                              top_n: int = 15,
                              title: str = "Feature Importance for Regime Detection",
                              figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create a plot showing feature importance for regime separation.
        
        Args:
            feature_importance: Dictionary with feature importance metrics
            top_n: Number of top features to display
            title: Plot title
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating feature importance plot")
        
        figsize = figsize or (12, 8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        if 'by_feature' not in feature_importance:
            self.logger.warning("No feature importance data available")
            return fig
        
        # Extract data
        features = []
        f_scores = []
        p_values = []
        significant = []
        
        for feature, metrics in feature_importance['by_feature'].items():
            features.append(feature)
            f_scores.append(metrics.get('f_score', 0))
            p_values.append(metrics.get('p_value', 1))
            significant.append(metrics.get('significant', False))
        
        # Sort by F-score
        sorted_indices = np.argsort(f_scores)[::-1][:top_n]
        
        top_features = [features[i] for i in sorted_indices]
        top_f_scores = [f_scores[i] for i in sorted_indices]
        top_p_values = [p_values[i] for i in sorted_indices]
        top_significant = [significant[i] for i in sorted_indices]
        
        # Plot 1: F-scores
        colors = ['green' if sig else 'gray' for sig in top_significant]
        bars1 = ax1.barh(range(len(top_features)), top_f_scores, color=colors, alpha=0.7)
        
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels([f.replace('_', ' ').title() for f in top_features])
        ax1.set_xlabel('F-Score')
        ax1.set_title('Feature F-Scores')
        ax1.grid(True, alpha=0.3)
        
        # Add significance legend
        ax1.legend([plt.Rectangle((0,0),1,1, color='green', alpha=0.7),
                   plt.Rectangle((0,0),1,1, color='gray', alpha=0.7)],
                  ['Significant (p < 0.05)', 'Not Significant'])
        
        # Plot 2: P-values (log scale)
        log_p_values = [-np.log10(max(p, 1e-10)) for p in top_p_values]  # Avoid log(0)
        colors2 = ['green' if sig else 'red' for sig in top_significant]
        
        bars2 = ax1.barh(range(len(top_features)), log_p_values, color=colors2, alpha=0.7)
        
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([f.replace('_', ' ').title() for f in top_features])
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_title('Feature Significance')
        ax2.axvline(-np.log10(0.05), color='black', linestyle='--', 
                   label='Significance Threshold (p=0.05)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def create_regime_summary_plot(self, regime_detector, dates: Optional[pd.DatetimeIndex] = None,
                                 market_data: Optional[pd.Series] = None,
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create a comprehensive summary plot of regime detection results.
        
        Args:
            regime_detector: Fitted RegimeDetector instance
            dates: DatetimeIndex for regime labels
            market_data: Optional market data to overlay
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object with multiple subplots
        """
        self.logger.info("Creating comprehensive regime summary plot")
        
        figsize = figsize or (16, 12)
        fig = plt.figure(figsize=figsize)
        
        # Get regime data
        regime_labels = regime_detector.get_regime_labels()
        regime_stats = regime_detector.get_regime_statistics()
        validation_results = regime_detector.get_validation_results()
        
        if regime_labels is None:
            self.logger.error("No regime labels available")
            return fig
        
        if dates is None:
            features = regime_detector.get_features()
            if features is not None:
                dates = features.index
            else:
                dates = pd.date_range('2020-01-01', periods=len(regime_labels), freq='D')
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Main timeline plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_timeline_on_axis(ax1, regime_labels, dates, market_data)
        
        # 2. Regime statistics
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_regime_bars_on_axis(ax2, regime_stats, 'frequency', 'Regime Frequency')
        
        # 3. Transition matrix (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        if validation_results and 'transition_matrix' in validation_results:
            transition_matrix = np.array(validation_results['transition_matrix'])
            self._plot_transition_on_axis(ax3, transition_matrix)
        else:
            ax3.text(0.5, 0.5, 'Transition Matrix\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        # 4. Duration statistics
        ax4 = fig.add_subplot(gs[2, 0])
        try:
            stability = regime_detector.analyze_regime_stability(dates)
            episodes = stability.get('regime_episodes', [])
            if episodes:
                self._plot_duration_bars_on_axis(ax4, episodes)
            else:
                ax4.text(0.5, 0.5, 'Duration Analysis\nNot Available', 
                        ha='center', va='center', transform=ax4.transAxes)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Duration Analysis\nError: {str(e)[:30]}...', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Quality metrics
        ax5 = fig.add_subplot(gs[2, 1])
        if validation_results and 'silhouette_score' in validation_results:
            silhouette_score = validation_results['silhouette_score']
            ax5.bar(['Silhouette Score'], [silhouette_score], color='skyblue', alpha=0.7)
            ax5.set_ylim(-1, 1)
            ax5.set_ylabel('Score')
            ax5.set_title('Quality Metrics')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Quality Metrics\nNot Available', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        plt.suptitle('Regime Detection Summary', fontsize=16, fontweight='bold')
        return fig
    
    def _plot_timeline_on_axis(self, ax, regime_labels: np.ndarray, dates: pd.DatetimeIndex, 
                              market_data: Optional[pd.Series] = None):
        """Helper method to plot regime timeline on given axis."""
        # Plot regime blocks
        for i in range(len(regime_labels)):
            regime = regime_labels[i]
            color = self.regime_colors.get(regime, '#7f7f7f')
            
            end_idx = i
            while end_idx < len(regime_labels) - 1 and regime_labels[end_idx + 1] == regime:
                end_idx += 1
            
            if i == 0 or regime_labels[i-1] != regime:
                start_date = dates[i]
                end_date = dates[end_idx]
                ax.axvspan(start_date, end_date, alpha=0.7, color=color, 
                          label=self.regime_labels.get(regime, f'Regime {regime}'))
        
        # Plot market data if provided
        if market_data is not None:
            ax2 = ax.twinx()
            ax2.plot(dates, market_data, color='black', linewidth=1, alpha=0.8)
            ax2.set_ylabel('Market Level')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax.set_title('Market Regime Timeline')
        ax.set_ylabel('Regimes')
        ax.grid(True, alpha=0.3)
    
    def _plot_regime_bars_on_axis(self, ax, regime_stats: Dict, metric: str, title: str):
        """Helper method to plot regime statistics bars."""
        regimes = sorted(regime_stats.keys())
        values = [regime_stats[r].get(metric, 0) for r in regimes]
        colors = [self.regime_colors.get(r, '#7f7f7f') for r in regimes]
        
        bars = ax.bar([f'R{r}' for r in regimes], values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    def _plot_transition_on_axis(self, ax, transition_matrix: np.ndarray):
        """Helper method to plot transition matrix."""
        n_regimes = transition_matrix.shape[0]
        regime_names = [f'R{i}' for i in range(n_regimes)]
        
        im = ax.imshow(transition_matrix, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(n_regimes):
            for j in range(n_regimes):
                text = ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_xticks(range(n_regimes))
        ax.set_yticks(range(n_regimes))
        ax.set_xticklabels(regime_names)
        ax.set_yticklabels(regime_names)
        ax.set_title('Transition Matrix')
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
    
    def _plot_duration_bars_on_axis(self, ax, episodes: List[Dict]):
        """Helper method to plot duration statistics."""
        regime_durations = {}
        for episode in episodes:
            regime_id = episode['regime_id']
            duration = episode['duration_days']
            
            if regime_id not in regime_durations:
                regime_durations[regime_id] = []
            regime_durations[regime_id].append(duration)
        
        regimes = sorted(regime_durations.keys())
        avg_durations = [np.mean(regime_durations[r]) for r in regimes]
        colors = [self.regime_colors.get(r, '#7f7f7f') for r in regimes]
        
        bars = ax.bar([f'R{r}' for r in regimes], avg_durations, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, duration in zip(bars, avg_durations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{duration:.1f}', ha='center', va='bottom')
        
        ax.set_title('Average Duration')
        ax.set_ylabel('Days')
        ax.grid(True, alpha=0.3)
    
    def save_all_plots(self, regime_detector, output_dir: str = "plots", 
                      dates: Optional[pd.DatetimeIndex] = None,
                      market_data: Optional[pd.Series] = None) -> Dict[str, str]:
        """Generate and save all regime visualization plots.
        
        Args:
            regime_detector: Fitted RegimeDetector instance
            output_dir: Directory to save plots
            dates: DatetimeIndex for regime labels
            market_data: Optional market data to overlay
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        import os
        
        self.logger.info(f"Saving all regime plots to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # Get data
            regime_labels = regime_detector.get_regime_labels()
            if regime_labels is None:
                return {'error': 'No regime labels available'}
            
            if dates is None:
                features = regime_detector.get_features()
                if features is not None:
                    dates = features.index
                else:
                    dates = pd.date_range('2020-01-01', periods=len(regime_labels), freq='D')
            
            # 1. Timeline plot
            fig1 = self.plot_regime_timeline(regime_labels, dates, market_data)
            path1 = os.path.join(output_dir, 'regime_timeline.png')
            fig1.savefig(path1, dpi=300, bbox_inches='tight')
            saved_plots['timeline'] = path1
            plt.close(fig1)
            
            # 2. Statistics plot
            regime_stats = regime_detector.get_regime_statistics()
            if regime_stats:
                fig2 = self.plot_regime_statistics(regime_stats)
                path2 = os.path.join(output_dir, 'regime_statistics.png')
                fig2.savefig(path2, dpi=300, bbox_inches='tight')
                saved_plots['statistics'] = path2
                plt.close(fig2)
            
            # 3. Transition matrix
            transition_matrix = regime_detector.get_transition_matrix()
            if transition_matrix is not None:
                fig3 = self.plot_transition_matrix(transition_matrix)
                path3 = os.path.join(output_dir, 'transition_matrix.png')
                fig3.savefig(path3, dpi=300, bbox_inches='tight')
                saved_plots['transition_matrix'] = path3
                plt.close(fig3)
            
            # 4. Duration analysis
            try:
                stability = regime_detector.analyze_regime_stability(dates)
                episodes = stability.get('regime_episodes', [])
                if episodes:
                    fig4 = self.plot_regime_duration_analysis(episodes)
                    path4 = os.path.join(output_dir, 'duration_analysis.png')
                    fig4.savefig(path4, dpi=300, bbox_inches='tight')
                    saved_plots['duration_analysis'] = path4
                    plt.close(fig4)
            except Exception as e:
                self.logger.warning(f"Could not create duration analysis plot: {str(e)}")
            
            # 5. Feature importance (if available)
            try:
                quality_metrics = regime_detector.calculate_quality_metrics()
                if 'feature_importance' in quality_metrics and 'by_feature' in quality_metrics['feature_importance']:
                    fig5 = self.plot_feature_importance(quality_metrics['feature_importance'])
                    path5 = os.path.join(output_dir, 'feature_importance.png')
                    fig5.savefig(path5, dpi=300, bbox_inches='tight')
                    saved_plots['feature_importance'] = path5
                    plt.close(fig5)
            except Exception as e:
                self.logger.warning(f"Could not create feature importance plot: {str(e)}")
            
            # 6. Summary plot
            fig6 = self.create_regime_summary_plot(regime_detector, dates, market_data)
            path6 = os.path.join(output_dir, 'regime_summary.png')
            fig6.savefig(path6, dpi=300, bbox_inches='tight')
            saved_plots['summary'] = path6
            plt.close(fig6)
            
        except Exception as e:
            self.logger.error(f"Error saving plots: {str(e)}")
            saved_plots['error'] = str(e)
        
        return saved_plots