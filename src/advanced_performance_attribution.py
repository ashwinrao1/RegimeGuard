"""
Advanced Performance Attribution - Phase 3 Implementation

This module provides comprehensive performance attribution analysis including:
- Regime-based attribution
- Factor-based attribution (Fama-French style)
- Risk budgeting and contribution analysis
- Style analysis and drift detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from logging_config import get_logger


class AdvancedPerformanceAttributor:
    """Advanced performance attribution with regime and factor analysis."""
    
    def __init__(self):
        """Initialize performance attributor."""
        self.logger = get_logger(__name__)
        
        # Attribution components
        self.regime_attribution = {}
        self.factor_attribution = {}
        self.risk_attribution = {}
        self.style_analysis = {}
        
        self.logger.info("Advanced performance attributor initialized")
    
    def analyze_performance(self, portfolio_returns: pd.Series,
                          portfolio_weights: pd.DataFrame,
                          asset_returns: pd.DataFrame,
                          regime_labels: np.ndarray,
                          benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Comprehensive performance attribution analysis."""
        
        self.logger.info("Starting comprehensive performance attribution...")
        
        results = {}
        
        # 1. Regime-based attribution
        results['regime_attribution'] = self._analyze_regime_attribution(
            portfolio_returns, regime_labels
        )
        
        # 2. Factor-based attribution
        results['factor_attribution'] = self._analyze_factor_attribution(
            portfolio_returns, asset_returns
        )
        
        # 3. Risk contribution analysis
        results['risk_attribution'] = self._analyze_risk_contribution(
            portfolio_weights, asset_returns
        )
        
        # 4. Style analysis
        results['style_analysis'] = self._analyze_style_drift(
            portfolio_weights, asset_returns
        )
        
        # 5. Benchmark relative attribution
        if benchmark_returns is not None:
            results['benchmark_attribution'] = self._analyze_benchmark_attribution(
                portfolio_returns, benchmark_returns, portfolio_weights, asset_returns
            )
        
        # 6. Performance decomposition
        results['performance_decomposition'] = self._decompose_performance(
            portfolio_returns, portfolio_weights, asset_returns
        )
        
        self.logger.info("Performance attribution analysis completed")
        return results
    
    def _analyze_regime_attribution(self, portfolio_returns: pd.Series,
                                  regime_labels: np.ndarray) -> Dict:
        """Analyze performance attribution by market regime."""
        
        try:
            # Align returns with regime labels
            if len(regime_labels) != len(portfolio_returns):
                # Map regime labels to return dates (simplified)
                regime_series = pd.Series(regime_labels, 
                                        index=portfolio_returns.index[:len(regime_labels)])
                regime_series = regime_series.reindex(portfolio_returns.index, method='ffill')
            else:
                regime_series = pd.Series(regime_labels, index=portfolio_returns.index)
            
            regime_attribution = {}
            
            # Calculate performance by regime
            unique_regimes = np.unique(regime_labels)
            for regime in unique_regimes:
                regime_mask = regime_series == regime
                regime_returns = portfolio_returns[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_attribution[f'regime_{regime}'] = {
                        'total_return': (1 + regime_returns).prod() - 1,
                        'annualized_return': regime_returns.mean() * 252,
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe_ratio': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                        'max_drawdown': self._calculate_max_drawdown(regime_returns),
                        'days_in_regime': len(regime_returns),
                        'contribution_to_total': regime_returns.sum() / portfolio_returns.sum() if portfolio_returns.sum() != 0 else 0
                    }
            
            # Regime transition analysis
            regime_transitions = self._analyze_regime_transitions(regime_series, portfolio_returns)
            regime_attribution['transitions'] = regime_transitions
            
            return regime_attribution
            
        except Exception as e:
            self.logger.warning(f"Regime attribution analysis failed: {str(e)}")
            return {}
    
    def _analyze_factor_attribution(self, portfolio_returns: pd.Series,
                                  asset_returns: pd.DataFrame) -> Dict:
        """Analyze performance attribution using factor models."""
        
        try:
            # Create factor returns
            factors = self._create_factor_returns(asset_returns)
            
            if factors.empty:
                return {}
            
            # Align data
            aligned_portfolio, aligned_factors = portfolio_returns.align(factors, join='inner')
            
            if len(aligned_portfolio) < 20:  # Need sufficient data
                return {}
            
            # Multi-factor regression
            X = aligned_factors.values
            y = aligned_portfolio.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Factor loadings (betas)
            factor_loadings = dict(zip(aligned_factors.columns, model.coef_))
            
            # Factor contributions
            factor_contributions = {}
            for i, factor_name in enumerate(aligned_factors.columns):
                factor_return = aligned_factors[factor_name].mean() * 252
                factor_loading = model.coef_[i]
                contribution = factor_loading * factor_return
                
                factor_contributions[factor_name] = {
                    'loading': factor_loading,
                    'factor_return': factor_return,
                    'contribution': contribution
                }
            
            # Model statistics
            r_squared = model.score(X, y)
            residual_return = model.intercept_ * 252  # Alpha
            
            factor_attribution = {
                'factor_loadings': factor_loadings,
                'factor_contributions': factor_contributions,
                'alpha': residual_return,
                'r_squared': r_squared,
                'tracking_error': np.std(y - model.predict(X)) * np.sqrt(252)
            }
            
            return factor_attribution
            
        except Exception as e:
            self.logger.warning(f"Factor attribution analysis failed: {str(e)}")
            return {}
    
    def _analyze_risk_contribution(self, portfolio_weights: pd.DataFrame,
                                 asset_returns: pd.DataFrame) -> Dict:
        """Analyze risk contribution by asset and time."""
        
        try:
            if portfolio_weights.empty or asset_returns.empty:
                return {}
            
            # Align data
            aligned_weights, aligned_returns = portfolio_weights.align(asset_returns, join='inner')
            
            if len(aligned_weights) < 10:
                return {}
            
            risk_contributions = {}
            
            # Calculate rolling risk contributions
            window = min(60, len(aligned_weights) // 2)
            
            for i in range(window, len(aligned_weights)):
                date = aligned_weights.index[i]
                
                # Get window data
                window_returns = aligned_returns.iloc[i-window:i]
                current_weights = aligned_weights.iloc[i]
                
                # Calculate covariance matrix
                cov_matrix = window_returns.cov() * 252  # Annualized
                
                # Portfolio variance
                portfolio_var = np.dot(current_weights, np.dot(cov_matrix, current_weights))
                
                # Marginal risk contributions
                marginal_contrib = np.dot(cov_matrix, current_weights)
                
                # Risk contributions
                risk_contrib = current_weights * marginal_contrib / portfolio_var if portfolio_var > 0 else current_weights * 0
                
                risk_contributions[date] = {
                    'portfolio_volatility': np.sqrt(portfolio_var),
                    'risk_contributions': dict(zip(aligned_returns.columns, risk_contrib)),
                    'concentration': np.sum(risk_contrib ** 2)  # Herfindahl index
                }
            
            # Summary statistics
            avg_risk_contrib = {}
            for asset in aligned_returns.columns:
                contributions = [rc['risk_contributions'].get(asset, 0) for rc in risk_contributions.values()]
                avg_risk_contrib[asset] = {
                    'mean_contribution': np.mean(contributions),
                    'std_contribution': np.std(contributions),
                    'max_contribution': np.max(contributions),
                    'min_contribution': np.min(contributions)
                }
            
            return {
                'time_series': risk_contributions,
                'summary': avg_risk_contrib,
                'avg_concentration': np.mean([rc['concentration'] for rc in risk_contributions.values()])
            }
            
        except Exception as e:
            self.logger.warning(f"Risk contribution analysis failed: {str(e)}")
            return {}
    
    def _analyze_style_drift(self, portfolio_weights: pd.DataFrame,
                           asset_returns: pd.DataFrame) -> Dict:
        """Analyze portfolio style drift over time."""
        
        try:
            if portfolio_weights.empty:
                return {}
            
            style_analysis = {}
            
            # Define style categories
            style_categories = {
                'equity': ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO'],
                'bonds': ['AGG', 'TLT', 'SHY', 'TIP', 'LQD', 'HYG'],
                'alternatives': ['GLD', 'SLV', 'VNQ'],
                'growth': ['QQQ', 'VUG', 'XLK'],
                'value': ['VTV', 'XLF'],
                'international': ['VEA', 'VWO', 'EFA', 'EEM']
            }
            
            # Calculate style exposures over time
            style_exposures = {}
            
            for date, weights in portfolio_weights.iterrows():
                exposures = {}
                
                for style, assets in style_categories.items():
                    style_weight = 0
                    for asset in assets:
                        if asset in weights.index:
                            weight_col = f'weight_{asset}' if f'weight_{asset}' in weights.index else asset
                            if weight_col in weights.index:
                                style_weight += weights[weight_col]
                    
                    exposures[style] = style_weight
                
                style_exposures[date] = exposures
            
            # Convert to DataFrame for analysis
            style_df = pd.DataFrame(style_exposures).T
            
            # Calculate style drift metrics
            style_drift = {}
            for style in style_df.columns:
                if len(style_df[style].dropna()) > 1:
                    style_series = style_df[style].dropna()
                    style_drift[style] = {
                        'mean_exposure': style_series.mean(),
                        'std_exposure': style_series.std(),
                        'min_exposure': style_series.min(),
                        'max_exposure': style_series.max(),
                        'drift_volatility': style_series.std(),
                        'trend': self._calculate_trend(style_series)
                    }
            
            # Principal component analysis of style exposures
            if len(style_df) > 10 and len(style_df.columns) > 2:
                pca = PCA(n_components=min(3, len(style_df.columns)))
                pca_result = pca.fit_transform(style_df.fillna(0))
                
                pca_analysis = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'principal_components': pca.components_.tolist()
                }
            else:
                pca_analysis = {}
            
            return {
                'style_exposures': style_df.to_dict(),
                'style_drift_metrics': style_drift,
                'pca_analysis': pca_analysis
            }
            
        except Exception as e:
            self.logger.warning(f"Style drift analysis failed: {str(e)}")
            return {}
    
    def _analyze_benchmark_attribution(self, portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series,
                                     portfolio_weights: pd.DataFrame,
                                     asset_returns: pd.DataFrame) -> Dict:
        """Analyze performance relative to benchmark."""
        
        try:
            # Align returns
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
            
            if len(aligned_portfolio) < 10:
                return {}
            
            # Active returns
            active_returns = aligned_portfolio - aligned_benchmark
            
            # Basic attribution metrics
            attribution = {
                'total_active_return': active_returns.sum(),
                'annualized_active_return': active_returns.mean() * 252,
                'tracking_error': active_returns.std() * np.sqrt(252),
                'information_ratio': (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252)) if active_returns.std() > 0 else 0,
                'hit_rate': (active_returns > 0).mean(),
                'up_capture': self._calculate_capture_ratio(aligned_portfolio, aligned_benchmark, up=True),
                'down_capture': self._calculate_capture_ratio(aligned_portfolio, aligned_benchmark, up=False)
            }
            
            # Time-varying attribution
            window = 60
            rolling_attribution = {}
            
            for i in range(window, len(active_returns)):
                date = active_returns.index[i]
                window_active = active_returns.iloc[i-window:i]
                
                rolling_attribution[date] = {
                    'active_return': window_active.mean() * 252,
                    'tracking_error': window_active.std() * np.sqrt(252),
                    'information_ratio': (window_active.mean() * 252) / (window_active.std() * np.sqrt(252)) if window_active.std() > 0 else 0
                }
            
            attribution['rolling_attribution'] = rolling_attribution
            
            return attribution
            
        except Exception as e:
            self.logger.warning(f"Benchmark attribution analysis failed: {str(e)}")
            return {}
    
    def _decompose_performance(self, portfolio_returns: pd.Series,
                             portfolio_weights: pd.DataFrame,
                             asset_returns: pd.DataFrame) -> Dict:
        """Decompose portfolio performance into components."""
        
        try:
            # Align data
            aligned_weights, aligned_returns = portfolio_weights.align(asset_returns, join='inner')
            
            if len(aligned_weights) < 10:
                return {}
            
            decomposition = {}
            
            # Asset contribution analysis
            asset_contributions = {}
            
            for i in range(1, len(aligned_weights)):
                date = aligned_weights.index[i]
                prev_weights = aligned_weights.iloc[i-1]
                current_returns = aligned_returns.iloc[i]
                
                # Calculate asset contributions to return
                contributions = {}
                for asset in aligned_returns.columns:
                    weight_col = f'weight_{asset}' if f'weight_{asset}' in prev_weights.index else asset
                    if weight_col in prev_weights.index and asset in current_returns.index:
                        contribution = prev_weights[weight_col] * current_returns[asset]
                        contributions[asset] = contribution
                
                asset_contributions[date] = contributions
            
            # Summary of asset contributions
            asset_summary = {}
            for asset in aligned_returns.columns:
                contributions = [ac.get(asset, 0) for ac in asset_contributions.values()]
                if contributions:
                    asset_summary[asset] = {
                        'total_contribution': sum(contributions),
                        'mean_contribution': np.mean(contributions),
                        'volatility_contribution': np.std(contributions),
                        'contribution_sharpe': np.mean(contributions) / np.std(contributions) if np.std(contributions) > 0 else 0
                    }
            
            decomposition = {
                'asset_contributions': asset_contributions,
                'asset_summary': asset_summary,
                'total_return_explained': sum([summary['total_contribution'] for summary in asset_summary.values()])
            }
            
            return decomposition
            
        except Exception as e:
            self.logger.warning(f"Performance decomposition failed: {str(e)}")
            return {}
    
    def _create_factor_returns(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """Create factor returns for attribution analysis."""
        
        try:
            factors = pd.DataFrame(index=asset_returns.index)
            
            # Market factor
            factors['market'] = asset_returns.mean(axis=1)
            
            # Size factor (Small minus Big)
            small_cap = [col for col in asset_returns.columns if col in ['IWM']]
            large_cap = [col for col in asset_returns.columns if col in ['SPY', 'VOO']]
            
            if small_cap and large_cap:
                factors['size'] = asset_returns[small_cap].mean(axis=1) - asset_returns[large_cap].mean(axis=1)
            
            # Value factor (Value minus Growth)
            value_assets = [col for col in asset_returns.columns if col in ['VTV']]
            growth_assets = [col for col in asset_returns.columns if col in ['VUG', 'QQQ']]
            
            if value_assets and growth_assets:
                factors['value'] = asset_returns[value_assets].mean(axis=1) - asset_returns[growth_assets].mean(axis=1)
            
            # Bond factor
            bond_assets = [col for col in asset_returns.columns if col in ['AGG', 'TLT', 'LQD']]
            if bond_assets:
                factors['bonds'] = asset_returns[bond_assets].mean(axis=1)
            
            # Commodity factor
            commodity_assets = [col for col in asset_returns.columns if col in ['GLD', 'SLV']]
            if commodity_assets:
                factors['commodities'] = asset_returns[commodity_assets].mean(axis=1)
            
            return factors.dropna()
            
        except Exception as e:
            self.logger.warning(f"Factor creation failed: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend using linear regression."""
        if len(series) < 3:
            return 0.0
        
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        model = LinearRegression()
        model.fit(x, y)
        
        return model.coef_[0]
    
    def _calculate_capture_ratio(self, portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               up: bool = True) -> float:
        """Calculate up/down capture ratio."""
        
        if up:
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
        
        if mask.sum() == 0:
            return 0.0
        
        portfolio_filtered = portfolio_returns[mask]
        benchmark_filtered = benchmark_returns[mask]
        
        if benchmark_filtered.mean() == 0:
            return 0.0
        
        return portfolio_filtered.mean() / benchmark_filtered.mean()
    
    def _analyze_regime_transitions(self, regime_series: pd.Series,
                                  portfolio_returns: pd.Series) -> Dict:
        """Analyze performance during regime transitions."""
        
        try:
            transitions = {}
            
            # Find regime changes
            regime_changes = regime_series != regime_series.shift(1)
            change_dates = regime_changes[regime_changes].index
            
            # Analyze performance around transitions
            for change_date in change_dates[1:]:  # Skip first
                try:
                    # Get returns around transition (±5 days)
                    start_date = change_date - pd.Timedelta(days=5)
                    end_date = change_date + pd.Timedelta(days=5)
                    
                    transition_returns = portfolio_returns[start_date:end_date]
                    
                    if len(transition_returns) > 0:
                        transitions[change_date] = {
                            'pre_transition_return': transition_returns[:change_date].sum() if len(transition_returns[:change_date]) > 0 else 0,
                            'post_transition_return': transition_returns[change_date:].sum() if len(transition_returns[change_date:]) > 0 else 0,
                            'total_transition_return': transition_returns.sum(),
                            'from_regime': regime_series[change_date - pd.Timedelta(days=1)] if change_date - pd.Timedelta(days=1) in regime_series.index else None,
                            'to_regime': regime_series[change_date]
                        }
                except:
                    continue
            
            return transitions
            
        except Exception as e:
            self.logger.warning(f"Regime transition analysis failed: {str(e)}")
            return {}
    
    def generate_attribution_report(self, attribution_results: Dict) -> str:
        """Generate comprehensive attribution report."""
        
        report = []
        report.append("="*80)
        report.append("ADVANCED PERFORMANCE ATTRIBUTION REPORT")
        report.append("="*80)
        
        # Regime Attribution
        if 'regime_attribution' in attribution_results:
            report.append("\n📊 REGIME-BASED ATTRIBUTION")
            report.append("-" * 40)
            
            regime_attr = attribution_results['regime_attribution']
            for regime_key, metrics in regime_attr.items():
                if regime_key != 'transitions' and isinstance(metrics, dict):
                    report.append(f"\n{regime_key.upper()}:")
                    report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
                    report.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
                    report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                    report.append(f"  Days in Regime: {metrics.get('days_in_regime', 0)}")
        
        # Factor Attribution
        if 'factor_attribution' in attribution_results:
            report.append("\n🔍 FACTOR-BASED ATTRIBUTION")
            report.append("-" * 40)
            
            factor_attr = attribution_results['factor_attribution']
            if 'alpha' in factor_attr:
                report.append(f"Alpha (Residual Return): {factor_attr['alpha']:.2%}")
                report.append(f"R-squared: {factor_attr.get('r_squared', 0):.3f}")
                report.append(f"Tracking Error: {factor_attr.get('tracking_error', 0):.2%}")
            
            if 'factor_contributions' in factor_attr:
                report.append("\nFactor Contributions:")
                for factor, contrib in factor_attr['factor_contributions'].items():
                    report.append(f"  {factor}: {contrib.get('contribution', 0):.2%} "
                                f"(loading: {contrib.get('loading', 0):.3f})")
        
        # Risk Attribution
        if 'risk_attribution' in attribution_results:
            report.append("\n⚖️ RISK CONTRIBUTION ANALYSIS")
            report.append("-" * 40)
            
            risk_attr = attribution_results['risk_attribution']
            if 'avg_concentration' in risk_attr:
                report.append(f"Average Risk Concentration: {risk_attr['avg_concentration']:.3f}")
            
            if 'summary' in risk_attr:
                report.append("\nTop Risk Contributors:")
                summary = risk_attr['summary']
                sorted_contributors = sorted(summary.items(), 
                                           key=lambda x: x[1].get('mean_contribution', 0), 
                                           reverse=True)
                
                for asset, contrib in sorted_contributors[:5]:
                    mean_contrib = contrib.get('mean_contribution', 0)
                    report.append(f"  {asset}: {mean_contrib:.1%}")
        
        # Style Analysis
        if 'style_analysis' in attribution_results:
            report.append("\n🎨 STYLE DRIFT ANALYSIS")
            report.append("-" * 40)
            
            style_attr = attribution_results['style_analysis']
            if 'style_drift_metrics' in style_attr:
                for style, metrics in style_attr['style_drift_metrics'].items():
                    mean_exp = metrics.get('mean_exposure', 0)
                    drift_vol = metrics.get('drift_volatility', 0)
                    report.append(f"{style.capitalize()}: {mean_exp:.1%} ± {drift_vol:.1%}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)