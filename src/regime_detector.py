"""
Regime detection module for the robust portfolio optimization system.

This module implements market regime detection using clustering algorithms
to identify distinct market states (bull, bear, neutral) from historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from hmmlearn import hmm
import warnings
from datetime import datetime

from interfaces import RegimeDetectorInterface
from config import get_config
from logging_config import get_logger


class RegimeClusterer:
    """Implements clustering algorithms for regime detection."""
    
    def __init__(self, method: str = "kmeans", random_state: Optional[int] = None):
        """Initialize regime clusterer.
        
        Args:
            method: Clustering method ("kmeans" or "hmm")
            random_state: Random seed for reproducibility
        """
        self.method = method.lower()
        self.random_state = random_state
        self.logger = get_logger(__name__)
        
        # Initialize clustering model
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Regime statistics
        self.regime_centers = None
        self.regime_statistics = {}
        
    def fit(self, features: pd.DataFrame, n_regimes: int = 3) -> np.ndarray:
        """Fit clustering model and return regime labels.
        
        Args:
            features: DataFrame with regime detection features
            n_regimes: Number of regimes to detect
            
        Returns:
            Array of regime labels for each observation
        """
        self.logger.info(f"Fitting {self.method} clustering with {n_regimes} regimes")
        
        # Validate inputs
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        if n_regimes < 2 or n_regimes > 5:
            raise ValueError("Number of regimes must be between 2 and 5")
        
        # Handle missing values
        features_clean = features.dropna()
        if len(features_clean) < n_regimes * 10:
            raise ValueError(f"Insufficient data: need at least {n_regimes * 10} observations")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # Fit clustering model
        if self.method == "kmeans":
            regime_labels = self._fit_kmeans(features_scaled, n_regimes)
        elif self.method == "hmm":
            regime_labels = self._fit_hmm(features_scaled, n_regimes)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Store regime centers and statistics
        self._compute_regime_statistics(features_clean, regime_labels)
        
        # Map labels back to original DataFrame index
        full_labels = np.full(len(features), -1)  # -1 for missing data
        clean_indices = features.dropna().index
        # Convert pandas index to numpy array positions
        original_positions = np.array([features.index.get_loc(idx) for idx in clean_indices])
        full_labels[original_positions] = regime_labels
        
        self.is_fitted = True
        self.logger.info(f"Clustering completed. Regime distribution: {np.bincount(regime_labels)}")
        
        return full_labels
    
    def _fit_kmeans(self, features: np.ndarray, n_regimes: int) -> np.ndarray:
        """Fit K-means clustering model."""
        self.model = KMeans(
            n_clusters=n_regimes,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        regime_labels = self.model.fit_predict(features)
        self.regime_centers = self.model.cluster_centers_
        
        return regime_labels
    
    def _fit_hmm(self, features: np.ndarray, n_regimes: int) -> np.ndarray:
        """Fit Hidden Markov Model."""
        # Suppress convergence warnings for HMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                random_state=self.random_state,
                n_iter=100
            )
            
            # Fit model and predict states
            self.model.fit(features)
            regime_labels = self.model.predict(features)
            
            # Store regime centers (means)
            self.regime_centers = self.model.means_
        
        return regime_labels
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels for new data.
        
        Args:
            features: DataFrame with regime detection features
            
        Returns:
            Array of predicted regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle missing values
        features_clean = features.dropna()
        if features_clean.empty:
            return np.full(len(features), -1)
        
        # Standardize features using fitted scaler
        features_scaled = self.scaler.transform(features_clean)
        
        # Predict using fitted model
        if self.method == "kmeans":
            regime_labels = self.model.predict(features_scaled)
        elif self.method == "hmm":
            regime_labels = self.model.predict(features_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Map labels back to original DataFrame index
        full_labels = np.full(len(features), -1)
        clean_indices = features.dropna().index
        # Convert pandas index to numpy array positions
        original_positions = np.array([features.index.get_loc(idx) for idx in clean_indices])
        full_labels[original_positions] = regime_labels
        
        return full_labels
    
    def _compute_regime_statistics(self, features: pd.DataFrame, regime_labels: np.ndarray) -> None:
        """Compute statistics for each regime."""
        self.regime_statistics = {}
        
        for regime_id in np.unique(regime_labels):
            regime_mask = regime_labels == regime_id
            regime_features = features[regime_mask]
            
            stats = {
                'count': int(np.sum(regime_mask)),
                'frequency': float(np.mean(regime_mask)),
                'mean_features': regime_features.mean().to_dict(),
                'std_features': regime_features.std().to_dict()
            }
            
            self.regime_statistics[regime_id] = stats
    
    def get_regime_centers(self) -> Optional[np.ndarray]:
        """Get regime cluster centers."""
        return self.regime_centers
    
    def get_regime_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get regime statistics."""
        return self.regime_statistics


class FeatureEngineer:
    """Creates and preprocesses features for regime detection."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def create_regime_features(self, returns: pd.DataFrame, macro_data: pd.DataFrame, 
                             window: int = 20) -> pd.DataFrame:
        """Create comprehensive features for regime detection.
        
        Args:
            returns: DataFrame with asset returns
            macro_data: DataFrame with macroeconomic indicators
            window: Rolling window size for feature calculation
            
        Returns:
            DataFrame with regime detection features
        """
        self.logger.info(f"Creating regime features with {window}-day window")
        
        features = pd.DataFrame(index=returns.index)
        
        # Market-based features
        market_returns = returns.mean(axis=1)
        features['market_return'] = market_returns
        features['market_volatility'] = market_returns.rolling(window=window).std() * np.sqrt(252)
        
        # Cross-sectional features
        features['return_dispersion'] = returns.std(axis=1)
        features['correlation_mean'] = returns.rolling(window=window).corr().groupby(level=0).mean().mean(axis=1)
        
        # Momentum and trend features
        features['momentum_1m'] = market_returns.rolling(window=20).mean()
        features['momentum_3m'] = market_returns.rolling(window=60).mean()
        features['trend_strength'] = self._calculate_trend_strength(market_returns, window)
        
        # Volatility regime features
        features['vol_regime'] = self._calculate_volatility_regime(returns, window)
        features['vol_clustering'] = self._calculate_volatility_clustering(returns, window)
        
        # Add macro features if available
        if not macro_data.empty:
            features = self._add_macro_features(features, macro_data, window)
        
        # Remove rows with insufficient data
        features = features.dropna()
        
        self.logger.info(f"Created {len(features.columns)} features for {len(features)} observations")
        return features
    
    def _calculate_trend_strength(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def trend_slope(x):
            if len(x) < window // 2:
                return np.nan
            y = np.arange(len(x))
            return np.polyfit(y, x, 1)[0]
        
        return returns.rolling(window=window).apply(trend_slope, raw=False)
    
    def _calculate_volatility_regime(self, returns: pd.DataFrame, window: int) -> pd.Series:
        """Calculate volatility regime indicator."""
        # Calculate rolling volatility
        vol = returns.std(axis=1).rolling(window=window).mean()
        
        # Create volatility regime (high/low relative to historical)
        vol_percentile = vol.rolling(window=window*4).rank(pct=True)
        
        return vol_percentile
    
    def _calculate_volatility_clustering(self, returns: pd.DataFrame, window: int) -> pd.Series:
        """Calculate volatility clustering measure."""
        # Calculate absolute returns (proxy for volatility)
        abs_returns = returns.abs().mean(axis=1)
        
        # Calculate autocorrelation of absolute returns
        def vol_clustering(x):
            if len(x) < window // 2:
                return np.nan
            return np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        
        return abs_returns.rolling(window=window).apply(vol_clustering, raw=False)
    
    def _add_macro_features(self, features: pd.DataFrame, macro_data: pd.DataFrame, 
                           window: int) -> pd.DataFrame:
        """Add macroeconomic features to the feature set."""
        # Align macro data with features index
        macro_aligned = macro_data.reindex(features.index, method='ffill')
        
        for col in macro_aligned.columns:
            if macro_aligned[col].notna().sum() > len(features) * 0.5:  # At least 50% coverage
                # Add level
                features[f'{col}_level'] = macro_aligned[col]
                
                # Add change
                features[f'{col}_change'] = macro_aligned[col].diff()
                
                # Add rolling statistics
                features[f'{col}_ma'] = macro_aligned[col].rolling(window=window).mean()
                features[f'{col}_std'] = macro_aligned[col].rolling(window=window).std()
        
        # Create derived macro features
        if 'DGS10_level' in features.columns and 'DGS2_level' in features.columns:
            features['yield_spread'] = features['DGS10_level'] - features['DGS2_level']
            features['yield_spread_change'] = features['yield_spread'].diff()
        
        return features


class RegimeValidator:
    """Validates regime detection results and provides quality metrics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.regime_analysis_cache = {}
    
    def validate_regimes(self, features: pd.DataFrame, regime_labels: np.ndarray, 
                        min_regime_size: int = 10) -> Dict[str, Any]:
        """Validate regime detection results.
        
        Args:
            features: DataFrame with regime detection features
            regime_labels: Array of regime labels
            min_regime_size: Minimum number of observations per regime
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        self.logger.info("Validating regime detection results")
        
        # Remove missing labels
        valid_mask = regime_labels != -1
        valid_features = features[valid_mask]
        valid_labels = regime_labels[valid_mask]
        
        if len(valid_labels) == 0:
            return {'valid': False, 'reason': 'No valid regime labels'}
        
        validation_results = {
            'valid': True,
            'n_regimes': len(np.unique(valid_labels)),
            'n_observations': len(valid_labels),
            'regime_distribution': dict(zip(*np.unique(valid_labels, return_counts=True)))
        }
        
        # Check minimum regime size
        regime_counts = np.bincount(valid_labels)
        min_count = np.min(regime_counts)
        if min_count < min_regime_size:
            validation_results['valid'] = False
            validation_results['reason'] = f'Regime too small: {min_count} < {min_regime_size}'
            return validation_results
        
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(valid_features, valid_labels)
            validation_results['silhouette_score'] = float(silhouette_avg)
            
            # Calculate per-regime silhouette scores
            silhouette_samples_scores = silhouette_samples(valid_features, valid_labels)
            regime_silhouettes = {}
            for regime_id in np.unique(valid_labels):
                regime_mask = valid_labels == regime_id
                regime_silhouettes[regime_id] = float(np.mean(silhouette_samples_scores[regime_mask]))
            
            validation_results['regime_silhouettes'] = regime_silhouettes
            
        except Exception as e:
            self.logger.warning(f"Could not calculate silhouette score: {str(e)}")
            validation_results['silhouette_score'] = None
        
        # Check regime stability (minimum duration)
        regime_stability = self._check_regime_stability(valid_labels)
        validation_results.update(regime_stability)
        
        # Economic interpretation check
        economic_check = self._check_economic_interpretation(valid_features, valid_labels)
        validation_results.update(economic_check)
        
        self.logger.info(f"Validation completed. Valid: {validation_results['valid']}")
        return validation_results
    
    def _check_regime_stability(self, regime_labels: np.ndarray, min_duration: int = 5) -> Dict[str, Any]:
        """Check regime stability and transition patterns."""
        # Calculate regime durations
        regime_changes = np.diff(regime_labels) != 0
        regime_durations = []
        
        current_duration = 1
        for change in regime_changes:
            if change:
                regime_durations.append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
        regime_durations.append(current_duration)  # Add final duration
        
        avg_duration = np.mean(regime_durations)
        min_duration_observed = np.min(regime_durations)
        
        # Calculate transition matrix
        n_regimes = len(np.unique(regime_labels))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            from_regime = regime_labels[i]
            to_regime = regime_labels[i + 1]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return {
            'avg_regime_duration': float(avg_duration),
            'min_regime_duration': int(min_duration_observed),
            'regime_stable': min_duration_observed >= min_duration,
            'transition_matrix': transition_matrix.tolist(),
            'regime_persistence': float(np.mean(np.diag(transition_matrix)))
        }
    
    def _check_economic_interpretation(self, features: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Check if regimes have economically meaningful interpretation."""
        regime_profiles = {}
        
        for regime_id in np.unique(regime_labels):
            regime_mask = regime_labels == regime_id
            regime_features = features[regime_mask]
            
            profile = {
                'mean_return': float(regime_features.get('market_return', pd.Series([0])).mean()),
                'mean_volatility': float(regime_features.get('market_volatility', pd.Series([0])).mean()),
                'count': int(np.sum(regime_mask))
            }
            
            # Add VIX level if available
            vix_cols = [col for col in features.columns if 'VIX' in col.upper()]
            if vix_cols:
                profile['mean_vix'] = float(regime_features[vix_cols[0]].mean())
            
            regime_profiles[regime_id] = profile
        
        # Classify regimes based on return/volatility characteristics
        regime_interpretations = {}
        for regime_id, profile in regime_profiles.items():
            if profile['mean_return'] > 0 and profile['mean_volatility'] < np.median([p['mean_volatility'] for p in regime_profiles.values()]):
                interpretation = 'bull_market'
            elif profile['mean_return'] < 0 or profile['mean_volatility'] > np.percentile([p['mean_volatility'] for p in regime_profiles.values()], 75):
                interpretation = 'bear_market'
            else:
                interpretation = 'neutral_market'
            
            regime_interpretations[regime_id] = interpretation
        
        return {
            'regime_profiles': regime_profiles,
            'regime_interpretations': regime_interpretations,
            'economically_meaningful': len(set(regime_interpretations.values())) > 1
        }
    
    def analyze_regime_stability(self, regime_labels: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze regime stability over time with detailed statistics.
        
        Args:
            regime_labels: Array of regime labels
            dates: DatetimeIndex corresponding to regime labels
            
        Returns:
            Dictionary with comprehensive stability analysis
        """
        self.logger.info("Analyzing regime stability")
        
        # Remove missing labels
        valid_mask = regime_labels != -1
        valid_labels = regime_labels[valid_mask]
        valid_dates = dates[valid_mask]
        
        if len(valid_labels) == 0:
            return {'error': 'No valid regime labels for stability analysis'}
        
        # Calculate regime episodes (continuous periods in same regime)
        regime_episodes = self._identify_regime_episodes(valid_labels, valid_dates)
        
        # Calculate transition statistics
        transition_stats = self._calculate_transition_statistics(valid_labels, valid_dates)
        
        # Calculate regime duration statistics
        duration_stats = self._calculate_duration_statistics(regime_episodes)
        
        # Calculate regime frequency over time
        frequency_stats = self._calculate_frequency_statistics(valid_labels, valid_dates)
        
        return {
            'regime_episodes': regime_episodes,
            'transition_statistics': transition_stats,
            'duration_statistics': duration_stats,
            'frequency_statistics': frequency_stats,
            'total_transitions': len(regime_episodes) - 1,
            'analysis_period': {
                'start_date': str(valid_dates[0].date()),
                'end_date': str(valid_dates[-1].date()),
                'total_days': len(valid_dates)
            }
        }
    
    def _identify_regime_episodes(self, regime_labels: np.ndarray, dates: pd.DatetimeIndex) -> List[Dict[str, Any]]:
        """Identify continuous episodes of each regime."""
        episodes = []
        current_regime = regime_labels[0]
        episode_start = 0
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != current_regime:
                # End of current episode
                episodes.append({
                    'regime_id': int(current_regime),
                    'start_date': str(dates[episode_start].date()),
                    'end_date': str(dates[i-1].date()),
                    'duration_days': i - episode_start,
                    'start_index': episode_start,
                    'end_index': i - 1
                })
                
                # Start new episode
                current_regime = regime_labels[i]
                episode_start = i
        
        # Add final episode
        episodes.append({
            'regime_id': int(current_regime),
            'start_date': str(dates[episode_start].date()),
            'end_date': str(dates[-1].date()),
            'duration_days': len(regime_labels) - episode_start,
            'start_index': episode_start,
            'end_index': len(regime_labels) - 1
        })
        
        return episodes
    
    def _calculate_transition_statistics(self, regime_labels: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Calculate detailed transition statistics."""
        n_regimes = len(np.unique(regime_labels))
        
        # Count transitions
        transition_counts = np.zeros((n_regimes, n_regimes))
        transition_dates = []
        
        for i in range(len(regime_labels) - 1):
            from_regime = regime_labels[i]
            to_regime = regime_labels[i + 1]
            
            if from_regime != to_regime:
                transition_counts[from_regime, to_regime] += 1
                transition_dates.append({
                    'date': str(dates[i + 1].date()),
                    'from_regime': int(from_regime),
                    'to_regime': int(to_regime)
                })
        
        # Calculate transition probabilities
        transition_probs = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            regime_count = np.sum(regime_labels == i)
            if regime_count > 0:
                transition_probs[i, :] = transition_counts[i, :] / regime_count
        
        # Calculate persistence (diagonal elements)
        persistence = np.diag(transition_probs)
        
        return {
            'transition_counts': transition_counts.tolist(),
            'transition_probabilities': transition_probs.tolist(),
            'regime_persistence': persistence.tolist(),
            'avg_persistence': float(np.mean(persistence)),
            'transition_dates': transition_dates,
            'total_transitions': len(transition_dates)
        }
    
    def _calculate_duration_statistics(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate regime duration statistics."""
        regime_durations = {}
        
        # Group episodes by regime
        for episode in episodes:
            regime_id = episode['regime_id']
            duration = episode['duration_days']
            
            if regime_id not in regime_durations:
                regime_durations[regime_id] = []
            regime_durations[regime_id].append(duration)
        
        # Calculate statistics for each regime
        duration_stats = {}
        for regime_id, durations in regime_durations.items():
            duration_stats[regime_id] = {
                'count': len(durations),
                'mean_duration': float(np.mean(durations)),
                'median_duration': float(np.median(durations)),
                'std_duration': float(np.std(durations)),
                'min_duration': int(np.min(durations)),
                'max_duration': int(np.max(durations)),
                'total_days': int(np.sum(durations))
            }
        
        # Overall statistics
        all_durations = [d for durations in regime_durations.values() for d in durations]
        overall_stats = {
            'overall_mean_duration': float(np.mean(all_durations)),
            'overall_median_duration': float(np.median(all_durations)),
            'overall_std_duration': float(np.std(all_durations)),
            'total_episodes': len(all_durations)
        }
        
        return {
            'by_regime': duration_stats,
            'overall': overall_stats
        }
    
    def _calculate_frequency_statistics(self, regime_labels: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Calculate regime frequency statistics over different time periods."""
        # Overall frequency
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        total_obs = len(regime_labels)
        
        overall_freq = {}
        for regime_id, count in zip(unique_regimes, counts):
            overall_freq[int(regime_id)] = {
                'count': int(count),
                'frequency': float(count / total_obs),
                'percentage': float(count / total_obs * 100)
            }
        
        # Monthly frequency (if enough data)
        monthly_freq = {}
        if len(dates) > 60:  # At least 2 months of data
            df = pd.DataFrame({'regime': regime_labels, 'date': dates})
            df['year_month'] = df['date'].dt.to_period('M')
            
            monthly_counts = df.groupby(['year_month', 'regime']).size().unstack(fill_value=0)
            monthly_freq = monthly_counts.to_dict('index')
            
            # Convert period index to string
            monthly_freq = {str(k): v for k, v in monthly_freq.items()}
        
        # Yearly frequency (if enough data)
        yearly_freq = {}
        if len(dates) > 365:  # At least 1 year of data
            df = pd.DataFrame({'regime': regime_labels, 'date': dates})
            df['year'] = df['date'].dt.year
            
            yearly_counts = df.groupby(['year', 'regime']).size().unstack(fill_value=0)
            yearly_freq = yearly_counts.to_dict('index')
        
        return {
            'overall': overall_freq,
            'monthly': monthly_freq,
            'yearly': yearly_freq
        }
    
    def calculate_regime_quality_metrics(self, features: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for regime detection.
        
        Args:
            features: DataFrame with regime detection features
            regime_labels: Array of regime labels
            
        Returns:
            Dictionary with quality metrics
        """
        self.logger.info("Calculating regime quality metrics")
        
        # Remove missing labels
        valid_mask = regime_labels != -1
        valid_features = features[valid_mask]
        valid_labels = regime_labels[valid_mask]
        
        if len(valid_labels) == 0:
            return {'error': 'No valid regime labels for quality analysis'}
        
        quality_metrics = {}
        
        # Silhouette analysis
        try:
            silhouette_avg = silhouette_score(valid_features, valid_labels)
            silhouette_samples_scores = silhouette_samples(valid_features, valid_labels)
            
            quality_metrics['silhouette'] = {
                'average_score': float(silhouette_avg),
                'by_regime': {},
                'score_distribution': {
                    'mean': float(np.mean(silhouette_samples_scores)),
                    'std': float(np.std(silhouette_samples_scores)),
                    'min': float(np.min(silhouette_samples_scores)),
                    'max': float(np.max(silhouette_samples_scores))
                }
            }
            
            for regime_id in np.unique(valid_labels):
                regime_mask = valid_labels == regime_id
                regime_silhouettes = silhouette_samples_scores[regime_mask]
                quality_metrics['silhouette']['by_regime'][regime_id] = {
                    'mean_score': float(np.mean(regime_silhouettes)),
                    'std_score': float(np.std(regime_silhouettes)),
                    'min_score': float(np.min(regime_silhouettes)),
                    'max_score': float(np.max(regime_silhouettes))
                }
        
        except Exception as e:
            self.logger.warning(f"Could not calculate silhouette metrics: {str(e)}")
            quality_metrics['silhouette'] = {'error': str(e)}
        
        # Intra-cluster vs inter-cluster distances
        try:
            intra_inter_metrics = self._calculate_intra_inter_distances(valid_features, valid_labels)
            quality_metrics['cluster_separation'] = intra_inter_metrics
        except Exception as e:
            self.logger.warning(f"Could not calculate cluster separation metrics: {str(e)}")
            quality_metrics['cluster_separation'] = {'error': str(e)}
        
        # Feature importance for regime separation
        try:
            feature_importance = self._calculate_feature_importance(valid_features, valid_labels)
            quality_metrics['feature_importance'] = feature_importance
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {str(e)}")
            quality_metrics['feature_importance'] = {'error': str(e)}
        
        return quality_metrics
    
    def _calculate_intra_inter_distances(self, features: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate intra-cluster and inter-cluster distances."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        features_array = features.values
        unique_regimes = np.unique(regime_labels)
        
        # Calculate centroids
        centroids = {}
        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            centroids[regime_id] = np.mean(features_array[regime_mask], axis=0)
        
        # Intra-cluster distances (within regime)
        intra_distances = {}
        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_features = features_array[regime_mask]
            
            if len(regime_features) > 1:
                centroid = centroids[regime_id]
                distances = euclidean_distances(regime_features, [centroid]).flatten()
                intra_distances[regime_id] = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'max': float(np.max(distances))
                }
            else:
                intra_distances[regime_id] = {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        # Inter-cluster distances (between regimes)
        inter_distances = {}
        centroid_array = np.array([centroids[regime_id] for regime_id in unique_regimes])
        inter_dist_matrix = euclidean_distances(centroid_array)
        
        for i, regime_i in enumerate(unique_regimes):
            for j, regime_j in enumerate(unique_regimes):
                if i < j:  # Only upper triangle
                    inter_distances[f'{regime_i}_{regime_j}'] = float(inter_dist_matrix[i, j])
        
        # Overall metrics
        avg_intra = np.mean([d['mean'] for d in intra_distances.values()])
        avg_inter = np.mean(list(inter_distances.values()))
        separation_ratio = avg_inter / avg_intra if avg_intra > 0 else float('inf')
        
        return {
            'intra_cluster_distances': intra_distances,
            'inter_cluster_distances': inter_distances,
            'average_intra_distance': float(avg_intra),
            'average_inter_distance': float(avg_inter),
            'separation_ratio': float(separation_ratio)
        }
    
    def _calculate_feature_importance(self, features: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate feature importance for regime separation using ANOVA F-test."""
        from sklearn.feature_selection import f_classif
        
        try:
            f_scores, p_values = f_classif(features.values, regime_labels)
            
            feature_importance = {}
            for i, (feature_name, f_score, p_value) in enumerate(zip(features.columns, f_scores, p_values)):
                feature_importance[feature_name] = {
                    'f_score': float(f_score),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            
            # Sort by F-score
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['f_score'], reverse=True)
            
            return {
                'by_feature': feature_importance,
                'top_features': [name for name, _ in sorted_features[:10]],
                'significant_features': [name for name, metrics in feature_importance.items() if metrics['significant']]
            }
        
        except Exception as e:
            return {'error': str(e)}


class RegimeDetector(RegimeDetectorInterface):
    """Main regime detection class implementing the RegimeDetectorInterface."""
    
    def __init__(self, method: str = "kmeans", random_state: Optional[int] = None):
        """Initialize regime detector.
        
        Args:
            method: Clustering method ("kmeans" or "hmm")
            random_state: Random seed for reproducibility
        """
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.clusterer = RegimeClusterer(method=method, random_state=random_state)
        self.feature_engineer = FeatureEngineer()
        self.validator = RegimeValidator()
        
        # State
        self.regime_labels = None
        self.features = None
        self.validation_results = None
        
        self.logger.info(f"RegimeDetector initialized with method: {method}")
    
    def fit_regimes(self, features: pd.DataFrame, n_regimes: int = 3) -> np.ndarray:
        """Fit regime detection model and return regime labels."""
        self.logger.info(f"Fitting regime detection with {n_regimes} regimes")
        
        # Store features for later use
        self.features = features.copy()
        
        # Fit clustering model
        self.regime_labels = self.clusterer.fit(features, n_regimes)
        
        # Validate results
        self.validation_results = self.validator.validate_regimes(
            features, self.regime_labels, 
            min_regime_size=self.config.regime.min_regime_length
        )
        
        if not self.validation_results['valid']:
            self.logger.warning(f"Regime validation failed: {self.validation_results.get('reason', 'Unknown')}")
        
        return self.regime_labels
    
    def predict_regime(self, features: pd.DataFrame) -> int:
        """Predict regime for new data (single observation)."""
        if self.regime_labels is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Predict regime labels
        predicted_labels = self.clusterer.predict(features)
        
        # Return the most recent valid prediction
        valid_predictions = predicted_labels[predicted_labels != -1]
        if len(valid_predictions) == 0:
            return -1  # No valid prediction
        
        return int(valid_predictions[-1])
    
    def get_regime_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistical summary of detected regimes."""
        if self.regime_labels is None:
            return {}
        
        # Get basic statistics from clusterer
        stats = self.clusterer.get_regime_statistics()
        
        # Add validation metrics if available
        if self.validation_results:
            for regime_id in stats:
                if 'regime_silhouettes' in self.validation_results:
                    stats[regime_id]['silhouette_score'] = self.validation_results['regime_silhouettes'].get(regime_id, 0.0)
                
                if 'regime_profiles' in self.validation_results:
                    profile = self.validation_results['regime_profiles'].get(regime_id, {})
                    stats[regime_id].update(profile)
        
        return stats
    
    def validate_regimes(self, regime_labels: np.ndarray) -> bool:
        """Validate regime detection results."""
        if self.features is None:
            return False
        
        validation_results = self.validator.validate_regimes(self.features, regime_labels)
        return validation_results['valid']
    
    def get_validation_results(self) -> Optional[Dict[str, Any]]:
        """Get detailed validation results."""
        return self.validation_results
    
    def get_regime_labels(self) -> Optional[np.ndarray]:
        """Get the fitted regime labels."""
        return self.regime_labels
    
    def get_features(self) -> Optional[pd.DataFrame]:
        """Get the features used for regime detection."""
        return self.features
    
    def analyze_regime_stability(self, dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """Analyze regime stability over time.
        
        Args:
            dates: DatetimeIndex for regime labels. If None, uses features index.
            
        Returns:
            Dictionary with stability analysis results
        """
        if self.regime_labels is None:
            raise ValueError("Model must be fitted before stability analysis")
        
        if dates is None and self.features is not None:
            dates = self.features.index
        elif dates is None:
            raise ValueError("Dates must be provided if features index is not available")
        
        return self.validator.analyze_regime_stability(self.regime_labels, dates)
    
    def calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for regime detection.
        
        Returns:
            Dictionary with quality metrics
        """
        if self.regime_labels is None or self.features is None:
            raise ValueError("Model must be fitted before quality analysis")
        
        return self.validator.calculate_regime_quality_metrics(self.features, self.regime_labels)
    
    def get_regime_episodes(self, dates: Optional[pd.DatetimeIndex] = None) -> List[Dict[str, Any]]:
        """Get detailed information about regime episodes.
        
        Args:
            dates: DatetimeIndex for regime labels. If None, uses features index.
            
        Returns:
            List of regime episodes with start/end dates and durations
        """
        stability_analysis = self.analyze_regime_stability(dates)
        return stability_analysis.get('regime_episodes', [])
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Get regime transition probability matrix.
        
        Returns:
            Transition probability matrix or None if not available
        """
        if self.validation_results and 'transition_matrix' in self.validation_results:
            return np.array(self.validation_results['transition_matrix'])
        return None
    
    def summarize_regimes(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of regime detection results.
        
        Returns:
            Dictionary with regime summary including statistics, validation, and quality metrics
        """
        if self.regime_labels is None:
            return {'error': 'Model must be fitted before generating summary'}
        
        summary = {
            'basic_statistics': self.get_regime_statistics(),
            'validation_results': self.validation_results,
            'n_regimes': len(np.unique(self.regime_labels[self.regime_labels != -1])),
            'total_observations': len(self.regime_labels),
            'valid_observations': np.sum(self.regime_labels != -1)
        }
        
        # Add quality metrics if possible
        try:
            summary['quality_metrics'] = self.calculate_quality_metrics()
        except Exception as e:
            summary['quality_metrics'] = {'error': str(e)}
        
        # Add stability analysis if possible
        try:
            if self.features is not None:
                stability = self.analyze_regime_stability()
                summary['stability_analysis'] = {
                    'total_episodes': len(stability.get('regime_episodes', [])),
                    'avg_duration': stability.get('duration_statistics', {}).get('overall', {}).get('overall_mean_duration', 0),
                    'total_transitions': stability.get('total_transitions', 0)
                }
        except Exception as e:
            summary['stability_analysis'] = {'error': str(e)}
        
        return summary
    
    def create_visualizations(self, output_dir: str = "plots", 
                            dates: Optional[pd.DatetimeIndex] = None,
                            market_data: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Create and save all regime visualization plots.
        
        Args:
            output_dir: Directory to save plots
            dates: DatetimeIndex for regime labels
            market_data: Optional market data to overlay
            
        Returns:
            Dictionary with plot information and file paths
        """
        try:
            from regime_visualization import RegimeVisualizer
            
            visualizer = RegimeVisualizer()
            saved_plots = visualizer.save_all_plots(self, output_dir, dates, market_data)
            
            return {
                'success': True,
                'output_directory': output_dir,
                'saved_plots': saved_plots
            }
        
        except ImportError as e:
            self.logger.error(f"Could not import visualization module: {str(e)}")
            return {'success': False, 'error': 'Visualization module not available'}
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def plot_regime_timeline(self, dates: Optional[pd.DatetimeIndex] = None,
                           market_data: Optional[pd.Series] = None):
        """Create and display regime timeline plot.
        
        Args:
            dates: DatetimeIndex for regime labels
            market_data: Optional market data to overlay
            
        Returns:
            Matplotlib figure object
        """
        try:
            from regime_visualization import RegimeVisualizer
            
            if self.regime_labels is None:
                raise ValueError("Model must be fitted before plotting")
            
            if dates is None and self.features is not None:
                dates = self.features.index
            elif dates is None:
                dates = pd.date_range('2020-01-01', periods=len(self.regime_labels), freq='D')
            
            visualizer = RegimeVisualizer()
            return visualizer.plot_regime_timeline(self.regime_labels, dates, market_data)
        
        except ImportError as e:
            self.logger.error(f"Could not import visualization module: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating timeline plot: {str(e)}")
            return None