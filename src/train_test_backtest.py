"""
Train/Test Split Backtesting Implementation

This module implements the train/test split backtesting functionality that trains
the model on data up to 2023 and tests on 2024-current with $1M simulated capital.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import warnings

from interfaces import TrainTestResult
from data_manager import DataManager
from regime_detector import RegimeDetector
from risk_estimator import RiskEstimator
from robust_optimizer import RobustOptimizer
from backtest_engine import BacktestEngine, PerformanceCalculator
from config import get_config
from logging_config import get_logger


class TrainTestBacktester:
    """Implements train/test split backtesting with simulated capital tracking."""
    
    def __init__(self):
        """Initialize the train/test backtester."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.data_manager = DataManager()
        self.regime_detector = RegimeDetector()
        self.risk_estimator = RiskEstimator()
        self.robust_optimizer = RobustOptimizer()
        self.performance_calculator = PerformanceCalculator()
        
        self.logger.info("TrainTestBacktester initialized")
    
    def _get_comprehensive_ticker_universe(self) -> List[str]:
        """Get a comprehensive universe of tickers across multiple asset classes and sectors."""
        
        # Large Cap US Stocks (S&P 500 representation)
        large_cap_stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'INTU', 'AMAT', 'MU', 'LRCX',
            
            # Healthcare & Biotech
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY', 'MRK',
            'AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB', 'ILMN', 'ISRG', 'SYK', 'BSX', 'MDT',
            
            # Financial Services
            'BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB', 'PNC',
            'TFC', 'COF', 'SCHW', 'BLK', 'SPGI', 'ICE', 'CME', 'AON', 'MMC', 'AJG',
            
            # Consumer Discretionary
            'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'GM', 'F',
            'TSLA', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX', 'PYPL', 'EBAY', 'ETSY', 'ROKU',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY',
            'MDLZ', 'CPB', 'CAG', 'SJM', 'CHD', 'CLX', 'TSN', 'HRL', 'MKC', 'ADM',
            
            # Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
            'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'FDX', 'WM', 'RSG', 'PCAR',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'BKR',
            'HAL', 'DVN', 'FANG', 'APA', 'EQT', 'CTRA', 'MRO', 'HES', 'KMI', 'OKE',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'NUE',
            'STLD', 'VMC', 'MLM', 'PKG', 'IP', 'CF', 'MOS', 'FMC', 'LYB', 'CE',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED',
            'EIX', 'WEC', 'AWK', 'DTE', 'ES', 'FE', 'AEE', 'CMS', 'NI', 'LNT',
            
            # Real Estate (REITs)
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'SPG', 'O', 'SBAC', 'DLR',
            'AVB', 'EQR', 'BXP', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'FRT', 'REG'
        ]
        
        # ETFs for broader exposure and international diversification
        etfs = [
            # US Market ETFs
            'SPY',   # S&P 500
            'QQQ',   # NASDAQ 100
            'IWM',   # Russell 2000 (Small Cap)
            'VTI',   # Total Stock Market
            'VOO',   # S&P 500 (Vanguard)
            'VEA',   # Developed Markets ex-US
            'VWO',   # Emerging Markets
            
            # Sector ETFs
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
            
            # Bond ETFs
            'AGG',   # Aggregate Bond
            'TLT',   # 20+ Year Treasury
            'IEF',   # 7-10 Year Treasury
            'SHY',   # 1-3 Year Treasury
            'TIP',   # TIPS (Inflation Protected)
            'LQD',   # Investment Grade Corporate
            'HYG',   # High Yield Corporate
            'EMB',   # Emerging Market Bonds
            'MUB',   # Municipal Bonds
            'VCIT',  # Intermediate Corporate
            
            # Commodity ETFs
            'GLD',   # Gold
            'SLV',   # Silver
            'DBC',   # Commodities
            'USO',   # Oil
            'UNG',   # Natural Gas
            'PDBC',  # Commodities (Invesco)
            'IAU',   # Gold (iShares)
            'PPLT',  # Platinum
            'PALL',  # Palladium
            'CORN',  # Corn
            
            # International ETFs
            'EFA',   # EAFE (Europe, Australasia, Far East)
            'EEM',   # Emerging Markets
            'FXI',   # China Large Cap
            'EWJ',   # Japan
            'EWG',   # Germany
            'EWU',   # United Kingdom
            'EWZ',   # Brazil
            'INDA',  # India
            'RSX',   # Russia (if available)
            'EWY',   # South Korea
            
            # Alternative/Specialty ETFs
            'VNQ',   # REITs
            'VTEB',  # Tax-Exempt Bonds
            'VGIT',  # Intermediate Treasury
            'VGSH',  # Short Treasury
            'BND',   # Total Bond Market
            'BNDX',  # International Bonds
            'VMOT',  # Multisector Bond
            'VCSH',  # Short Corporate
            'VCLT',  # Long Corporate
            'VGLT'   # Long Treasury
        ]
        
        # Combine all tickers and remove duplicates
        all_tickers = list(set(large_cap_stocks + etfs))
        
        # Sort for consistency
        all_tickers.sort()
        
        self.logger.info(f"Using comprehensive ticker universe with {len(all_tickers)} assets")
        self.logger.info(f"Asset breakdown: {len(large_cap_stocks)} individual stocks, {len(etfs)} ETFs")
        
        return all_tickers
    
    def _map_regimes_to_daily_frequency(self, regime_labels: np.ndarray, 
                                       features: pd.DataFrame, 
                                       returns: pd.DataFrame) -> np.ndarray:
        """Map monthly regime labels to daily returns frequency."""
        
        # Create a mapping from feature dates to regime labels
        feature_dates = features.index
        regime_mapping = dict(zip(feature_dates, regime_labels))
        
        # For each return date, find the most recent regime
        daily_regimes = []
        
        for return_date in returns.index:
            # Find the most recent feature date that's <= return_date
            valid_feature_dates = feature_dates[feature_dates <= return_date]
            
            if len(valid_feature_dates) > 0:
                most_recent_feature_date = valid_feature_dates[-1]
                regime = regime_mapping[most_recent_feature_date]
            else:
                # If no prior feature date, use the first regime
                regime = regime_labels[0]
            
            daily_regimes.append(regime)
        
        daily_regime_array = np.array(daily_regimes)
        
        self.logger.info(f"Mapped {len(regime_labels)} monthly regimes to {len(daily_regime_array)} daily observations")
        
        # Log regime distribution
        unique_regimes, counts = np.unique(daily_regime_array, return_counts=True)
        for regime_id, count in zip(unique_regimes, counts):
            pct = count / len(daily_regime_array) * 100
            self.logger.info(f"  Regime {regime_id}: {count} days ({pct:.1f}%)")
        
        return daily_regime_array
    
    def run_train_test_backtest(self, 
                               train_end_date: str = "2023-12-31",
                               initial_capital: float = 1000000.0,
                               tickers: Optional[List[str]] = None) -> TrainTestResult:
        """Run complete train/test split backtesting.
        
        Args:
            train_end_date: End date for training period
            initial_capital: Initial capital amount
            tickers: List of asset tickers (uses default if None)
            
        Returns:
            TrainTestResult with complete performance tracking
        """
        self.logger.info(f"Starting train/test backtest with ${initial_capital:,.0f} initial capital")
        self.logger.info(f"Training period ends: {train_end_date}")
        
        # Use default tickers if none provided
        if tickers is None:
            tickers = self._get_comprehensive_ticker_universe()
        
        try:
            # Step 1: Download and prepare data
            self.logger.info("Downloading and preparing data...")
            returns_data, macro_data = self._prepare_data(tickers, train_end_date)
            
            # Step 2: Split data into train/test
            train_data, test_data = self._split_data(returns_data, macro_data, train_end_date)
            
            # Step 3: Train model on training data
            self.logger.info("Training model on historical data...")
            trained_model = self._train_model(train_data)
            
            # Step 4: Run test period simulation
            self.logger.info("Running test period simulation...")
            test_results = self._simulate_test_period(trained_model, test_data, initial_capital)
            
            # Step 5: Calculate performance metrics
            performance_summary = self._calculate_performance_summary(test_results, initial_capital)
            
            # Create result object
            result = TrainTestResult(
                train_period_end=pd.to_datetime(train_end_date),
                test_period_start=test_data['returns'].index[0],
                initial_capital=initial_capital,
                final_portfolio_value=test_results['final_value'],
                monthly_portfolio_values=test_results['monthly_values'],
                test_period_returns=test_results['returns'],
                regime_allocations=test_results['allocations'],
                performance_summary=performance_summary
            )
            
            self.logger.info(f"Train/test backtest completed successfully")
            self.logger.info(f"Final portfolio value: ${result.final_portfolio_value:,.2f}")
            self.logger.info(f"Total return: {((result.final_portfolio_value / initial_capital) - 1) * 100:.2f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Train/test backtest failed: {str(e)}")
            raise
    
    def _prepare_data(self, tickers: List[str], train_end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download and prepare asset and macro data."""
        # Calculate start date (need enough history for regime detection)
        train_end = pd.to_datetime(train_end_date)
        start_date = train_end - timedelta(days=5*365)  # 5 years of history
        
        # Get current date for end of test period
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download asset data in batches to avoid filename length issues
        self.logger.info(f"Downloading asset data for {len(tickers)} tickers")
        try:
            # Download in smaller batches to avoid cache filename issues
            batch_size = 50
            all_asset_data = []
            
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i+batch_size]
                self.logger.info(f"Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {len(batch_tickers)} tickers")
                
                batch_data = self.data_manager.download_asset_data(
                    tickers=batch_tickers,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=current_date
                )
                all_asset_data.append(batch_data)
            
            # Combine all batches
            asset_data = pd.concat(all_asset_data, axis=1)
            self.logger.info(f"Combined data from {len(all_asset_data)} batches: {asset_data.shape}")
            
            # Filter out assets with insufficient data
            min_observations = 252 * 2  # At least 2 years of data
            valid_assets = []
            
            for ticker in asset_data.columns:
                valid_data_points = asset_data[ticker].dropna()
                if len(valid_data_points) >= min_observations:
                    valid_assets.append(ticker)
                else:
                    self.logger.warning(f"Excluding {ticker}: only {len(valid_data_points)} valid observations")
            
            if len(valid_assets) < 10:
                raise ValueError(f"Insufficient valid assets: only {len(valid_assets)} assets have enough data")
            
            # Keep only valid assets
            asset_data = asset_data[valid_assets]
            self.logger.info(f"Using {len(valid_assets)} assets with sufficient data")
            
        except Exception as e:
            self.logger.error(f"Failed to download comprehensive data: {str(e)}")
            self.logger.info("Falling back to core ETF universe...")
            # Fallback to a smaller, more reliable set
            fallback_tickers = [
                'SPY', 'QQQ', 'IWM', 'VEA', 'VWO',  # Equity
                'AGG', 'TLT', 'LQD', 'HYG', 'TIP',  # Bonds
                'GLD', 'SLV', 'DBC', 'VNQ',         # Alternatives
                'XLK', 'XLF', 'XLV', 'XLE', 'XLI'   # Sectors
            ]
            asset_data = self.data_manager.download_asset_data(
                tickers=fallback_tickers,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=current_date
            )
        
        # Calculate returns
        returns_data = self.data_manager.compute_returns(asset_data)
        
        # Download macro data
        self.logger.info("Downloading macroeconomic data")
        macro_series = ['VIXCLS', 'DGS10', 'DGS2', 'UNRATE']
        macro_data = self.data_manager.download_macro_data(
            series_ids=macro_series,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=current_date
        )
        
        return returns_data, macro_data
    
    def _split_data(self, returns_data: pd.DataFrame, macro_data: pd.DataFrame, 
                   train_end_date: str) -> Tuple[Dict, Dict]:
        """Split data into training and testing periods."""
        train_end = pd.to_datetime(train_end_date)
        test_start = train_end + timedelta(days=1)
        
        # Split returns data
        train_returns = returns_data[returns_data.index <= train_end]
        test_returns = returns_data[returns_data.index >= test_start]
        
        # Split macro data
        train_macro = macro_data[macro_data.index <= train_end]
        test_macro = macro_data[macro_data.index >= test_start]
        
        # Create regime features for training
        train_features = self.data_manager.create_regime_features(train_returns, train_macro)
        test_features = self.data_manager.create_regime_features(test_returns, test_macro)
        
        train_data = {
            'returns': train_returns,
            'macro': train_macro,
            'features': train_features
        }
        
        test_data = {
            'returns': test_returns,
            'macro': test_macro,
            'features': test_features
        }
        
        self.logger.info(f"Training period: {train_returns.index[0].date()} to {train_returns.index[-1].date()}")
        self.logger.info(f"Testing period: {test_returns.index[0].date()} to {test_returns.index[-1].date()}")
        
        return train_data, test_data
    
    def _train_model(self, train_data: Dict) -> Dict:
        """Train the regime detection and optimization model."""
        # Detect regimes on training data
        self.logger.info("Detecting market regimes...")
        regime_labels = self.regime_detector.fit_regimes(train_data['features'], n_regimes=3)
        
        # Map regime labels to daily returns frequency
        self.logger.info("Mapping regime labels to daily frequency...")
        daily_regime_labels = self._map_regimes_to_daily_frequency(
            regime_labels, train_data['features'], train_data['returns']
        )
        
        # Estimate regime-specific parameters
        self.logger.info("Estimating regime-specific risk parameters...")
        regime_covariances = self.risk_estimator.estimate_regime_covariance(
            train_data['returns'], daily_regime_labels
        )
        regime_returns = self.risk_estimator.estimate_regime_returns(
            train_data['returns'], daily_regime_labels
        )
        
        # Calculate regime probabilities
        unique_regimes, regime_counts = np.unique(daily_regime_labels, return_counts=True)
        regime_probs = regime_counts / len(daily_regime_labels)
        
        trained_model = {
            'regime_detector': self.regime_detector,
            'regime_covariances': regime_covariances,
            'regime_returns': regime_returns,
            'regime_probabilities': dict(zip(unique_regimes, regime_probs)),
            'asset_names': train_data['returns'].columns.tolist()
        }
        
        self.logger.info(f"Model trained with {len(unique_regimes)} regimes")
        for regime_id, prob in trained_model['regime_probabilities'].items():
            self.logger.info(f"  Regime {regime_id}: {prob:.1%} of training period")
        
        return trained_model
    
    def _simulate_test_period(self, trained_model: Dict, test_data: Dict, 
                            initial_capital: float) -> Dict:
        """Simulate portfolio performance during test period."""
        test_returns = test_data['returns']
        test_features = test_data['features']
        
        if test_returns.empty:
            raise ValueError("No test period data available")
        
        # Initialize tracking variables
        portfolio_values = [initial_capital]
        portfolio_returns = []
        monthly_values = []
        allocations = []
        
        current_value = initial_capital
        current_weights = None
        
        # Get monthly rebalancing dates
        rebalance_dates = self._get_monthly_rebalance_dates(test_returns.index)
        
        self.logger.info(f"Simulating {len(rebalance_dates)} rebalancing periods")
        
        for i, rebalance_date in enumerate(rebalance_dates):
            try:
                # Get data up to rebalance date
                available_features = test_features[test_features.index <= rebalance_date]
                
                if available_features.empty:
                    continue
                
                # Predict current regime
                current_regime = trained_model['regime_detector'].predict_regime(
                    available_features.tail(1)
                )
                
                # Get optimal weights using robust optimization
                new_weights = self._get_optimal_weights(trained_model, current_regime)
                
                # Calculate period return if we have previous weights
                if current_weights is not None and i > 0:
                    # Get returns for the period
                    prev_date = rebalance_dates[i-1]
                    period_returns = test_returns[
                        (test_returns.index > prev_date) & 
                        (test_returns.index <= rebalance_date)
                    ]
                    
                    if not period_returns.empty:
                        # Calculate portfolio returns for the period
                        period_portfolio_returns = (period_returns * current_weights).sum(axis=1)
                        
                        # Update portfolio value
                        for daily_return in period_portfolio_returns:
                            current_value *= (1 + daily_return)
                            portfolio_values.append(current_value)
                            portfolio_returns.append(daily_return)
                
                # Update weights
                current_weights = new_weights
                
                # Record allocation
                allocation_record = {
                    'date': rebalance_date,
                    'regime': current_regime,
                    'portfolio_value': current_value
                }
                
                for j, asset in enumerate(trained_model['asset_names']):
                    allocation_record[f'weight_{asset}'] = new_weights[j]
                
                allocations.append(allocation_record)
                monthly_values.append(current_value)
                
                self.logger.debug(f"Rebalance {i+1}/{len(rebalance_dates)}: "
                                f"Regime {current_regime}, Value ${current_value:,.0f}")
                
            except Exception as e:
                self.logger.warning(f"Error at rebalance date {rebalance_date}: {str(e)}")
                continue
        
        # Handle any remaining period
        if current_weights is not None and len(rebalance_dates) > 0:
            last_rebalance = rebalance_dates[-1]
            remaining_returns = test_returns[test_returns.index > last_rebalance]
            
            if not remaining_returns.empty:
                remaining_portfolio_returns = (remaining_returns * current_weights).sum(axis=1)
                for daily_return in remaining_portfolio_returns:
                    current_value *= (1 + daily_return)
                    portfolio_values.append(current_value)
                    portfolio_returns.append(daily_return)
        
        # Create DataFrames
        allocations_df = pd.DataFrame(allocations)
        if not allocations_df.empty:
            allocations_df.set_index('date', inplace=True)
        
        monthly_values_series = pd.Series(
            monthly_values,
            index=rebalance_dates[:len(monthly_values)]
        )
        
        portfolio_returns_series = pd.Series(
            portfolio_returns,
            index=test_returns.index[:len(portfolio_returns)]
        )
        
        return {
            'final_value': current_value,
            'monthly_values': monthly_values_series,
            'returns': portfolio_returns_series,
            'allocations': allocations_df,
            'portfolio_values': portfolio_values
        }
    
    def _get_monthly_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Get monthly rebalancing dates from the test period."""
        if date_index.empty:
            return []
        
        # Get month-end dates
        monthly_dates = []
        current_month = None
        
        for date in date_index:
            if current_month is None or date.month != current_month:
                if current_month is not None:
                    # Add the last trading day of the previous month
                    monthly_dates.append(prev_date)
                current_month = date.month
            prev_date = date
        
        # Add the last date
        if date_index[-1] not in monthly_dates:
            monthly_dates.append(date_index[-1])
        
        return monthly_dates
    
    def _get_optimal_weights(self, trained_model: Dict, current_regime: int) -> np.ndarray:
        """Get optimal portfolio weights using robust optimization."""
        try:
            # Use worst-case optimization across all regimes
            regime_covariances = trained_model['regime_covariances']
            
            # Set up constraints appropriate for large universe
            n_assets = len(trained_model['asset_names'])
            max_individual_weight = min(0.15, 5.0 / n_assets)  # Max 15% or 5/N, whichever is smaller
            
            # Create constraint manager
            from robust_optimizer import ConstraintManager
            constraint_manager = ConstraintManager()
            
            # Add constraints
            constraint_manager.add_constraint("budget", {"target": 1.0})  # Full investment
            constraint_manager.add_constraint("long_only", {})  # No short selling
            constraint_manager.add_constraint("box", {
                "min_weight": 0.001, 
                "max_weight": max_individual_weight
            })  # Position limits
            
            # Solve optimization
            result = self.robust_optimizer.worst_case_optimizer.optimize(
                regime_covariances, constraint_manager
            )
            
            optimal_weights = result.weights
            
            # Check if optimization was successful
            if result.solver_status != "optimal" or not result.constraints_satisfied:
                self.logger.warning(f"Optimization failed (status: {result.solver_status}), using equal weights")
                optimal_weights = np.ones(len(trained_model['asset_names'])) / len(trained_model['asset_names'])
            elif optimal_weights is None or len(optimal_weights) != len(trained_model['asset_names']):
                self.logger.warning("Invalid optimization result, using equal weights")
                optimal_weights = np.ones(len(trained_model['asset_names'])) / len(trained_model['asset_names'])
            else:
                # Normalize weights
                if np.sum(optimal_weights) > 0:
                    optimal_weights = optimal_weights / np.sum(optimal_weights)
                else:
                    optimal_weights = np.ones(len(trained_model['asset_names'])) / len(trained_model['asset_names'])
                
                self.logger.info(f"Optimization successful! Objective value: {result.objective_value:.6f}")
            
            return optimal_weights
            
        except Exception as e:
            self.logger.warning(f"Optimization error: {str(e)}, using equal weights")
            return np.ones(len(trained_model['asset_names'])) / len(trained_model['asset_names'])
    
    def _calculate_performance_summary(self, test_results: Dict, initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance summary."""
        if test_results['returns'].empty:
            return {'error': 'No returns data for performance calculation'}
        
        # Use performance calculator
        performance_metrics = self.performance_calculator.calculate_performance_metrics(
            test_results['returns']
        )
        
        # Add additional metrics specific to train/test
        final_value = test_results['final_value']
        total_return = (final_value / initial_capital) - 1
        
        # Calculate time-based metrics
        test_days = len(test_results['returns'])
        test_years = test_days / 252.0
        
        performance_metrics.update({
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'test_period_days': test_days,
            'test_period_years': test_years,
            'annualized_return_actual': (final_value / initial_capital) ** (1/test_years) - 1 if test_years > 0 else 0
        })
        
        return performance_metrics


def add_train_test_method_to_backtest_engine():
    """Add the train/test method to the existing BacktestEngine class."""
    
    def run_train_test_backtest(self, train_end_date: str = "2023-12-31", 
                               initial_capital: float = 1000000.0) -> TrainTestResult:
        """Run train/test split backtesting with simulated capital."""
        backtester = TrainTestBacktester()
        return backtester.run_train_test_backtest(train_end_date, initial_capital)
    
    # Add method to BacktestEngine class
    BacktestEngine.run_train_test_backtest = run_train_test_backtest