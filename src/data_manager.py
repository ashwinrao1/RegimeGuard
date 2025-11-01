"""
Data management module for the robust portfolio optimization system.

This module implements data acquisition, preprocessing, and validation functionality
for both asset price data and macroeconomic indicators.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import sqlite3
from pathlib import Path
import os

from interfaces import DataManagerInterface
from config import get_config
from logging_config import get_logger


class DataDownloadError(Exception):
    """Custom exception for data download failures."""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


class DataDownloader:
    """Handles downloading data from external APIs with error handling and retries."""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize data downloader.
        
        Args:
            fred_api_key: FRED API key. If None, will try to get from environment.
        """
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize FRED API
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            self.fred = None
            self.logger.warning("FRED API key not provided. Macroeconomic data download will be unavailable.")
    
    def download_asset_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download asset price data from Yahoo Finance.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with adjusted close prices for each ticker
            
        Raises:
            DataDownloadError: If download fails after all retries
        """
        self.logger.info(f"Downloading asset data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        for attempt in range(self.config.data.max_retries):
            try:
                # Download data using yfinance
                data = yf.download(
                    tickers=tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    timeout=self.config.data.api_timeout
                )
                
                if data.empty:
                    raise DataDownloadError("No data returned from Yahoo Finance")
                
                # Handle single ticker case (yfinance returns different structure)
                if len(tickers) == 1:
                    # For single ticker, yfinance may return different column structures
                    if 'Adj Close' in data.columns:
                        prices = data[['Adj Close']].copy()
                        prices.columns = tickers
                    elif 'Close' in data.columns:
                        # Fallback to regular close if adj close not available
                        prices = data[['Close']].copy()
                        prices.columns = tickers
                        self.logger.warning("Using Close price instead of Adj Close")
                    else:
                        # Check if data has multi-level columns
                        if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                            if 'Adj Close' in data.columns.get_level_values(0):
                                prices = data['Adj Close'].copy()
                            elif 'Close' in data.columns.get_level_values(0):
                                prices = data['Close'].copy()
                                self.logger.warning("Using Close price instead of Adj Close")
                            else:
                                raise DataDownloadError("No suitable price data found")
                        else:
                            raise DataDownloadError("No suitable price data found")
                else:
                    # Multiple tickers - extract adjusted close prices
                    if hasattr(data.columns, 'levels') and 'Adj Close' in data.columns.get_level_values(0):
                        prices = data['Adj Close'].copy()
                    elif hasattr(data.columns, 'levels') and 'Close' in data.columns.get_level_values(0):
                        prices = data['Close'].copy()
                        self.logger.warning("Using Close prices instead of Adj Close")
                    else:
                        raise DataDownloadError("No suitable price data found")
                
                # Validate data
                if prices.isnull().all().any():
                    missing_tickers = prices.columns[prices.isnull().all()].tolist()
                    self.logger.warning(f"No data available for tickers: {missing_tickers}")
                    # Remove tickers with no data
                    prices = prices.dropna(axis=1, how='all')
                
                if prices.empty:
                    raise DataDownloadError("No valid price data after cleaning")
                
                self.logger.info(f"Successfully downloaded data for {len(prices.columns)} tickers, {len(prices)} observations")
                return prices
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.data.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise DataDownloadError(f"Failed to download asset data after {self.config.data.max_retries} attempts: {str(e)}")
    
    def download_macro_data(self, series_ids: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download macroeconomic data from FRED API.
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with macroeconomic time series
            
        Raises:
            DataDownloadError: If download fails or FRED API not available
        """
        if self.fred is None:
            raise DataDownloadError("FRED API not available. Please provide FRED_API_KEY.")
        
        self.logger.info(f"Downloading macro data for {len(series_ids)} series from {start_date} to {end_date}")
        
        macro_data = pd.DataFrame()
        
        for series_id in series_ids:
            for attempt in range(self.config.data.max_retries):
                try:
                    series_data = self.fred.get_series(
                        series_id=series_id,
                        observation_start=start_date,
                        observation_end=end_date
                    )
                    
                    if series_data.empty:
                        self.logger.warning(f"No data available for series {series_id}")
                        break
                    
                    # Add to combined dataframe
                    if macro_data.empty:
                        macro_data = pd.DataFrame({series_id: series_data})
                    else:
                        macro_data[series_id] = series_data
                    
                    self.logger.debug(f"Downloaded {len(series_data)} observations for {series_id}")
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {series_id}: {str(e)}")
                    if attempt < self.config.data.max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Failed to download {series_id} after {self.config.data.max_retries} attempts")
        
        if macro_data.empty:
            raise DataDownloadError("No macroeconomic data could be downloaded")
        
        # Ensure daily frequency and forward fill missing values
        macro_data = macro_data.asfreq('D', method='ffill')
        
        self.logger.info(f"Successfully downloaded macro data: {list(macro_data.columns)}, {len(macro_data)} observations")
        return macro_data


class DataValidator:
    """Validates data quality and handles data cleaning."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_data(self, data: pd.DataFrame, data_type: str = "price") -> bool:
        """Validate data quality and completeness.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data ("price", "macro", "returns")
            
        Returns:
            True if data passes validation
            
        Raises:
            DataValidationError: If data fails critical validation checks
        """
        if data.empty:
            raise DataValidationError("Data is empty")
        
        # Check for minimum data requirements
        min_observations = 252  # At least 1 year of daily data
        if len(data) < min_observations:
            raise DataValidationError(f"Insufficient data: {len(data)} observations, minimum {min_observations} required")
        
        # Check for excessive missing values
        missing_pct = data.isnull().sum() / len(data)
        max_missing_pct = 0.1  # Allow up to 10% missing values
        
        problematic_columns = missing_pct[missing_pct > max_missing_pct]
        if not problematic_columns.empty:
            self.logger.warning(f"Columns with >10% missing values: {problematic_columns.to_dict()}")
        
        # Data type specific validations
        if data_type == "price":
            self._validate_price_data(data)
        elif data_type == "returns":
            self._validate_returns_data(data)
        elif data_type == "macro":
            self._validate_macro_data(data)
        
        self.logger.info(f"Data validation passed for {data_type} data: {data.shape}")
        return True
    
    def _validate_price_data(self, data: pd.DataFrame) -> None:
        """Validate price data specific requirements."""
        # Check for non-positive prices
        if (data <= 0).any().any():
            negative_cols = data.columns[(data <= 0).any()].tolist()
            raise DataValidationError(f"Non-positive prices found in columns: {negative_cols}")
        
        # Check for extreme price movements (>50% daily change)
        returns = data.pct_change().dropna()
        extreme_moves = (returns.abs() > 0.5).any()
        if extreme_moves.any():
            extreme_cols = returns.columns[extreme_moves].tolist()
            self.logger.warning(f"Extreme price movements detected in: {extreme_cols}")
    
    def _validate_returns_data(self, data: pd.DataFrame) -> None:
        """Validate returns data specific requirements."""
        # Check for extreme returns (>100% daily)
        extreme_returns = (data.abs() > 1.0).any()
        if extreme_returns.any():
            extreme_cols = data.columns[extreme_returns].tolist()
            self.logger.warning(f"Extreme returns (>100%) detected in: {extreme_cols}")
        
        # Check for constant returns (no variation)
        constant_returns = (data.std() == 0).any()
        if constant_returns.any():
            constant_cols = data.columns[constant_returns].tolist()
            raise DataValidationError(f"Constant returns (no variation) in: {constant_cols}")
    
    def _validate_macro_data(self, data: pd.DataFrame) -> None:
        """Validate macroeconomic data specific requirements."""
        # Check for reasonable ranges for common macro indicators
        for col in data.columns:
            if 'VIX' in col.upper():
                # VIX should be between 5 and 100
                if (data[col] < 0).any() or (data[col] > 200).any():
                    self.logger.warning(f"Unusual VIX values in {col}")
            elif 'DGS' in col.upper():  # Treasury yields
                # Yields should be between -5% and 20%
                if (data[col] < -5).any() or (data[col] > 20).any():
                    self.logger.warning(f"Unusual yield values in {col}")
    
    def clean_data(self, data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """Clean data by handling missing values and outliers.
        
        Args:
            data: DataFrame to clean
            method: Cleaning method ("forward_fill", "interpolate", "drop")
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        if method == "forward_fill":
            cleaned_data = cleaned_data.fillna(method='ffill')
            # Backward fill any remaining NaNs at the beginning
            cleaned_data = cleaned_data.fillna(method='bfill')
        elif method == "interpolate":
            cleaned_data = cleaned_data.interpolate(method='linear')
        elif method == "drop":
            cleaned_data = cleaned_data.dropna()
        
        # Remove any remaining NaN values
        initial_shape = cleaned_data.shape
        cleaned_data = cleaned_data.dropna()
        
        if cleaned_data.shape != initial_shape:
            self.logger.info(f"Data shape changed from {initial_shape} to {cleaned_data.shape} after cleaning")
        
        return cleaned_data


class DataManager(DataManagerInterface):
    """Main data management class implementing the DataManagerInterface."""
    
    def __init__(self, fred_api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize data manager.
        
        Args:
            fred_api_key: FRED API key for macroeconomic data
            cache_dir: Directory for data caching
        """
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        self.downloader = DataDownloader(fred_api_key)
        self.validator = DataValidator()
        
        # Set up caching directory
        self.cache_dir = Path(cache_dir or self.config.data.data_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DataManager initialized successfully")
    
    def download_asset_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download asset price data with caching and validation."""
        # Check cache first
        cache_key = f"asset_data_{'_'.join(sorted(tickers))}_{start_date}_{end_date}"
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            self.logger.info(f"Loaded asset data from cache: {cache_key}")
            return cached_data
        
        # Download fresh data
        data = self.downloader.download_asset_data(tickers, start_date, end_date)
        
        # Validate data
        self.validator.validate_data(data, "price")
        
        # Cache the data
        self._save_to_cache(data, cache_key)
        
        return data
    
    def download_macro_data(self, series_ids: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download macroeconomic data with caching and validation."""
        # Check cache first
        cache_key = f"macro_data_{'_'.join(sorted(series_ids))}_{start_date}_{end_date}"
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            self.logger.info(f"Loaded macro data from cache: {cache_key}")
            return cached_data
        
        # Download fresh data
        data = self.downloader.download_macro_data(series_ids, start_date, end_date)
        
        # Validate data
        self.validator.validate_data(data, "macro")
        
        # Cache the data
        self._save_to_cache(data, cache_key)
        
        return data
    
    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns from price data."""
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Validate returns
        self.validator.validate_data(returns, "returns")
        
        return returns
    
    def create_regime_features(self, returns: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime detection."""
        features = pd.DataFrame(index=returns.index)
        
        # Rolling volatility (annualized)
        window = self.config.regime.feature_window
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        features['market_volatility'] = rolling_vol.mean(axis=1)
        
        # Market returns (equal-weighted)
        features['market_return'] = returns.mean(axis=1)
        
        # Add macro features (align dates and forward fill)
        macro_aligned = macro_data.reindex(returns.index, method='ffill')
        
        for col in macro_aligned.columns:
            if not macro_aligned[col].isnull().all():
                features[col] = macro_aligned[col]
        
        # Create additional derived features
        if 'DGS10' in features.columns and 'DGS2' in features.columns:
            features['yield_spread'] = features['DGS10'] - features['DGS2']
        
        # Remove rows with missing values
        features = features.dropna()
        
        self.logger.info(f"Created regime features: {list(features.columns)}, {len(features)} observations")
        return features
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data using the validator."""
        return self.validator.validate_data(data)
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired (24 hours for now)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > 24 * 3600:  # 24 hours
                self.logger.debug(f"Cache expired for {cache_key}")
                return None
            
            data = pd.read_pickle(cache_file)
            return data
            
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_key}: {str(e)}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            data.to_pickle(cache_file)
            self.logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {str(e)}")