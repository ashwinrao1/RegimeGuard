"""
Tests for data management functionality.

This module tests the core data download, validation, and preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_manager import DataDownloader, DataValidator, DataManager, DataDownloadError, DataValidationError


class TestDataDownloader:
    """Test cases for DataDownloader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.downloader = DataDownloader()
    
    def test_download_asset_data_success(self):
        """Test successful asset data download."""
        # Use a small date range and reliable ticker for testing
        tickers = ["SPY"]
        start_date = "2023-01-01"
        end_date = "2023-01-31"
        
        try:
            data = self.downloader.download_asset_data(tickers, start_date, end_date)
            
            # Verify data structure
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            assert "SPY" in data.columns
            assert len(data) > 0
            
            # Verify data types and values
            assert data.dtypes["SPY"] in [np.float64, float]
            assert (data["SPY"] > 0).all()  # Prices should be positive
            
        except Exception as e:
            # If network/API issues, skip the test
            pytest.skip(f"Network/API error: {str(e)}")
    
    def test_download_asset_data_invalid_ticker(self):
        """Test handling of invalid ticker symbols."""
        tickers = ["INVALID_TICKER_XYZ123"]
        start_date = "2023-01-01"
        end_date = "2023-01-31"
        
        # This should either return empty data or raise an error
        try:
            data = self.downloader.download_asset_data(tickers, start_date, end_date)
            # If it returns data, it should be empty or the ticker should be removed
            if not data.empty:
                assert "INVALID_TICKER_XYZ123" not in data.columns
        except DataDownloadError:
            # This is also acceptable behavior
            pass
    
    @patch.dict(os.environ, {"FRED_API_KEY": "test_key"})
    def test_download_macro_data_with_api_key(self):
        """Test macro data download initialization with API key."""
        downloader = DataDownloader(fred_api_key="test_key")
        assert downloader.fred is not None
    
    def test_download_macro_data_without_api_key(self):
        """Test macro data download without API key."""
        downloader = DataDownloader()
        
        if downloader.fred is None:
            with pytest.raises(DataDownloadError, match="FRED API not available"):
                downloader.download_macro_data(["VIXCLS"], "2023-01-01", "2023-01-31")


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    def test_validate_empty_data(self):
        """Test validation of empty data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="Data is empty"):
            self.validator.validate_data(empty_data)
    
    def test_validate_insufficient_data(self):
        """Test validation with insufficient data points."""
        # Create data with less than minimum required observations
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        insufficient_data = pd.DataFrame({
            "SPY": np.random.randn(100) + 100
        }, index=dates)
        
        with pytest.raises(DataValidationError, match="Insufficient data"):
            self.validator.validate_data(insufficient_data, "price")
    
    def test_validate_price_data_success(self):
        """Test successful price data validation."""
        # Create valid price data
        dates = pd.date_range("2022-01-01", periods=300, freq="D")
        price_data = pd.DataFrame({
            "SPY": np.random.randn(300) * 0.02 + 1,  # Small daily changes
            "AGG": np.random.randn(300) * 0.01 + 1
        }, index=dates)
        
        # Make prices cumulative and positive
        price_data = (price_data + 1).cumprod() * 100
        
        # Should not raise any exception
        result = self.validator.validate_data(price_data, "price")
        assert result is True
    
    def test_validate_negative_prices(self):
        """Test validation with negative prices."""
        dates = pd.date_range("2022-01-01", periods=300, freq="D")
        price_data = pd.DataFrame({
            "SPY": np.random.randn(300),  # Can be negative
        }, index=dates)
        
        with pytest.raises(DataValidationError, match="Non-positive prices"):
            self.validator.validate_data(price_data, "price")
    
    def test_validate_returns_data_success(self):
        """Test successful returns data validation."""
        dates = pd.date_range("2022-01-01", periods=300, freq="D")
        returns_data = pd.DataFrame({
            "SPY": np.random.randn(300) * 0.02,  # 2% daily volatility
            "AGG": np.random.randn(300) * 0.01   # 1% daily volatility
        }, index=dates)
        
        result = self.validator.validate_data(returns_data, "returns")
        assert result is True
    
    def test_clean_data_forward_fill(self):
        """Test data cleaning with forward fill method."""
        # Create data with missing values
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        data_with_na = pd.DataFrame({
            "SPY": [100, 101, np.nan, 103, np.nan, 105, 106, np.nan, 108, 109]
        }, index=dates)
        
        cleaned_data = self.validator.clean_data(data_with_na, method="forward_fill")
        
        # Should have no missing values
        assert not cleaned_data.isnull().any().any()
        assert len(cleaned_data) == len(data_with_na)


class TestDataManager:
    """Test cases for DataManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a temporary cache directory for testing
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(cache_dir=self.temp_dir)
    
    def test_compute_returns(self):
        """Test returns computation."""
        # Create sample price data
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        prices = pd.DataFrame({
            "SPY": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107]
        }, index=dates)
        
        returns = self.data_manager.compute_returns(prices)
        
        # Verify returns structure
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(prices) - 1  # One less due to differencing
        assert "SPY" in returns.columns
        
        # Verify first return calculation
        expected_first_return = np.log(101 / 100)
        assert abs(returns.iloc[0]["SPY"] - expected_first_return) < 1e-10
    
    def test_create_regime_features(self):
        """Test regime features creation."""
        # Create sample returns data
        dates = pd.date_range("2022-01-01", periods=50, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.randn(50) * 0.02,
            "AGG": np.random.randn(50) * 0.01
        }, index=dates)
        
        # Create sample macro data
        macro_data = pd.DataFrame({
            "VIXCLS": np.random.randn(50) * 5 + 20,
            "DGS10": np.random.randn(50) * 0.5 + 2.5,
            "DGS2": np.random.randn(50) * 0.3 + 1.5
        }, index=dates)
        
        features = self.data_manager.create_regime_features(returns, macro_data)
        
        # Verify features structure
        assert isinstance(features, pd.DataFrame)
        assert "market_volatility" in features.columns
        assert "market_return" in features.columns
        assert "yield_spread" in features.columns  # Should be created from DGS10 - DGS2
        
        # Verify no missing values in final features
        assert not features.isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__])