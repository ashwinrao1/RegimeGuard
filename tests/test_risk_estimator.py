"""
Tests for risk estimation functionality.

This module tests covariance estimation, return estimation, validation, and storage.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk_estimator import (
    CovarianceEstimator, ReturnEstimator, RiskValidator, 
    RiskEstimator, RiskParameterStorage, RiskParameterValidator
)
from interfaces import RegimeParameters


class TestCovarianceEstimator:
    """Test cases for CovarianceEstimator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = CovarianceEstimator()
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        self.returns = pd.DataFrame({
            'Asset1': np.random.randn(100) * 0.02,
            'Asset2': np.random.randn(100) * 0.015,
            'Asset3': np.random.randn(100) * 0.025
        }, index=dates)
        
        # Create regime labels
        self.regime_labels = np.random.choice([0, 1, 2], size=100)
    
    def test_sample_covariance_estimation(self):
        """Test sample covariance estimation."""
        cov_matrix = self.estimator.estimate_sample_covariance(self.returns)
        
        # Check properties
        assert cov_matrix.shape == (3, 3)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
    
    def test_ledoit_wolf_estimation(self):
        """Test Ledoit-Wolf shrinkage estimation."""
        cov_matrix, shrinkage = self.estimator.estimate_ledoit_wolf_covariance(self.returns)
        
        # Check properties
        assert cov_matrix.shape == (3, 3)
        assert np.allclose(cov_matrix, cov_matrix.T)
        assert 0 <= shrinkage <= 1
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_custom_shrinkage(self):
        """Test custom shrinkage methods."""
        sample_cov = self.estimator.estimate_sample_covariance(self.returns)
        
        # Test different shrinkage targets
        for target in ["identity", "diagonal", "market", "constant_correlation"]:
            shrunk_cov = self.estimator.apply_custom_shrinkage(sample_cov, 0.3, target)
            
            assert shrunk_cov.shape == sample_cov.shape
            assert np.allclose(shrunk_cov, shrunk_cov.T)
            
            # Check positive semi-definiteness
            eigenvalues = np.linalg.eigvals(shrunk_cov)
            assert np.all(eigenvalues >= -1e-10)
    
    def test_regime_covariance_estimation(self):
        """Test regime-specific covariance estimation."""
        result = self.estimator.estimate_regime_covariance(self.returns, self.regime_labels, 0)
        
        # Check result structure
        assert 'covariance_matrix' in result
        assert 'regime_id' in result
        assert 'n_observations' in result
        assert 'is_valid' in result
        
        cov_matrix = result['covariance_matrix']
        assert cov_matrix.shape == (3, 3)
        assert np.allclose(cov_matrix, cov_matrix.T)
    
    def test_covariance_validation(self):
        """Test covariance matrix validation."""
        # Valid matrix
        valid_cov = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        assert self.estimator.validate_covariance_matrix(valid_cov)
        
        # Invalid matrix (not symmetric)
        invalid_cov = np.array([[1.0, 0.5, 0.3], [0.4, 1.0, 0.2], [0.3, 0.2, 1.0]])
        assert not self.estimator.validate_covariance_matrix(invalid_cov)
        
        # Invalid matrix (negative eigenvalue)
        invalid_cov2 = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert not self.estimator.validate_covariance_matrix(invalid_cov2)
    
    def test_regularization(self):
        """Test covariance matrix regularization."""
        # Create nearly singular matrix
        singular_cov = np.array([[1.0, 0.999, 0.999], [0.999, 1.0, 0.999], [0.999, 0.999, 1.0]])
        
        regularized_cov = self.estimator._regularize_covariance_matrix(singular_cov)
        
        # Check that regularization improves condition number
        original_cond = np.linalg.cond(singular_cov)
        regularized_cond = np.linalg.cond(regularized_cov)
        
        assert regularized_cond < original_cond
        assert self.estimator.validate_covariance_matrix(regularized_cov)


class TestReturnEstimator:
    """Test cases for ReturnEstimator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = ReturnEstimator()
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        self.returns = pd.DataFrame({
            'Asset1': np.random.randn(100) * 0.02 + 0.001,  # Positive drift
            'Asset2': np.random.randn(100) * 0.015,         # Zero drift
            'Asset3': np.random.randn(100) * 0.025 - 0.001  # Negative drift
        }, index=dates)
        
        self.regime_labels = np.random.choice([0, 1, 2], size=100)
    
    def test_sample_mean_estimation(self):
        """Test sample mean estimation."""
        mean_returns = self.estimator.estimate_sample_mean(self.returns)
        
        assert len(mean_returns) == 3
        assert np.isfinite(mean_returns).all()
        
        # Check that means are reasonable
        assert np.all(np.abs(mean_returns) < 0.1)  # Less than 10% daily return
    
    def test_robust_mean_estimation(self):
        """Test robust mean estimation with outlier handling."""
        # Add outliers to data
        returns_with_outliers = self.returns.copy()
        returns_with_outliers.iloc[10, 0] = 0.5  # 50% return outlier
        returns_with_outliers.iloc[20, 1] = -0.4  # -40% return outlier
        
        robust_means = self.estimator.estimate_robust_mean(returns_with_outliers)
        sample_means = self.estimator.estimate_sample_mean(returns_with_outliers)
        
        # Robust means should be less affected by outliers
        assert len(robust_means) == 3
        assert np.isfinite(robust_means).all()
        
        # For assets with outliers, robust mean should be different from sample mean
        assert abs(robust_means[0] - sample_means[0]) > 0.001
        assert abs(robust_means[1] - sample_means[1]) > 0.001
    
    def test_shrunk_mean_estimation(self):
        """Test shrunk mean estimation."""
        shrunk_means = self.estimator.estimate_shrunk_mean(self.returns, 0.0, 0.5)
        sample_means = self.estimator.estimate_sample_mean(self.returns)
        
        # Shrunk means should be closer to zero (shrinkage target)
        assert len(shrunk_means) == 3
        assert np.all(np.abs(shrunk_means) <= np.abs(sample_means))
    
    def test_capm_returns(self):
        """Test CAPM return estimation."""
        capm_returns = self.estimator.estimate_capm_returns(self.returns)
        
        assert len(capm_returns) == 3
        assert np.isfinite(capm_returns).all()
    
    def test_regime_return_estimation(self):
        """Test regime-specific return estimation."""
        result = self.estimator.estimate_regime_returns(self.returns, self.regime_labels, 0)
        
        # Check result structure
        assert 'mean_returns' in result
        assert 'regime_id' in result
        assert 'n_observations' in result
        assert 'return_volatility' in result
        assert 'confidence_intervals' in result
        
        mean_returns = result['mean_returns']
        assert len(mean_returns) == 3
        assert np.isfinite(mean_returns).all()
    
    def test_forecasting(self):
        """Test return forecasting methods."""
        # Test historical method
        forecast_result = self.estimator.forecast_returns(self.returns, method="historical")
        
        assert 'forecasted_returns' in forecast_result
        assert 'forecast_std' in forecast_result
        assert 'confidence_intervals' in forecast_result
        
        forecasted_returns = forecast_result['forecasted_returns']
        assert len(forecasted_returns) == 3
        assert np.isfinite(forecasted_returns).all()
        
        # Test AR method
        ar_forecast = self.estimator.forecast_returns(self.returns, method="ar")
        assert 'forecasted_returns' in ar_forecast
        
        # Test ensemble method
        ensemble_forecast = self.estimator.forecast_returns(self.returns, method="ensemble")
        assert 'forecasted_returns' in ensemble_forecast
    
    def test_return_attribution(self):
        """Test return attribution analysis."""
        benchmark = self.returns.mean(axis=1)  # Equal-weighted benchmark
        
        attribution = self.estimator.calculate_return_attribution(self.returns, benchmark)
        
        assert len(attribution) == 3
        for asset, metrics in attribution.items():
            assert 'alpha' in metrics
            assert 'beta' in metrics
            assert 'tracking_error' in metrics
            assert 'information_ratio' in metrics
            assert 'correlation' in metrics


class TestRiskValidator:
    """Test cases for RiskValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = RiskValidator()
        
        # Create valid regime parameters
        self.valid_params = RegimeParameters(
            regime_id=0,
            mean_returns=np.array([0.001, 0.0005, 0.0015]),
            covariance_matrix=np.array([
                [0.0004, 0.0001, 0.0002],
                [0.0001, 0.0002, 0.0001],
                [0.0002, 0.0001, 0.0006]
            ]),
            regime_probability=0.4,
            start_date=pd.Timestamp('2022-01-01'),
            end_date=pd.Timestamp('2022-12-31')
        )
    
    def test_valid_regime_parameters(self):
        """Test validation of valid regime parameters."""
        validation_result = self.validator.validate_regime_parameters(self.valid_params)
        
        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0
    
    def test_invalid_covariance_matrix(self):
        """Test validation with invalid covariance matrix."""
        # Create parameters with invalid covariance (not positive semi-definite)
        invalid_params = RegimeParameters(
            regime_id=0,
            mean_returns=np.array([0.001, 0.0005]),
            covariance_matrix=np.array([
                [1.0, 2.0],
                [2.0, 1.0]  # This matrix has negative eigenvalue
            ]),
            regime_probability=0.4,
            start_date=pd.Timestamp('2022-01-01'),
            end_date=pd.Timestamp('2022-12-31')
        )
        
        validation_result = self.validator.validate_regime_parameters(invalid_params)
        assert validation_result['valid'] is False
        assert len(validation_result['errors']) > 0
    
    def test_dimension_mismatch(self):
        """Test validation with dimension mismatch."""
        invalid_params = RegimeParameters(
            regime_id=0,
            mean_returns=np.array([0.001, 0.0005]),  # 2 assets
            covariance_matrix=np.array([
                [0.0004, 0.0001, 0.0002],
                [0.0001, 0.0002, 0.0001],
                [0.0002, 0.0001, 0.0006]
            ]),  # 3x3 matrix
            regime_probability=0.4,
            start_date=pd.Timestamp('2022-01-01'),
            end_date=pd.Timestamp('2022-12-31')
        )
        
        validation_result = self.validator.validate_regime_parameters(invalid_params)
        assert validation_result['valid'] is False


class TestRiskParameterStorage:
    """Test cases for RiskParameterStorage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        self.temp_file.close()
        
        self.storage = RiskParameterStorage(self.temp_file.name)
        
        # Create test parameters
        self.test_params = {
            0: RegimeParameters(
                regime_id=0,
                mean_returns=np.array([0.001, 0.0005, 0.0015]),
                covariance_matrix=np.array([
                    [0.0004, 0.0001, 0.0002],
                    [0.0001, 0.0002, 0.0001],
                    [0.0002, 0.0001, 0.0006]
                ]),
                regime_probability=0.4,
                start_date=pd.Timestamp('2022-01-01'),
                end_date=pd.Timestamp('2022-12-31')
            )
        }
    
    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_save_and_load_parameters(self):
        """Test saving and loading parameters."""
        # Save parameters
        success = self.storage.save_regime_parameters(self.test_params, {'test': 'metadata'})
        assert success is True
        
        # Load parameters
        loaded_params = self.storage.load_regime_parameters()
        assert loaded_params is not None
        assert len(loaded_params) == 1
        assert 0 in loaded_params
        
        # Check parameter values
        loaded_param = loaded_params[0]
        original_param = self.test_params[0]
        
        np.testing.assert_array_almost_equal(loaded_param.mean_returns, original_param.mean_returns)
        np.testing.assert_array_almost_equal(loaded_param.covariance_matrix, original_param.covariance_matrix)
        assert loaded_param.regime_probability == original_param.regime_probability
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        nonexistent_storage = RiskParameterStorage("nonexistent_file.pkl")
        loaded_params = nonexistent_storage.load_regime_parameters()
        assert loaded_params is None
    
    def test_export_to_csv(self):
        """Test exporting parameters to CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = self.storage.export_parameters_to_csv(self.test_params, temp_dir)
            
            assert 'returns' in exported_files
            assert 'summary' in exported_files
            assert 'covariance_regime_0' in exported_files
            
            # Check that files exist
            for file_path in exported_files.values():
                assert os.path.exists(file_path)


class TestRiskEstimator:
    """Test cases for main RiskEstimator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = RiskEstimator()
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=200, freq="D")
        self.returns = pd.DataFrame({
            'SPY': np.random.randn(200) * 0.02 + 0.0003,
            'AGG': np.random.randn(200) * 0.01 + 0.0001,
            'GLD': np.random.randn(200) * 0.025 + 0.0002
        }, index=dates)
        
        # Create regime labels with sufficient observations per regime
        self.regime_labels = np.concatenate([
            np.full(70, 0),   # Regime 0: 70 observations
            np.full(80, 1),   # Regime 1: 80 observations
            np.full(50, 2)    # Regime 2: 50 observations
        ])
    
    def test_estimate_regime_covariance(self):
        """Test regime covariance estimation."""
        regime_covariances = self.estimator.estimate_regime_covariance(self.returns, self.regime_labels)
        
        assert len(regime_covariances) == 3  # 3 regimes
        
        for regime_id, cov_matrix in regime_covariances.items():
            assert cov_matrix.shape == (3, 3)
            assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
            
            # Check positive semi-definiteness
            eigenvalues = np.linalg.eigvals(cov_matrix)
            assert np.all(eigenvalues >= -1e-10)
    
    def test_estimate_regime_returns(self):
        """Test regime return estimation."""
        regime_returns = self.estimator.estimate_regime_returns(self.returns, self.regime_labels)
        
        assert len(regime_returns) == 3  # 3 regimes
        
        for regime_id, mean_returns in regime_returns.items():
            assert len(mean_returns) == 3  # 3 assets
            assert np.isfinite(mean_returns).all()
    
    def test_estimate_all_regime_parameters(self):
        """Test estimation of all regime parameters."""
        regime_parameters = self.estimator.estimate_all_regime_parameters(self.returns, self.regime_labels)
        
        assert len(regime_parameters) == 3
        
        for regime_id, params in regime_parameters.items():
            assert isinstance(params, RegimeParameters)
            assert params.regime_id == regime_id
            assert len(params.mean_returns) == 3
            assert params.covariance_matrix.shape == (3, 3)
            assert 0 <= params.regime_probability <= 1
    
    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation."""
        # First estimate parameters
        regime_parameters = self.estimator.estimate_all_regime_parameters(self.returns, self.regime_labels)
        
        # Test portfolio weights (equal weight)
        weights = np.array([1/3, 1/3, 1/3])
        
        risk_metrics = self.estimator.calculate_portfolio_risk(weights, 0)
        
        assert 'portfolio_variance' in risk_metrics
        assert 'portfolio_volatility' in risk_metrics
        assert 'portfolio_return' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        
        assert risk_metrics['portfolio_variance'] >= 0
        assert risk_metrics['portfolio_volatility'] >= 0
    
    def test_parameter_storage_integration(self):
        """Test integration with parameter storage."""
        # Estimate parameters
        regime_parameters = self.estimator.estimate_all_regime_parameters(self.returns, self.regime_labels)
        
        # Test saving (use temporary file)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Update storage path
            self.estimator.parameter_storage.storage_path = temp_path
            
            # Save parameters
            success = self.estimator.save_parameters({'test': 'metadata'})
            assert success is True
            
            # Clear current parameters
            self.estimator.regime_parameters = {}
            
            # Load parameters
            success = self.estimator.load_parameters()
            assert success is True
            assert len(self.estimator.regime_parameters) == 3
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_parameter_validation_integration(self):
        """Test integration with parameter validation."""
        # Estimate parameters
        regime_parameters = self.estimator.estimate_all_regime_parameters(self.returns, self.regime_labels)
        
        # Validate parameters
        validation_results = self.estimator.validate_current_parameters()
        
        assert 'overall_valid' in validation_results
        assert 'regime_validations' in validation_results
        assert 'cross_regime_checks' in validation_results
        
        # Generate validation report
        report = self.estimator.generate_validation_report()
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_parameter_summary(self):
        """Test parameter summary generation."""
        # Estimate parameters
        regime_parameters = self.estimator.estimate_all_regime_parameters(self.returns, self.regime_labels)
        
        summary = self.estimator.get_parameter_summary()
        
        assert 'n_regimes' in summary
        assert 'regime_ids' in summary
        assert 'total_probability' in summary
        assert 'parameter_details' in summary
        
        assert summary['n_regimes'] == 3
        assert len(summary['regime_ids']) == 3
        assert abs(summary['total_probability'] - 1.0) < 0.01  # Should sum to ~1


if __name__ == "__main__":
    pytest.main([__file__])