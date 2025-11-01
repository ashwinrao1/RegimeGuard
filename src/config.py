"""
Configuration management for the robust portfolio optimization system.

This module provides centralized configuration management with support for
environment variables, configuration files, and default values.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data management."""
    default_tickers: List[str]
    macro_series: List[str]
    start_date: str
    lookback_years: int
    data_cache_dir: str
    api_timeout: int
    max_retries: int


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    n_regimes: int
    clustering_method: str
    feature_window: int
    min_regime_length: int
    validation_method: str


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    optimization_method: str
    max_weight: float
    min_weight: float
    transaction_cost_bps: float
    rebalance_frequency: str
    solver_timeout: int


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    estimation_window: int
    min_history: int
    benchmark_tickers: List[str]
    performance_metrics: List[str]


@dataclass
class SystemConfig:
    """Main system configuration."""
    data: DataConfig
    regime: RegimeConfig
    optimization: OptimizationConfig
    backtest: BacktestConfig
    log_level: str
    random_seed: int
    n_jobs: int


class ConfigManager:
    """Manages system configuration with multiple sources."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file and environment variables."""
        # Start with default configuration
        config_dict = self._get_default_config()
        
        # Override with file configuration if provided
        if self.config_file and os.path.exists(self.config_file):
            file_config = self._load_config_file(self.config_file)
            config_dict = self._merge_configs(config_dict, file_config)
        
        # Override with environment variables
        env_config = self._load_env_config()
        config_dict = self._merge_configs(config_dict, env_config)
        
        return self._dict_to_config(config_dict)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "data": {
                "default_tickers": ["SPY", "AGG", "GLD", "XLE", "XLF", "XLK"],
                "macro_series": ["VIXCLS", "DGS10", "DGS2", "UNRATE"],
                "start_date": "2010-01-01",
                "lookback_years": 10,
                "data_cache_dir": "data/cache",
                "api_timeout": 30,
                "max_retries": 3
            },
            "regime": {
                "n_regimes": 3,
                "clustering_method": "kmeans",
                "feature_window": 20,
                "min_regime_length": 5,
                "validation_method": "silhouette"
            },
            "optimization": {
                "optimization_method": "worst_case",
                "max_weight": 0.4,
                "min_weight": 0.0,
                "transaction_cost_bps": 5.0,
                "rebalance_frequency": "monthly",
                "solver_timeout": 60
            },
            "backtest": {
                "estimation_window": 252,
                "min_history": 504,
                "benchmark_tickers": ["SPY", "AGG"],
                "performance_metrics": ["return", "volatility", "sharpe", "max_drawdown"]
            },
            "log_level": "INFO",
            "random_seed": 42,
            "n_jobs": -1
        }
    
    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file."""
        file_path = Path(config_file)
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            "RPO_LOG_LEVEL": ("log_level",),
            "RPO_RANDOM_SEED": ("random_seed",),
            "RPO_N_JOBS": ("n_jobs",),
            "RPO_DATA_CACHE_DIR": ("data", "data_cache_dir"),
            "RPO_N_REGIMES": ("regime", "n_regimes"),
            "RPO_MAX_WEIGHT": ("optimization", "max_weight"),
            "RPO_REBALANCE_FREQ": ("optimization", "rebalance_frequency")
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert string values to appropriate types
                if config_path[-1] in ["random_seed", "n_jobs", "n_regimes"]:
                    value = int(value)
                elif config_path[-1] in ["max_weight"]:
                    value = float(value)
                
                # Set nested configuration value
                current = env_config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
        
        return env_config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig dataclass."""
        return SystemConfig(
            data=DataConfig(**config_dict["data"]),
            regime=RegimeConfig(**config_dict["regime"]),
            optimization=OptimizationConfig(**config_dict["optimization"]),
            backtest=BacktestConfig(**config_dict["backtest"]),
            log_level=config_dict["log_level"],
            random_seed=config_dict["random_seed"],
            n_jobs=config_dict["n_jobs"]
        )
    
    @property
    def config(self) -> SystemConfig:
        """Get current configuration."""
        return self._config
    
    def save_config(self, output_file: str) -> None:
        """Save current configuration to file."""
        config_dict = asdict(self._config)
        
        file_path = Path(output_file)
        with open(file_path, 'w') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif file_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported output format: {file_path.suffix}")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        config_dict = asdict(self._config)
        
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'data.cache_dir'
                keys = key.split('.')
                current = config_dict
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value
            else:
                config_dict[key] = value
        
        self._config = self._dict_to_config(config_dict)


# Global configuration instance
_config_manager = None


def get_config(config_file: Optional[str] = None) -> SystemConfig:
    """Get global configuration instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    
    return _config_manager.config


def update_config(**kwargs) -> None:
    """Update global configuration."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    _config_manager.update_config(**kwargs)