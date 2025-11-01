# Robust Portfolio Optimization System

A quantitative portfolio optimization system that detects market regimes, estimates regime-specific risk parameters, and computes robust asset allocations that guard against worst-case or expected regime scenarios.

## Project Structure

```
robust-portfolio-optimization/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── interfaces.py            # Base interfaces and data structures
│   ├── config.py               # Configuration management
│   ├── logging_config.py       # Logging framework
│   └── main.py                 # Main application entry point
├── data/                        # Data storage
│   └── cache/                  # Cached data files
├── notebooks/                   # Jupyter demonstration notebooks
├── tests/                       # Unit and integration tests
├── logs/                        # Log files
├── config.yaml                 # Default configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Features

- **Market Regime Detection**: Automatically identify bull, bear, and neutral market conditions using clustering algorithms
- **Risk Parameter Estimation**: Calculate regime-specific covariance matrices and expected returns with shrinkage estimation
- **Robust Optimization**: Solve worst-case variance minimization and CVaR optimization problems
- **Comprehensive Backtesting**: Simulate out-of-sample performance with realistic transaction costs
- **Rich Visualizations**: Generate regime analysis, allocation heatmaps, and performance comparisons
- **Modular Architecture**: Clean interfaces enabling easy extension and customization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd robust-portfolio-optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system (optional):
```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml with your preferences
```

## Quick Start

### Command Line Interface

```bash
# Check system status
python src/main.py --status

# Run with custom configuration
python src/main.py --config my_config.yaml --run-pipeline
```

### Python API

```python
from src.main import RobustPortfolioOptimizer

# Initialize with default configuration
optimizer = RobustPortfolioOptimizer()

# Run the complete pipeline
results = optimizer.run_full_pipeline()
print(results)
```

## Configuration

The system uses a hierarchical configuration system supporting:

1. **Default values** (built-in)
2. **Configuration files** (YAML/JSON)
3. **Environment variables** (RPO_* prefix)

### Key Configuration Sections

- **Data**: Asset tickers, data sources, caching settings
- **Regime**: Clustering parameters, validation methods
- **Optimization**: Constraints, solver settings, objectives
- **Backtest**: Window sizes, performance metrics, benchmarks

### Environment Variables

```bash
export RPO_LOG_LEVEL=DEBUG
export RPO_N_REGIMES=3
export RPO_MAX_WEIGHT=0.4
export RPO_DATA_CACHE_DIR=data/cache
```

## Components

### Data Manager
- Downloads asset prices from Yahoo Finance
- Retrieves macroeconomic data from FRED API
- Computes returns and regime detection features
- Handles data validation and caching

### Regime Detector
- Implements K-means and HMM clustering algorithms
- Creates regime-specific feature engineering
- Validates regime stability and economic interpretation

### Risk Estimator
- Calculates sample and shrinkage covariance matrices
- Estimates regime-specific expected returns
- Ensures numerical stability and positive semi-definiteness

### Robust Optimizer
- Formulates worst-case variance minimization problems
- Implements CVaR optimization with regime weighting
- Handles portfolio constraints and solver integration

### Backtest Engine
- Simulates rolling window out-of-sample performance
- Calculates comprehensive performance metrics
- Compares against multiple benchmark strategies

### Visualization Engine
- Creates regime detection time series plots
- Generates portfolio allocation heatmaps
- Produces performance comparison charts and reports

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Mathematical Methodology

The system implements several key mathematical concepts:

1. **Regime Detection**: Uses clustering on features like rolling volatility, VIX levels, and yield spreads
2. **Risk Estimation**: Applies Ledoit-Wolf shrinkage to improve covariance matrix estimation
3. **Robust Optimization**: Solves min-max problems using epigraph reformulation
4. **Performance Attribution**: Decomposes returns by regime contributions

## License

[License information to be added]

## Contributing

[Contributing guidelines to be added]

## References

[Academic references and methodology sources to be added]