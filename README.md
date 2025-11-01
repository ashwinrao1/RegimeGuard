# Robust Portfolio Optimization System

A comprehensive quantitative portfolio optimization system that combines regime detection, risk modeling, and robust optimization techniques for institutional-quality portfolio management.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd robust-portfolio-optimization

# Install dependencies
pip install -r requirements.txt

# Optional: Set up FRED API key for macroeconomic data
export FRED_API_KEY="your_fred_api_key_here"
```

### 2. Basic Usage

```python
# Import the main components
from src.data_manager import DataManager
from src.regime_detector import RegimeDetector
from src.risk_estimator import RiskEstimator
from src.robust_optimizer import RobustOptimizer
from src.backtest_engine import BacktestEngine
from src.visualization_engine import VisualizationEngine

# Initialize components
data_manager = DataManager()
regime_detector = RegimeDetector()
risk_estimator = RiskEstimator()
optimizer = RobustOptimizer()
backtest_engine = BacktestEngine()
viz_engine = VisualizationEngine()

# Run the complete pipeline (see examples below)
```

### 3. Run the Example

```bash
# Run the complete example
python examples/complete_example.py

# Or start with the Jupyter notebook
jupyter notebook notebooks/01_data_preparation.ipynb
```

## 📊 Usage Examples

### Example 1: Basic Regime Detection and Optimization

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 1. Download and prepare data
tickers = ['SPY', 'AGG', 'GLD', 'XLE', 'XLF', 'XLK']
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

# Download asset data
asset_prices = data_manager.download_asset_data(tickers, start_date, end_date)
asset_returns = data_manager.compute_returns(asset_prices)

# Download macro data (optional - requires FRED API key)
try:
    macro_data = data_manager.download_macro_data(
        ['VIXCLS', 'DGS10', 'DGS2', 'UNRATE'], start_date, end_date
    )
except:
    macro_data = pd.DataFrame()  # Use empty if no FRED key

# 2. Create regime features and detect regimes
regime_features = data_manager.create_regime_features(asset_returns, macro_data)
regime_labels = regime_detector.fit_regimes(regime_features, n_regimes=3)

# 3. Estimate risk parameters
regime_parameters = risk_estimator.estimate_all_regime_parameters(asset_returns, regime_labels)

# 4. Optimize portfolio
regime_covariances = risk_estimator.estimate_regime_covariance(asset_returns, regime_labels)
optimal_weights = optimizer.optimize_worst_case(regime_covariances)

print(f"Optimal portfolio weights: {optimal_weights}")
print(f"Weights sum: {np.sum(optimal_weights):.6f}")
```

### Example 2: Complete Backtesting Pipeline

```python
# Define optimization function for backtesting
def optimization_function(returns_window):
    """Optimization function for backtesting."""
    # Create features for this window
    features = data_manager.create_regime_features(returns_window, pd.DataFrame())
    
    # Detect regimes
    regime_labels = regime_detector.fit_regimes(features, n_regimes=3)
    
    # Estimate risk parameters
    regime_covariances = risk_estimator.estimate_regime_covariance(returns_window, regime_labels)
    
    # Optimize portfolio
    optimal_weights = optimizer.optimize_worst_case(regime_covariances)
    
    return optimal_weights

# Run backtest
backtest_result = backtest_engine.run_strategy_backtest(
    returns=asset_returns,
    optimization_func=optimization_function,
    start_date='2020-01-01',
    strategy_name='robust_optimization'
)

# Display results
print("Backtest Results:")
print(f"Total Return: {backtest_result.performance_metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {backtest_result.performance_metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {backtest_result.performance_metrics['max_drawdown']:.2%}")
```

### Example 3: Visualization and Reporting

```python
# Create comprehensive visualizations
dashboard = viz_engine.create_comprehensive_dashboard(
    backtest_results={'robust_optimization': {'result': backtest_result}},
    save_path='plots'
)

# Generate report
report = viz_engine.generate_summary_report(
    {'robust_optimization': {'result': backtest_result}}
)
print(report)

# Plot regime detection
regime_timeline = viz_engine.plot_regime_detection(regime_labels, regime_features.index)
```

## 🔧 Configuration

The system uses a YAML configuration file (`config.yaml`) for settings:

```yaml
# Data settings
data:
  default_tickers: ["SPY", "AGG", "GLD", "XLE", "XLF", "XLK"]
  macro_series: ["VIXCLS", "DGS10", "DGS2", "UNRATE"]
  start_date: "2010-01-01"
  lookback_years: 10

# Regime detection settings
regime:
  n_regimes: 3
  clustering_method: "kmeans"
  feature_window: 20

# Optimization settings
optimization:
  optimization_method: "worst_case"
  max_weight: 0.4
  min_weight: 0.0
  transaction_cost_bps: 5.0
  rebalance_frequency: "monthly"

# Backtesting settings
backtest:
  estimation_window: 252
  min_history: 504
```

## 📁 Project Structure

```
robust-portfolio-optimization/
├── src/                          # Source code
│   ├── data_manager.py          # Data acquisition and preprocessing
│   ├── regime_detector.py       # Regime detection algorithms
│   ├── risk_estimator.py        # Risk parameter estimation
│   ├── robust_optimizer.py      # Portfolio optimization
│   ├── backtest_engine.py       # Backtesting framework
│   ├── visualization_engine.py  # Plotting and reporting
│   ├── interfaces.py            # Abstract base classes
│   ├── config.py               # Configuration management
│   └── logging_config.py       # Logging setup
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_regime_detection.ipynb
│   ├── 03_optimization.ipynb
│   └── 04_backtesting.ipynb
├── tests/                       # Unit tests
├── examples/                    # Example scripts
├── data/                        # Data storage
├── config.yaml                 # Configuration file
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🛠️ Advanced Usage

### Custom Optimization Strategies

```python
# Create custom constraints
custom_constraints = {
    "box": {
        "min_weights": np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0]),
        "max_weights": np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2])
    },
    "turnover": {
        "previous_weights": previous_weights,
        "max_turnover": 0.2
    }
}

# Optimize with custom constraints
optimal_weights = optimizer.optimize_worst_case(regime_covariances, custom_constraints)

# Or use CVaR optimization
regime_returns = risk_estimator.estimate_regime_returns(asset_returns, regime_labels)
regime_probs = np.array([0.4, 0.3, 0.3])  # Regime probabilities
cvar_weights = optimizer.optimize_cvar(regime_returns, regime_probs, alpha=0.05)
```

### Parameter Storage and Loading

```python
# Save risk parameters
risk_estimator.save_parameters({'estimation_date': datetime.now()})

# Load previously estimated parameters
success = risk_estimator.load_parameters()
if success:
    print("Parameters loaded successfully")

# Export to CSV
exported_files = risk_estimator.export_parameters('exports/')
```

### Performance Analysis

```python
# Calculate detailed performance metrics
metrics = backtest_engine.calculate_performance_metrics(portfolio_returns)

# Compare multiple strategies
strategy_comparison = backtest_engine.compare_strategies({
    'robust_optimization': robust_returns,
    'equal_weight': equal_weight_returns,
    'minimum_variance': min_var_returns
})

# Generate detailed report
detailed_report = backtest_engine.generate_backtest_report('robust_optimization')
```

## 🔍 Troubleshooting

### Common Issues

1. **FRED API Key Missing**
   ```bash
   export FRED_API_KEY="your_api_key"
   # Or set in your environment variables
   ```

2. **Optimization Solver Issues**
   ```bash
   # Install additional solvers
   pip install mosek  # Commercial solver (requires license)
   pip install clarabel  # Open source alternative
   ```

3. **Memory Issues with Large Datasets**
   ```python
   # Reduce estimation window or use chunked processing
   config.backtest.estimation_window = 126  # 6 months instead of 1 year
   ```

4. **Convergence Issues**
   ```python
   # Try different optimization methods
   optimizer = RobustOptimizer(solver="pulp")  # Alternative solver
   ```

## 📚 Documentation

- **API Documentation**: See docstrings in each module
- **Mathematical Background**: See `docs/methodology.md`
- **Examples**: Check the `examples/` directory
- **Notebooks**: Interactive tutorials in `notebooks/`

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_risk_estimator.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Tips

1. **Use caching for repeated runs**
2. **Set appropriate estimation windows**
3. **Use parallel processing where available**
4. **Cache regime detection results**
5. **Use appropriate solver for your problem size**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built using modern Python scientific computing stack
- Optimization powered by CVXPY
- Data from Yahoo Finance and FRED APIs
- Inspired by academic research in robust portfolio optimization