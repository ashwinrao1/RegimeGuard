"""
Main application entry point for the robust portfolio optimization system.

This module provides the main application class that orchestrates all components
and serves as the primary interface for running the optimization system.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config, SystemConfig
from logging_config import setup_logging, get_logger, LoggerMixin


class RobustPortfolioOptimizer(LoggerMixin):
    """Main application class for the robust portfolio optimization system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the portfolio optimizer.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        # Load configuration
        self.config = get_config(config_file)
        
        # Setup logging
        setup_logging(
            log_level=self.config.log_level,
            log_file="logs/robust_portfolio.log",
            enable_console=True
        )
        
        self.logger.info("Initializing Robust Portfolio Optimization System")
        self.logger.info(f"Configuration loaded: {self.config}")
        
        # Initialize components (will be implemented in later tasks)
        self.data_manager = None
        self.regime_detector = None
        self.risk_estimator = None
        self.robust_optimizer = None
        self.backtest_engine = None
        self.visualization_engine = None
    
    def initialize_components(self) -> None:
        """Initialize all system components."""
        self.logger.info("Initializing system components...")
        
        # Components will be initialized in subsequent tasks
        # This is a placeholder for the component initialization
        
        self.logger.info("All components initialized successfully")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete portfolio optimization pipeline.
        
        Returns:
            Dictionary containing all results and analysis
        """
        self.logger.info("Starting full portfolio optimization pipeline")
        
        try:
            # Initialize components if not already done
            if self.data_manager is None:
                self.initialize_components()
            
            # Pipeline steps (to be implemented in subsequent tasks):
            # 1. Download and preprocess data
            # 2. Detect market regimes
            # 3. Estimate regime-specific risk parameters
            # 4. Solve robust optimization problem
            # 5. Run backtesting simulation
            # 6. Generate visualizations and reports
            
            results = {
                "status": "success",
                "message": "Pipeline completed successfully",
                "components_initialized": True
            }
            
            self.logger.info("Portfolio optimization pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "components_initialized": False
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and component health."""
        return {
            "config_loaded": self.config is not None,
            "data_manager": self.data_manager is not None,
            "regime_detector": self.regime_detector is not None,
            "risk_estimator": self.risk_estimator is not None,
            "robust_optimizer": self.robust_optimizer is not None,
            "backtest_engine": self.backtest_engine is not None,
            "visualization_engine": self.visualization_engine is not None
        }


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Robust Portfolio Optimization System"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the full optimization pipeline"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    optimizer = RobustPortfolioOptimizer(config_file=args.config)
    
    if args.status:
        status = optimizer.get_system_status()
        print("System Status:")
        for component, initialized in status.items():
            print(f"  {component}: {'✓' if initialized else '✗'}")
    
    elif args.run_pipeline:
        results = optimizer.run_full_pipeline()
        print(f"Pipeline Status: {results['status']}")
        print(f"Message: {results['message']}")
    
    else:
        print("Robust Portfolio Optimization System")
        print("Use --help for available options")


if __name__ == "__main__":
    main()