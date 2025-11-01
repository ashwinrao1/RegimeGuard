"""
Robust optimization module for the robust portfolio optimization system.

This module implements robust optimization techniques including worst-case variance
minimization and CVaR optimization across different market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    pulp = None

from interfaces import RobustOptimizerInterface, OptimizationResult
from config import get_config
from logging_config import get_logger


class ConstraintManager:
    """Manages portfolio constraints for optimization problems."""
    
    def __init__(self):
        self.constraints = {}
        self.logger = get_logger(__name__)
    
    def add_constraint(self, constraint_type: str, parameters: Dict[str, Any]) -> None:
        """Add a constraint to the optimization problem.
        
        Args:
            constraint_type: Type of constraint ("budget", "long_only", "box", "turnover", "sector")
            parameters: Constraint parameters
        """
        self.constraints[constraint_type] = parameters
        self.logger.debug(f"Added {constraint_type} constraint: {parameters}")
    
    def remove_constraint(self, constraint_type: str) -> bool:
        """Remove a constraint.
        
        Args:
            constraint_type: Type of constraint to remove
            
        Returns:
            True if constraint was removed
        """
        if constraint_type in self.constraints:
            del self.constraints[constraint_type]
            self.logger.debug(f"Removed {constraint_type} constraint")
            return True
        return False
    
    def get_constraints(self) -> Dict[str, Any]:
        """Get all current constraints."""
        return self.constraints.copy()
    
    def validate_constraints(self, n_assets: int) -> List[str]:
        """Validate constraints for consistency.
        
        Args:
            n_assets: Number of assets
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check box constraints
        if "box" in self.constraints:
            box_params = self.constraints["box"]
            min_weights = box_params.get("min_weights")
            max_weights = box_params.get("max_weights")
            
            if min_weights is not None:
                if len(min_weights) != n_assets:
                    errors.append(f"min_weights length ({len(min_weights)}) != n_assets ({n_assets})")
                if np.any(min_weights < 0):
                    errors.append("min_weights contains negative values")
            
            if max_weights is not None:
                if len(max_weights) != n_assets:
                    errors.append(f"max_weights length ({len(max_weights)}) != n_assets ({n_assets})")
                if np.any(max_weights < 0):
                    errors.append("max_weights contains negative values")
            
            if min_weights is not None and max_weights is not None:
                if np.any(min_weights > max_weights):
                    errors.append("min_weights > max_weights for some assets")
        
        # Check budget constraint
        if "budget" in self.constraints:
            budget_params = self.constraints["budget"]
            target_sum = budget_params.get("target_sum", 1.0)
            
            if target_sum <= 0:
                errors.append("Budget target_sum must be positive")
        
        return errors


class WorstCaseOptimizer:
    """Implements worst-case variance minimization across regimes."""
    
    def __init__(self, solver: str = "auto"):
        """Initialize worst-case optimizer.
        
        Args:
            solver: Solver to use ("auto", "cvxpy", "pulp")
        """
        self.solver = solver
        self.logger = get_logger(__name__)
        
        # Check solver availability
        if solver == "auto":
            if CVXPY_AVAILABLE:
                self.solver = "cvxpy"
            elif PULP_AVAILABLE:
                self.solver = "pulp"
            else:
                raise ImportError("Neither CVXPY nor PULP is available")
        elif solver == "cvxpy" and not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is not available")
        elif solver == "pulp" and not PULP_AVAILABLE:
            raise ImportError("PULP is not available")
        
        self.logger.info(f"WorstCaseOptimizer initialized with solver: {self.solver}")
    
    def optimize(self, regime_covariances: Dict[int, np.ndarray], 
                constraints: ConstraintManager) -> OptimizationResult:
        """Solve worst-case variance minimization problem.
        
        The problem formulation is:
        minimize: t
        subject to: w^T * Σ_r * w ≤ t  for all regimes r
                   constraint set
        
        Args:
            regime_covariances: Dictionary mapping regime IDs to covariance matrices
            constraints: Constraint manager with portfolio constraints
            
        Returns:
            OptimizationResult with optimal weights and metadata
        """
        self.logger.info(f"Solving worst-case optimization with {len(regime_covariances)} regimes")
        
        start_time = time.time()
        
        # Get problem dimensions
        regime_ids = list(regime_covariances.keys())
        n_assets = regime_covariances[regime_ids[0]].shape[0]
        
        # Validate constraints
        constraint_errors = constraints.validate_constraints(n_assets)
        if constraint_errors:
            return OptimizationResult(
                weights=np.zeros(n_assets),
                objective_value=float('inf'),
                solver_status="constraint_error",
                computation_time=time.time() - start_time,
                constraints_satisfied=False
            )
        
        # Solve using appropriate solver
        if self.solver == "cvxpy":
            return self._solve_cvxpy(regime_covariances, constraints, start_time)
        elif self.solver == "pulp":
            return self._solve_pulp(regime_covariances, constraints, start_time)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
    
    def _solve_cvxpy(self, regime_covariances: Dict[int, np.ndarray], 
                    constraints: ConstraintManager, start_time: float) -> OptimizationResult:
        """Solve using CVXPY."""
        regime_ids = list(regime_covariances.keys())
        n_assets = regime_covariances[regime_ids[0]].shape[0]
        
        try:
            # Decision variables
            w = cp.Variable(n_assets)  # Portfolio weights
            t = cp.Variable()          # Worst-case variance
            
            # Objective: minimize worst-case variance
            objective = cp.Minimize(t)
            
            # Constraints list
            cvx_constraints = []
            
            # Worst-case constraints: w^T * Σ_r * w ≤ t for all regimes
            for regime_id, cov_matrix in regime_covariances.items():
                # Ensure covariance matrix is positive semi-definite for CVXPY
                try:
                    # Check if matrix is PSD
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    if np.min(eigenvals) < -1e-8:
                        # Regularize if not PSD
                        cov_matrix = cov_matrix + 1e-6 * np.eye(n_assets)
                    
                    cvx_constraints.append(cp.quad_form(w, cov_matrix) <= t)
                except Exception as e:
                    self.logger.warning(f"Issue with covariance matrix for regime {regime_id}: {str(e)}")
                    # Fallback: use diagonal approximation
                    diag_cov = np.diag(np.diag(cov_matrix))
                    cvx_constraints.append(cp.quad_form(w, diag_cov) <= t)
            
            # Add portfolio constraints
            cvx_constraints.extend(self._add_cvxpy_constraints(w, constraints))
            
            # Create and solve problem
            problem = cp.Problem(objective, cvx_constraints)
            
            # Solve with different solvers if needed
            solvers_to_try = ['ECOS', 'SCS', 'OSQP']
            
            for solver_name in solvers_to_try:
                try:
                    problem.solve(solver=solver_name, verbose=False)
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        break
                except Exception as e:
                    self.logger.debug(f"Solver {solver_name} failed: {str(e)}")
                    continue
            
            # Extract results
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                optimal_weights = w.value
                objective_value = t.value
                solver_status = "optimal"
                constraints_satisfied = True
                
                # Handle numerical issues
                if optimal_weights is None:
                    optimal_weights = np.zeros(n_assets)
                    objective_value = float('inf')
                    solver_status = "numerical_error"
                    constraints_satisfied = False
                else:
                    optimal_weights = np.array(optimal_weights).flatten()
                    
                    # Clean up small numerical errors
                    optimal_weights[np.abs(optimal_weights) < 1e-8] = 0
                    
                    # Normalize if budget constraint exists
                    constraint_dict = constraints.get_constraints()
                    if "budget" in constraint_dict:
                        target_sum = constraint_dict["budget"].get("target_sum", 1.0)
                        current_sum = np.sum(optimal_weights)
                        if abs(current_sum - target_sum) > 1e-6 and current_sum > 1e-8:
                            optimal_weights = optimal_weights * (target_sum / current_sum)
            
            else:
                optimal_weights = np.zeros(n_assets)
                objective_value = float('inf')
                solver_status = f"failed_{problem.status}"
                constraints_satisfied = False
            
            computation_time = time.time() - start_time
            
            return OptimizationResult(
                weights=optimal_weights,
                objective_value=float(objective_value) if objective_value is not None else float('inf'),
                solver_status=solver_status,
                computation_time=computation_time,
                constraints_satisfied=constraints_satisfied
            )
        
        except Exception as e:
            self.logger.error(f"CVXPY optimization failed: {str(e)}")
            return OptimizationResult(
                weights=np.zeros(n_assets),
                objective_value=float('inf'),
                solver_status=f"error_{str(e)[:50]}",
                computation_time=time.time() - start_time,
                constraints_satisfied=False
            )
    
    def _add_cvxpy_constraints(self, w, constraints: ConstraintManager) -> List:
        """Add CVXPY constraints based on constraint manager."""
        cvx_constraints = []
        constraint_dict = constraints.get_constraints()
        
        # Budget constraint (sum of weights)
        if "budget" in constraint_dict:
            budget_params = constraint_dict["budget"]
            target_sum = budget_params.get("target_sum", 1.0)
            tolerance = budget_params.get("tolerance", 0.0)
            
            if tolerance > 0:
                cvx_constraints.append(cp.sum(w) >= target_sum - tolerance)
                cvx_constraints.append(cp.sum(w) <= target_sum + tolerance)
            else:
                cvx_constraints.append(cp.sum(w) == target_sum)
        
        # Long-only constraint
        if "long_only" in constraint_dict:
            cvx_constraints.append(w >= 0)
        
        # Box constraints (individual weight bounds)
        if "box" in constraint_dict:
            box_params = constraint_dict["box"]
            min_weights = box_params.get("min_weights")
            max_weights = box_params.get("max_weights")
            
            if min_weights is not None:
                cvx_constraints.append(w >= min_weights)
            
            if max_weights is not None:
                cvx_constraints.append(w <= max_weights)
        
        # Turnover constraint (if previous weights provided)
        if "turnover" in constraint_dict:
            turnover_params = constraint_dict["turnover"]
            prev_weights = turnover_params.get("previous_weights")
            max_turnover = turnover_params.get("max_turnover")
            
            if prev_weights is not None and max_turnover is not None:
                # Turnover = sum of absolute changes in weights
                turnover = cp.norm(w - prev_weights, 1)
                cvx_constraints.append(turnover <= max_turnover)
        
        return cvx_constraints
    
    def _solve_pulp(self, regime_covariances: Dict[int, np.ndarray], 
                   constraints: ConstraintManager, start_time: float) -> OptimizationResult:
        """Solve using PULP (simplified version for linear approximation)."""
        self.logger.warning("PULP solver provides linear approximation of quadratic problem")
        
        regime_ids = list(regime_covariances.keys())
        n_assets = regime_covariances[regime_ids[0]].shape[0]
        
        try:
            # Create problem
            prob = pulp.LpProblem("WorstCaseOptimization", pulp.LpMinimize)
            
            # Decision variables
            w = [pulp.LpVariable(f"w_{i}", lowBound=0) for i in range(n_assets)]
            t = pulp.LpVariable("t", lowBound=0)
            
            # Objective: minimize t
            prob += t
            
            # Linear approximation of quadratic constraints
            # Use diagonal elements as approximation: sum(w_i^2 * σ_ii) ≤ t
            for regime_id, cov_matrix in regime_covariances.items():
                diagonal_vars = np.diag(cov_matrix)
                # Linear approximation: assume w_i ≈ 1/n for variance calculation
                approx_variance = pulp.lpSum([diagonal_vars[i] * w[i] * (1.0/n_assets) for i in range(n_assets)])
                prob += approx_variance <= t
            
            # Add constraints
            constraint_dict = constraints.get_constraints()
            
            # Budget constraint
            if "budget" in constraint_dict:
                target_sum = constraint_dict["budget"].get("target_sum", 1.0)
                prob += pulp.lpSum(w) == target_sum
            
            # Box constraints
            if "box" in constraint_dict:
                box_params = constraint_dict["box"]
                max_weights = box_params.get("max_weights")
                
                if max_weights is not None:
                    for i in range(n_assets):
                        prob += w[i] <= max_weights[i]
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Extract results
            if prob.status == pulp.LpStatusOptimal:
                optimal_weights = np.array([w[i].varValue for i in range(n_assets)])
                objective_value = t.varValue
                solver_status = "optimal"
                constraints_satisfied = True
            else:
                optimal_weights = np.zeros(n_assets)
                objective_value = float('inf')
                solver_status = f"failed_{pulp.LpStatus[prob.status]}"
                constraints_satisfied = False
            
            computation_time = time.time() - start_time
            
            return OptimizationResult(
                weights=optimal_weights,
                objective_value=float(objective_value) if objective_value is not None else float('inf'),
                solver_status=solver_status,
                computation_time=computation_time,
                constraints_satisfied=constraints_satisfied
            )
        
        except Exception as e:
            self.logger.error(f"PULP optimization failed: {str(e)}")
            return OptimizationResult(
                weights=np.zeros(n_assets),
                objective_value=float('inf'),
                solver_status=f"error_{str(e)[:50]}",
                computation_time=time.time() - start_time,
                constraints_satisfied=False
            )


class CVaROptimizer:
    """Implements Conditional Value at Risk (CVaR) optimization."""
    
    def __init__(self, solver: str = "auto"):
        """Initialize CVaR optimizer.
        
        Args:
            solver: Solver to use ("auto", "cvxpy", "pulp")
        """
        self.solver = solver
        self.logger = get_logger(__name__)
        
        # Check solver availability
        if solver == "auto":
            if CVXPY_AVAILABLE:
                self.solver = "cvxpy"
            elif PULP_AVAILABLE:
                self.solver = "pulp"
            else:
                raise ImportError("Neither CVXPY nor PULP is available")
        
        self.logger.info(f"CVaROptimizer initialized with solver: {self.solver}")
    
    def optimize(self, regime_returns: Dict[int, np.ndarray], 
                regime_probs: np.ndarray, alpha: float,
                constraints: ConstraintManager) -> OptimizationResult:
        """Solve CVaR optimization problem.
        
        The problem formulation is:
        minimize: VaR + (1/α) * E[max(0, -R - VaR)]
        
        Args:
            regime_returns: Dictionary mapping regime IDs to expected returns
            regime_probs: Array of regime probabilities
            alpha: Confidence level for CVaR (e.g., 0.05 for 95% CVaR)
            constraints: Constraint manager
            
        Returns:
            OptimizationResult with optimal weights and metadata
        """
        self.logger.info(f"Solving CVaR optimization with alpha={alpha}")
        
        start_time = time.time()
        
        # Get problem dimensions
        regime_ids = list(regime_returns.keys())
        n_assets = len(regime_returns[regime_ids[0]])
        n_regimes = len(regime_ids)
        
        # Validate inputs
        if len(regime_probs) != n_regimes:
            return OptimizationResult(
                weights=np.zeros(n_assets),
                objective_value=float('inf'),
                solver_status="dimension_error",
                computation_time=time.time() - start_time,
                constraints_satisfied=False
            )
        
        if not (0 < alpha < 1):
            return OptimizationResult(
                weights=np.zeros(n_assets),
                objective_value=float('inf'),
                solver_status="alpha_error",
                computation_time=time.time() - start_time,
                constraints_satisfied=False
            )
        
        # Solve using appropriate solver
        if self.solver == "cvxpy":
            return self._solve_cvar_cvxpy(regime_returns, regime_probs, alpha, constraints, start_time)
        else:
            return self._solve_cvar_pulp(regime_returns, regime_probs, alpha, constraints, start_time)
    
    def _solve_cvar_cvxpy(self, regime_returns: Dict[int, np.ndarray], 
                         regime_probs: np.ndarray, alpha: float,
                         constraints: ConstraintManager, start_time: float) -> OptimizationResult:
        """Solve CVaR using CVXPY."""
        regime_ids = list(regime_returns.keys())
        n_assets = len(regime_returns[regime_ids[0]])
        n_regimes = len(regime_ids)
        
        try:
            # Decision variables
            w = cp.Variable(n_assets)  # Portfolio weights
            var = cp.Variable()        # Value at Risk
            u = cp.Variable(n_regimes, nonneg=True)  # Auxiliary variables for CVaR
            
            # CVaR objective: VaR + (1/α) * E[u]
            expected_u = cp.sum(cp.multiply(regime_probs, u))
            objective = cp.Minimize(var + (1/alpha) * expected_u)
            
            # Constraints
            cvx_constraints = []
            
            # CVaR constraints: u_r ≥ -R_r - VaR for all regimes r
            for i, regime_id in enumerate(regime_ids):
                returns_r = regime_returns[regime_id]
                portfolio_return_r = cp.sum(cp.multiply(returns_r, w))
                cvx_constraints.append(u[i] >= -portfolio_return_r - var)
            
            # Add portfolio constraints
            cvx_constraints.extend(self._add_cvxpy_constraints(w, constraints))
            
            # Create and solve problem
            problem = cp.Problem(objective, cvx_constraints)
            
            # Try different solvers
            solvers_to_try = ['ECOS', 'SCS', 'OSQP']
            
            for solver_name in solvers_to_try:
                try:
                    problem.solve(solver=solver_name, verbose=False)
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        break
                except Exception as e:
                    self.logger.debug(f"Solver {solver_name} failed: {str(e)}")
                    continue
            
            # Extract results
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                optimal_weights = w.value
                objective_value = problem.value
                solver_status = "optimal"
                constraints_satisfied = True
                
                if optimal_weights is None:
                    optimal_weights = np.zeros(n_assets)
                    objective_value = float('inf')
                    solver_status = "numerical_error"
                    constraints_satisfied = False
                else:
                    optimal_weights = np.array(optimal_weights).flatten()
                    optimal_weights[np.abs(optimal_weights) < 1e-8] = 0
            else:
                optimal_weights = np.zeros(n_assets)
                objective_value = float('inf')
                solver_status = f"failed_{problem.status}"
                constraints_satisfied = False
            
            computation_time = time.time() - start_time
            
            return OptimizationResult(
                weights=optimal_weights,
                objective_value=float(objective_value) if objective_value is not None else float('inf'),
                solver_status=solver_status,
                computation_time=computation_time,
                constraints_satisfied=constraints_satisfied
            )
        
        except Exception as e:
            self.logger.error(f"CVaR CVXPY optimization failed: {str(e)}")
            return OptimizationResult(
                weights=np.zeros(n_assets),
                objective_value=float('inf'),
                solver_status=f"error_{str(e)[:50]}",
                computation_time=time.time() - start_time,
                constraints_satisfied=False
            )
    
    def _add_cvxpy_constraints(self, w, constraints: ConstraintManager) -> List:
        """Add CVXPY constraints (same as WorstCaseOptimizer)."""
        cvx_constraints = []
        constraint_dict = constraints.get_constraints()
        
        # Budget constraint
        if "budget" in constraint_dict:
            budget_params = constraint_dict["budget"]
            target_sum = budget_params.get("target_sum", 1.0)
            tolerance = budget_params.get("tolerance", 0.0)
            
            if tolerance > 0:
                cvx_constraints.append(cp.sum(w) >= target_sum - tolerance)
                cvx_constraints.append(cp.sum(w) <= target_sum + tolerance)
            else:
                cvx_constraints.append(cp.sum(w) == target_sum)
        
        # Long-only constraint
        if "long_only" in constraint_dict:
            cvx_constraints.append(w >= 0)
        
        # Box constraints
        if "box" in constraint_dict:
            box_params = constraint_dict["box"]
            min_weights = box_params.get("min_weights")
            max_weights = box_params.get("max_weights")
            
            if min_weights is not None:
                cvx_constraints.append(w >= min_weights)
            
            if max_weights is not None:
                cvx_constraints.append(w <= max_weights)
        
        return cvx_constraints
    
    def _solve_cvar_pulp(self, regime_returns: Dict[int, np.ndarray], 
                        regime_probs: np.ndarray, alpha: float,
                        constraints: ConstraintManager, start_time: float) -> OptimizationResult:
        """Solve CVaR using PULP (linear approximation)."""
        self.logger.warning("PULP CVaR optimization uses linear approximation")
        
        regime_ids = list(regime_returns.keys())
        n_assets = len(regime_returns[regime_ids[0]])
        n_regimes = len(regime_ids)
        
        try:
            # Create problem
            prob = pulp.LpProblem("CVaROptimization", pulp.LpMinimize)
            
            # Decision variables
            w = [pulp.LpVariable(f"w_{i}", lowBound=0) for i in range(n_assets)]
            var = pulp.LpVariable("var")
            u = [pulp.LpVariable(f"u_{r}", lowBound=0) for r in range(n_regimes)]
            
            # Objective: VaR + (1/α) * E[u]
            expected_u = pulp.lpSum([regime_probs[r] * u[r] for r in range(n_regimes)])
            prob += var + (1/alpha) * expected_u
            
            # CVaR constraints
            for r, regime_id in enumerate(regime_ids):
                returns_r = regime_returns[regime_id]
                portfolio_return_r = pulp.lpSum([returns_r[i] * w[i] for i in range(n_assets)])
                prob += u[r] >= -portfolio_return_r - var
            
            # Portfolio constraints
            constraint_dict = constraints.get_constraints()
            
            if "budget" in constraint_dict:
                target_sum = constraint_dict["budget"].get("target_sum", 1.0)
                prob += pulp.lpSum(w) == target_sum
            
            if "box" in constraint_dict:
                box_params = constraint_dict["box"]
                max_weights = box_params.get("max_weights")
                
                if max_weights is not None:
                    for i in range(n_assets):
                        prob += w[i] <= max_weights[i]
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Extract results
            if prob.status == pulp.LpStatusOptimal:
                optimal_weights = np.array([w[i].varValue for i in range(n_assets)])
                objective_value = prob.objective.value()
                solver_status = "optimal"
                constraints_satisfied = True
            else:
                optimal_weights = np.zeros(n_assets)
                objective_value = float('inf')
                solver_status = f"failed_{pulp.LpStatus[prob.status]}"
                constraints_satisfied = False
            
            computation_time = time.time() - start_time
            
            return OptimizationResult(
                weights=optimal_weights,
                objective_value=float(objective_value) if objective_value is not None else float('inf'),
                solver_status=solver_status,
                computation_time=computation_time,
                constraints_satisfied=constraints_satisfied
            )
        
        except Exception as e:
            self.logger.error(f"CVaR PULP optimization failed: {str(e)}")
            return OptimizationResult(
                weights=np.zeros(n_assets),
                objective_value=float('inf'),
                solver_status=f"error_{str(e)[:50]}",
                computation_time=time.time() - start_time,
                constraints_satisfied=False
            )


class RobustOptimizer(RobustOptimizerInterface):
    """Main robust optimizer class implementing the RobustOptimizerInterface."""
    
    def __init__(self, solver: str = "auto"):
        """Initialize robust optimizer.
        
        Args:
            solver: Solver to use ("auto", "cvxpy", "pulp")
        """
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize optimizers
        self.worst_case_optimizer = WorstCaseOptimizer(solver)
        self.cvar_optimizer = CVaROptimizer(solver)
        self.constraint_manager = ConstraintManager()
        
        # State
        self.last_result = None
        self.optimization_history = []
        
        # Add default constraints
        self._setup_default_constraints()
        
        self.logger.info(f"RobustOptimizer initialized with solver: {solver}")
    
    def _setup_default_constraints(self):
        """Set up default portfolio constraints from config."""
        # Budget constraint (weights sum to 1)
        self.constraint_manager.add_constraint("budget", {"target_sum": 1.0})
        
        # Long-only constraint
        self.constraint_manager.add_constraint("long_only", {})
        
        # Box constraints from config
        max_weight = self.config.optimization.max_weight
        min_weight = self.config.optimization.min_weight
        
        if max_weight < 1.0 or min_weight > 0.0:
            # Will be set per problem based on number of assets
            pass
    
    def optimize_worst_case(self, regime_covariances: Dict[int, np.ndarray], 
                           constraints: Optional[Dict] = None) -> np.ndarray:
        """Solve worst-case variance minimization problem."""
        self.logger.info("Starting worst-case optimization")
        
        # Update constraints if provided
        if constraints:
            self._update_constraints(constraints, len(next(iter(regime_covariances.values()))))
        else:
            # Set default box constraints
            n_assets = len(next(iter(regime_covariances.values())))
            self._set_default_box_constraints(n_assets)
        
        # Solve optimization
        result = self.worst_case_optimizer.optimize(regime_covariances, self.constraint_manager)
        
        # Store result
        self.last_result = result
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'method': 'worst_case',
            'objective_value': result.objective_value,
            'solver_status': result.solver_status,
            'computation_time': result.computation_time
        })
        
        self.logger.info(f"Worst-case optimization completed: {result.solver_status}, objective={result.objective_value:.6f}")
        
        return result.weights
    
    def optimize_cvar(self, regime_returns: Dict[int, np.ndarray], 
                     regime_probs: np.ndarray, alpha: float) -> np.ndarray:
        """Solve CVaR optimization problem."""
        self.logger.info(f"Starting CVaR optimization with alpha={alpha}")
        
        # Set default box constraints
        n_assets = len(next(iter(regime_returns.values())))
        self._set_default_box_constraints(n_assets)
        
        # Solve optimization
        result = self.cvar_optimizer.optimize(regime_returns, regime_probs, alpha, self.constraint_manager)
        
        # Store result
        self.last_result = result
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'method': 'cvar',
            'alpha': alpha,
            'objective_value': result.objective_value,
            'solver_status': result.solver_status,
            'computation_time': result.computation_time
        })
        
        self.logger.info(f"CVaR optimization completed: {result.solver_status}, objective={result.objective_value:.6f}")
        
        return result.weights
    
    def add_constraint(self, constraint_type: str, parameters: Dict) -> None:
        """Add portfolio constraint to optimization problem."""
        self.constraint_manager.add_constraint(constraint_type, parameters)
        self.logger.info(f"Added constraint: {constraint_type}")
    
    def solve_optimization(self) -> OptimizationResult:
        """Solve the configured optimization problem."""
        if self.last_result is None:
            raise ValueError("No optimization has been run yet")
        
        return self.last_result
    
    def _update_constraints(self, constraints: Dict, n_assets: int):
        """Update constraint manager with provided constraints."""
        # Clear existing constraints except defaults
        current_constraints = self.constraint_manager.get_constraints()
        for constraint_type in list(current_constraints.keys()):
            if constraint_type not in ["budget", "long_only"]:
                self.constraint_manager.remove_constraint(constraint_type)
        
        # Add new constraints
        for constraint_type, params in constraints.items():
            self.constraint_manager.add_constraint(constraint_type, params)
        
        # Ensure box constraints are set
        if "box" not in constraints:
            self._set_default_box_constraints(n_assets)
    
    def _set_default_box_constraints(self, n_assets: int):
        """Set default box constraints based on config."""
        max_weight = self.config.optimization.max_weight
        min_weight = self.config.optimization.min_weight
        
        box_constraints = {
            "min_weights": np.full(n_assets, min_weight),
            "max_weights": np.full(n_assets, max_weight)
        }
        
        self.constraint_manager.add_constraint("box", box_constraints)
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def get_last_result(self) -> Optional[OptimizationResult]:
        """Get last optimization result."""
        return self.last_result
    
    def validate_solution(self, weights: np.ndarray) -> Dict[str, Any]:
        """Validate an optimization solution against constraints.
        
        Args:
            weights: Portfolio weights to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'violations': [],
            'metrics': {}
        }
        
        constraints = self.constraint_manager.get_constraints()
        
        # Check budget constraint
        if "budget" in constraints:
            target_sum = constraints["budget"].get("target_sum", 1.0)
            tolerance = constraints["budget"].get("tolerance", 1e-6)
            actual_sum = np.sum(weights)
            
            validation['metrics']['weight_sum'] = float(actual_sum)
            validation['metrics']['budget_target'] = target_sum
            
            if abs(actual_sum - target_sum) > tolerance:
                validation['valid'] = False
                validation['violations'].append(f"Budget violation: sum={actual_sum:.6f}, target={target_sum}")
        
        # Check long-only constraint
        if "long_only" in constraints:
            negative_weights = weights < -1e-8
            if np.any(negative_weights):
                validation['valid'] = False
                negative_indices = np.where(negative_weights)[0]
                validation['violations'].append(f"Negative weights at indices: {negative_indices.tolist()}")
        
        # Check box constraints
        if "box" in constraints:
            box_params = constraints["box"]
            min_weights = box_params.get("min_weights")
            max_weights = box_params.get("max_weights")
            
            if min_weights is not None:
                violations = weights < (min_weights - 1e-8)
                if np.any(violations):
                    validation['valid'] = False
                    violation_indices = np.where(violations)[0]
                    validation['violations'].append(f"Min weight violations at indices: {violation_indices.tolist()}")
            
            if max_weights is not None:
                violations = weights > (max_weights + 1e-8)
                if np.any(violations):
                    validation['valid'] = False
                    violation_indices = np.where(violations)[0]
                    validation['violations'].append(f"Max weight violations at indices: {violation_indices.tolist()}")
        
        # Additional metrics
        validation['metrics'].update({
            'min_weight': float(np.min(weights)),
            'max_weight': float(np.max(weights)),
            'weight_concentration': float(np.sum(weights**2)),  # Herfindahl index
            'n_nonzero_weights': int(np.sum(np.abs(weights) > 1e-8))
        })
        
        return validation
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                  regime_covariances: Dict[int, np.ndarray],
                                  regime_returns: Optional[Dict[int, np.ndarray]] = None,
                                  regime_probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate portfolio risk and return metrics.
        
        Args:
            weights: Portfolio weights
            regime_covariances: Regime covariance matrices
            regime_returns: Regime expected returns (optional)
            regime_probs: Regime probabilities (optional)
            
        Returns:
            Dictionary with portfolio metrics
        """
        metrics = {}
        
        # Calculate regime-specific risks
        regime_risks = {}
        for regime_id, cov_matrix in regime_covariances.items():
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            regime_risks[regime_id] = {
                'variance': float(portfolio_variance),
                'volatility': float(portfolio_volatility)
            }
            
            # Add returns if available
            if regime_returns and regime_id in regime_returns:
                portfolio_return = weights.T @ regime_returns[regime_id]
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                regime_risks[regime_id].update({
                    'return': float(portfolio_return),
                    'sharpe_ratio': float(sharpe_ratio)
                })
        
        metrics['regime_risks'] = regime_risks
        
        # Worst-case metrics
        worst_case_variance = max(risk['variance'] for risk in regime_risks.values())
        worst_case_volatility = np.sqrt(worst_case_variance)
        
        metrics['worst_case'] = {
            'variance': float(worst_case_variance),
            'volatility': float(worst_case_volatility)
        }
        
        # Expected metrics (if probabilities available)
        if regime_probs is not None and len(regime_probs) == len(regime_risks):
            regime_ids = list(regime_risks.keys())
            expected_variance = sum(regime_probs[i] * regime_risks[regime_ids[i]]['variance'] 
                                  for i in range(len(regime_ids)))
            expected_volatility = np.sqrt(expected_variance)
            
            metrics['expected'] = {
                'variance': float(expected_variance),
                'volatility': float(expected_volatility)
            }
            
            if regime_returns:
                expected_return = sum(regime_probs[i] * regime_risks[regime_ids[i]].get('return', 0)
                                    for i in range(len(regime_ids)))
                expected_sharpe = expected_return / expected_volatility if expected_volatility > 0 else 0
                
                metrics['expected'].update({
                    'return': float(expected_return),
                    'sharpe_ratio': float(expected_sharpe)
                })
        
        return metrics