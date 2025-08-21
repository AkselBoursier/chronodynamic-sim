#!/usr/bin/env python3
"""
Differential Solvers for Chronodynamic Cosmology
=================================================

High-precision numerical solvers for the modified Einstein equations
in the Chronodynamic Cosmological Divergence (CCD) model.

Author: Aksel Boursier
Date: August 2025
"""

import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve
from typing import Callable, Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """Configuration for numerical solvers"""
    method: str = 'RK45'  # Integration method
    rtol: float = 1e-10   # Relative tolerance
    atol: float = 1e-12   # Absolute tolerance
    max_step: float = 0.01 # Maximum step size
    dense_output: bool = True  # Enable dense output
    
    
class AdaptiveStepSolver:
    """
    Adaptive step-size solver for chronodynamic evolution equations.
    
    Implements high-precision integration with automatic error control
    and stability monitoring for the coupled system:
    - Scale factor evolution: a(τ)
    - Dynamic time evolution: T(τ)
    - Chronodynamic tensor components: C_μν(τ)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
        self.integration_stats = {
            'total_steps': 0,
            'rejected_steps': 0,
            'min_step_size': np.inf,
            'max_step_size': 0.0
        }
    
    def solve_chronodynamic_system(self, 
                                 system_func: Callable,
                                 tau_span: Tuple[float, float],
                                 initial_conditions: np.ndarray,
                                 tau_eval: Optional[np.ndarray] = None) -> Dict:
        """
        Solve the chronodynamic system with adaptive step control.
        
        Args:
            system_func: Function defining dy/dτ = f(τ, y)
            tau_span: Integration interval (tau_start, tau_end)
            initial_conditions: Initial values y(tau_start)
            tau_eval: Specific points to evaluate solution
            
        Returns:
            Dictionary with solution data and statistics
        """
        logger.info(f"Starting chronodynamic system integration over τ ∈ {tau_span}")
        
        # Enhanced solver with event detection for stability
        def monitored_system(tau, y):
            """System function with monitoring"""
            dydt = system_func(tau, y)
            
            # Check for numerical instabilities
            if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
                logger.error(f"Numerical instability detected at τ={tau}")
                raise RuntimeError("Integration became unstable")
            
            return dydt
        
        # Event functions for detecting critical points
        def scale_factor_event(tau, y):
            """Detect when scale factor approaches zero"""
            return y[0] - 1e-10  # a(τ) approaching zero
        
        scale_factor_event.terminal = True
        scale_factor_event.direction = -1
        
        # Solve with monitoring
        solution = solve_ivp(
            monitored_system,
            tau_span,
            initial_conditions,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.max_step,
            dense_output=self.config.dense_output,
            events=[scale_factor_event],
            t_eval=tau_eval
        )
        
        if not solution.success:
            logger.error(f"Integration failed: {solution.message}")
            raise RuntimeError(f"Solver failed: {solution.message}")
        
        # Update statistics
        self.integration_stats['total_steps'] = solution.nfev
        
        logger.info(f"Integration completed successfully with {solution.nfev} function evaluations")
        
        return {
            'tau': solution.t,
            'y': solution.y,
            'success': solution.success,
            'message': solution.message,
            'nfev': solution.nfev,
            'njev': solution.njev,
            'nlu': solution.nlu,
            'events': solution.t_events,
            'sol': solution.sol if self.config.dense_output else None
        }
    
    def solve_constraint_equations(self, 
                                 constraint_func: Callable,
                                 initial_guess: np.ndarray,
                                 tau: float) -> Dict:
        """
        Solve constraint equations at a given time τ.
        
        Used for enforcing Hamiltonian and momentum constraints
        in the chronodynamic system.
        """
        def constraint_system(variables):
            """Constraint equations to be solved"""
            constraints = constraint_func(tau, variables)
            return constraints
        
        # Solve constraints
        solution = fsolve(
            constraint_system,
            initial_guess,
            xtol=1e-12,
            full_output=True
        )
        
        variables, info, converged, message = solution
        
        if converged != 1:
            logger.warning(f"Constraint solver did not converge at τ={tau}: {message}")
        
        return {
            'variables': variables,
            'residual': info['fvec'],
            'converged': converged == 1,
            'message': message,
            'function_calls': info['nfev']
        }


class StabilityAnalyzer:
    """
    Analyzes numerical stability of chronodynamic solutions.
    """
    
    def __init__(self):
        self.stability_metrics = {}
    
    def analyze_solution_stability(self, 
                                 tau: np.ndarray, 
                                 solution: np.ndarray) -> Dict:
        """
        Analyze the stability of a chronodynamic solution.
        
        Args:
            tau: Time array
            solution: Solution array [a(τ), a'(τ), T(τ), ...]
            
        Returns:
            Stability metrics and diagnostics
        """
        n_vars = solution.shape[0]
        stability_report = {
            'is_stable': True,
            'growth_rates': [],
            'oscillation_detection': [],
            'conservation_violations': []
        }
        
        for i in range(n_vars):
            var_data = solution[i]
            
            # Check for exponential growth
            growth_rate = self._compute_growth_rate(tau, var_data)
            stability_report['growth_rates'].append(growth_rate)
            
            if growth_rate > 10.0:  # Threshold for instability
                stability_report['is_stable'] = False
                logger.warning(f"Variable {i} shows exponential growth: λ={growth_rate}")
            
            # Check for high-frequency oscillations
            oscillation_freq = self._detect_oscillations(tau, var_data)
            stability_report['oscillation_detection'].append(oscillation_freq)
            
            if oscillation_freq > 1000:  # High frequency threshold
                stability_report['is_stable'] = False
                logger.warning(f"Variable {i} shows high-frequency oscillations")
        
        return stability_report
    
    def _compute_growth_rate(self, tau: np.ndarray, variable: np.ndarray) -> float:
        """Compute exponential growth rate of a variable"""
        if len(variable) < 10:
            return 0.0
        
        # Fit exponential growth: var(t) ≈ var₀ * exp(λt)
        log_var = np.log(np.abs(variable) + 1e-16)  # Avoid log(0)
        
        # Linear fit to log(var) vs τ
        coeffs = np.polyfit(tau, log_var, 1)
        growth_rate = coeffs[0]  # Slope = growth rate
        
        return growth_rate
    
    def _detect_oscillations(self, tau: np.ndarray, variable: np.ndarray) -> float:
        """Detect high-frequency oscillations using FFT"""
        if len(variable) < 20:
            return 0.0
        
        # Compute power spectral density
        dt = np.mean(np.diff(tau))
        freqs = np.fft.fftfreq(len(variable), dt)
        fft_var = np.fft.fft(variable)
        power = np.abs(fft_var)**2
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1
        dominant_freq = np.abs(freqs[dominant_freq_idx])
        
        return dominant_freq


class ConvergenceAnalyzer:
    """
    Analyzes numerical convergence of chronodynamic simulations.
    """
    
    def __init__(self):
        self.convergence_history = []
    
    def test_spatial_convergence(self, 
                                solver_func: Callable,
                                grid_sizes: List[int],
                                reference_solution: Optional[np.ndarray] = None) -> Dict:
        """
        Test spatial convergence using Richardson extrapolation.
        
        Args:
            solver_func: Function that runs simulation for given grid size
            grid_sizes: List of grid sizes to test [N₁, N₂, N₃, ...]
            reference_solution: High-resolution reference solution
            
        Returns:
            Convergence analysis results
        """
        solutions = {}
        errors = {}
        
        logger.info("Starting spatial convergence analysis")
        
        for N in grid_sizes:
            logger.info(f"Computing solution for grid size N={N}")
            solutions[N] = solver_func(N)
        
        # Compute convergence rates
        convergence_rates = {}
        
        for i in range(len(grid_sizes) - 1):
            N1, N2 = grid_sizes[i], grid_sizes[i+1]
            sol1, sol2 = solutions[N1], solutions[N2]
            
            # Interpolate coarser solution to finer grid
            sol1_interp = self._interpolate_solution(sol1, sol2['tau'])
            
            # Compute error
            error = np.abs(sol2['y'] - sol1_interp)
            max_error = np.max(error)
            
            # Estimate convergence rate: error ∝ h^p where h = 1/N
            h1, h2 = 1.0/N1, 1.0/N2
            if max_error > 1e-16:  # Avoid numerical noise
                p = np.log(max_error / np.max(np.abs(sol1['y'] - sol1_interp))) / np.log(h2/h1)
                convergence_rates[f'{N1}→{N2}'] = p
            
            errors[f'{N1}→{N2}'] = max_error
        
        return {
            'solutions': solutions,
            'errors': errors,
            'convergence_rates': convergence_rates,
            'is_converged': all(p > 1.5 for p in convergence_rates.values())
        }
    
    def test_temporal_convergence(self,
                                solver_func: Callable,
                                dt_values: List[float]) -> Dict:
        """
        Test temporal convergence by varying time step sizes.
        
        Args:
            solver_func: Function that runs simulation with given dt
            dt_values: List of time step sizes
            
        Returns:
            Temporal convergence analysis
        """
        solutions = {}
        errors = {}
        
        for dt in dt_values:
            logger.info(f"Computing solution for dt={dt}")
            solutions[dt] = solver_func(dt)
        
        # Use finest resolution as reference
        dt_ref = min(dt_values)
        ref_solution = solutions[dt_ref]
        
        for dt in dt_values[:-1]:  # Exclude reference
            sol = solutions[dt]
            
            # Interpolate to reference time grid
            sol_interp = self._interpolate_solution(sol, ref_solution['tau'])
            
            # Compute temporal error
            error = np.abs(ref_solution['y'] - sol_interp)
            errors[dt] = np.max(error)
        
        return {
            'solutions': solutions,
            'errors': errors,
            'reference_dt': dt_ref
        }
    
    def _interpolate_solution(self, 
                            source_solution: Dict, 
                            target_tau: np.ndarray) -> np.ndarray:
        """Interpolate solution to target time grid"""
        from scipy.interpolate import interp1d
        
        interpolator = interp1d(
            source_solution['tau'],
            source_solution['y'],
            axis=1,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        return interpolator(target_tau)


class ConstraintPreservation:
    """
    Ensures preservation of physical constraints during evolution.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.constraint_violations = []
    
    def hamiltonian_constraint(self, 
                             tau: float, 
                             variables: np.ndarray,
                             tensor_components: np.ndarray) -> float:
        """
        Compute Hamiltonian constraint violation.
        
        The Hamiltonian constraint in chronodynamic cosmology:
        H = G₀₀ + Λ + C₀₀ - 8πT₀₀ = 0
        """
        a, a_prime = variables[:2]
        C00 = tensor_components[0, 0]
        
        # Einstein tensor component G₀₀
        H_conf_squared = (a_prime / a)**2
        G00 = 3 * H_conf_squared / a**2
        
        # Matter energy density (simplified)
        rho_matter = 1.0  # Placeholder - should be computed from matter model
        T00 = rho_matter
        
        # Hamiltonian constraint
        H_constraint = G00 + C00 - 8 * np.pi * T00
        
        return H_constraint
    
    def momentum_constraint(self, 
                          tau: float, 
                          variables: np.ndarray,
                          tensor_components: np.ndarray) -> np.ndarray:
        """
        Compute momentum constraint violations.
        
        In homogeneous cosmology, momentum constraints are automatically satisfied,
        but we check for consistency with chronodynamic modifications.
        """
        # For FLRW spacetime, momentum constraints are trivial
        # But chronodynamic effects might introduce violations
        C0i = tensor_components[0, 1:4]  # Time-space components
        
        # Momentum constraint: ∇ⱼ(Gⱼᵢ + Cⱼᵢ) = 8π∇ⱼTⱼᵢ
        # In homogeneous case, this reduces to checking C₀ᵢ = 0
        momentum_violations = C0i
        
        return momentum_violations
    
    def energy_conservation(self, 
                          tau_array: np.ndarray, 
                          solution: np.ndarray) -> np.ndarray:
        """
        Check energy conservation throughout evolution.
        
        Compute ∇_μ T^μν = 0 for the chronodynamic system.
        """
        conservation_violations = []
        
        for i, tau in enumerate(tau_array):
            # Extract solution at this time
            variables = solution[:, i]
            
            # Compute energy density evolution
            # This is a simplified check - full implementation would compute
            # the covariant divergence of the stress-energy tensor
            
            if i > 0:
                dt = tau_array[i] - tau_array[i-1]
                energy_change = (variables[0]**3 - solution[0, i-1]**3) / dt
                
                # Expected change from chronodynamic effects
                expected_change = 0.0  # Placeholder for full calculation
                
                violation = abs(energy_change - expected_change)
                conservation_violations.append(violation)
            else:
                conservation_violations.append(0.0)
        
        return np.array(conservation_violations)
    
    def monitor_constraints(self, 
                          solution_data: Dict,
                          tensor_data: Dict) -> Dict:
        """
        Monitor all constraints throughout the evolution.
        
        Returns comprehensive constraint violation report.
        """
        tau_array = solution_data['tau']
        solution = solution_data['y']
        
        violations = {
            'hamiltonian': [],
            'momentum': [],
            'energy_conservation': [],
            'max_violation': 0.0,
            'is_satisfied': True
        }
        
        for i, tau in enumerate(tau_array):
            variables = solution[:, i]
            tensor_components = tensor_data['C'][i]  # Assuming stored tensor data
            
            # Check Hamiltonian constraint
            H_viol = abs(self.hamiltonian_constraint(tau, variables, tensor_components))
            violations['hamiltonian'].append(H_viol)
            
            # Check momentum constraints
            mom_viol = np.max(np.abs(self.momentum_constraint(tau, variables, tensor_components)))
            violations['momentum'].append(mom_viol)
            
            # Update maximum violation
            violations['max_violation'] = max(
                violations['max_violation'], 
                H_viol, 
                mom_viol
            )
        
        # Check energy conservation
        energy_viol = self.energy_conservation(tau_array, solution)
        violations['energy_conservation'] = energy_viol.tolist()
        violations['max_violation'] = max(violations['max_violation'], np.max(energy_viol))
        
        # Determine if constraints are satisfied
        violations['is_satisfied'] = violations['max_violation'] < self.tolerance
        
        if not violations['is_satisfied']:
            logger.warning(f"Constraint violations detected: max = {violations['max_violation']}")
        
        return violations


# Example usage and testing
if __name__ == "__main__":
    print("Testing Chronodynamic Differential Solvers")
    print("=" * 50)
    
    # Test adaptive solver
    config = SolverConfig(method='RK45', rtol=1e-10, atol=1e-12)
    solver = AdaptiveStepSolver(config)
    
    # Simple test system: harmonic oscillator
    def harmonic_system(tau, y):
        x, v = y
        return np.array([v, -x])
    
    tau_span = (0, 2*np.pi)
    initial_conditions = np.array([1.0, 0.0])
    
    try:
        result = solver.solve_chronodynamic_system(
            harmonic_system, tau_span, initial_conditions
        )
        print(f"Test integration successful: {result['nfev']} evaluations")
        
        # Test stability analysis
        analyzer = StabilityAnalyzer()
        stability = analyzer.analyze_solution_stability(result['tau'], result['y'])
        print(f"Solution stability: {stability['is_stable']}")
        
    except Exception as e:
        print(f"Test failed: {e}")