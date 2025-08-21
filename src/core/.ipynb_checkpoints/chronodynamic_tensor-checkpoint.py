#!/usr/bin/env python3
"""
Chronodynamic Tensor Implementation
===================================

Implementation of the chronodynamic tensor C_μν that modifies Einstein's equations
in the Chronodynamic Cosmological Divergence (CCD) model by Aksel Boursier.

The modified Einstein equations are:
G_μν + Λg_μν + C_μν = 8πT_μν

Author: Aksel Boursier
Date: August 2025
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from typing import Tuple, Dict, Callable, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CosmologicalParams:
    """Cosmological parameters for CCD model"""
    H0: float = 67.4  # Hubble constant [km/s/Mpc]
    Omega_m: float = 0.315  # Matter density parameter
    Omega_lambda: float = 0.685  # Dark energy density parameter
    Omega_r: float = 8.24e-5  # Radiation density parameter
    S_chrono: float = 1.0  # Chronodynamic coupling strength
    T0_scale: float = 1.0  # Initial time scale factor
    q0_decel: float = -0.55  # Deceleration parameter
    j0_jerk: float = 1.0  # Jerk parameter


class ChronodynamicTensor:
    """
    Implementation of the chronodynamic tensor C_μν.
    
    This tensor encodes the dynamic temporal evolution that emerges from the
    observer-cosmos interface in the CCD model.
    """
    
    def __init__(self, params: CosmologicalParams, grid_size: int = 256):
        """
        Initialize the chronodynamic tensor.
        
        Args:
            params: Cosmological parameters
            grid_size: Spatial grid resolution
        """
        self.params = params
        self.grid_size = grid_size
        self.components = np.zeros((4, 4, grid_size, grid_size, grid_size))
        
        # Initialize time function T(τ) - dynamic cosmic time
        self.T_function = self._initialize_time_function()
        
        logger.info(f"Initialized ChronodynamicTensor with grid_size={grid_size}")
    
    def _initialize_time_function(self) -> Callable:
        """
        Initialize the dynamic time function T(τ).
        
        In CCD model, cosmic time emerges dynamically from observer-cosmos interface.
        T(τ) represents the local temporal rhythm that varies across spacetime.
        """
        def T(tau: float, x: np.ndarray) -> float:
            """
            Dynamic time function T(τ, x).
            
            Args:
                tau: Conformal time
                x: Spatial coordinates [x, y, z]
            
            Returns:
                Local dynamic time
            """
            # Base temporal evolution
            T_base = self.params.T0_scale * tau
            
            # Chronodynamic corrections
            chrono_correction = self.params.S_chrono * np.exp(-tau / self.params.T0_scale)
            
            # Spatial inhomogeneities (emerging from observer-cosmos interface)
            spatial_modulation = 1.0 + 0.1 * np.sin(np.linalg.norm(x) / 100.0)
            
            return T_base * (1.0 + chrono_correction * spatial_modulation)
        
        return T
    
    def compute_metric_derivatives(self, tau: float, a: float) -> Dict[str, np.ndarray]:
        """
        Compute derivatives of the Friedmann-Lemaître metric.
        
        Args:
            tau: Conformal time
            a: Scale factor
            
        Returns:
            Dictionary containing metric derivatives
        """
        # Friedmann-Lemaître metric: ds² = a²(τ)[-dτ² + δᵢⱼdxⁱdxʲ]
        a_prime = self._compute_scale_factor_derivative(tau, a)
        a_double_prime = self._compute_scale_factor_second_derivative(tau, a)
        
        return {
            'a_prime': a_prime,
            'a_double_prime': a_double_prime,
            'H_conformal': a_prime / a  # Conformal Hubble parameter
        }
    
    def _compute_scale_factor_derivative(self, tau: float, a: float) -> float:
        """Compute da/dτ from modified Friedmann equations"""
        H_conf = np.sqrt(
            8 * np.pi * self.params.Omega_m / (3 * a) +
            8 * np.pi * self.params.Omega_lambda * a**2 / 3 +
            8 * np.pi * self.params.Omega_r / (3 * a**2)
        )
        return a * H_conf
    
    def _compute_scale_factor_second_derivative(self, tau: float, a: float) -> float:
        """Compute d²a/dτ² from modified Friedmann equations"""
        a_prime = self._compute_scale_factor_derivative(tau, a)
        H_conf = a_prime / a
        
        # Second Friedmann equation with chronodynamic corrections
        acceleration = -4 * np.pi * a * (
            self.params.Omega_m / a**3 +
            2 * self.params.Omega_lambda * a +
            2 * self.params.Omega_r / a**4
        )
        
        # Chronodynamic contribution
        chrono_acceleration = self._compute_chronodynamic_acceleration(tau, a)
        
        return acceleration + chrono_acceleration
    
    def _compute_chronodynamic_acceleration(self, tau: float, a: float) -> float:
        """
        Compute chronodynamic contribution to acceleration.
        
        This represents the effect of dynamic time T(τ) on cosmic expansion.
        """
        T_val = self.T_function(tau, np.array([0, 0, 0]))  # At origin
        T_derivative = self._numerical_derivative(
            lambda t: self.T_function(t, np.array([0, 0, 0])), tau
        )
        
        # Chronodynamic acceleration term
        chrono_term = self.params.S_chrono * T_derivative / T_val
        
        return chrono_term * a
    
    def _numerical_derivative(self, func: Callable, x: float, h: float = 1e-8) -> float:
        """Compute numerical derivative using central difference"""
        return (func(x + h) - func(x - h)) / (2 * h)
    
    def compute_tensor_components(self, tau: float, x: np.ndarray) -> np.ndarray:
        """
        Compute all components of the chronodynamic tensor C_μν.
        
        Args:
            tau: Conformal time
            x: Spatial coordinates [x, y, z]
            
        Returns:
            4x4 tensor components C_μν
        """
        C = np.zeros((4, 4))
        
        # Get current scale factor (simplified)
        a = self._get_scale_factor(tau)
        
        # Time-time component C₀₀
        C[0, 0] = self._compute_C00(tau, x, a)
        
        # Time-space components C₀ᵢ
        for i in range(1, 4):
            C[0, i] = self._compute_C0i(tau, x, a, i-1)
            C[i, 0] = C[0, i]  # Symmetry
        
        # Space-space components Cᵢⱼ
        for i in range(1, 4):
            for j in range(1, 4):
                C[i, j] = self._compute_Cij(tau, x, a, i-1, j-1)
        
        return C
    
    def _get_scale_factor(self, tau: float) -> float:
        """Get scale factor at conformal time tau (simplified)"""
        # This should be integrated from the full system
        # For now, use approximate solution
        return np.exp(self.params.H0 * tau / 299792.458)  # c in km/s
    
    def _compute_C00(self, tau: float, x: np.ndarray, a: float) -> float:
        """
        Compute C₀₀ component (time-time).
        
        This component encodes the temporal compression/dilation effects
        from the dynamic time function T(τ).
        """
        T_val = self.T_function(tau, x)
        T_tau = self._numerical_derivative(
            lambda t: self.T_function(t, x), tau
        )
        
        # Chronodynamic time-time component
        C00 = self.params.S_chrono * (T_tau / T_val)**2
        
        return C00 / a**2  # Conformal factor
    
    def _compute_C0i(self, tau: float, x: np.ndarray, a: float, i: int) -> float:
        """
        Compute C₀ᵢ components (time-space).
        
        These encode the coupling between temporal and spatial variations.
        """
        T_val = self.T_function(tau, x)
        
        # Spatial derivative of T
        x_perturbed = x.copy()
        x_perturbed[i] += 1e-8
        T_xi = (self.T_function(tau, x_perturbed) - T_val) / 1e-8
        
        # Chronodynamic time-space component
        C0i = self.params.S_chrono * T_xi / T_val
        
        return C0i / a**2
    
    def _compute_Cij(self, tau: float, x: np.ndarray, a: float, i: int, j: int) -> float:
        """
        Compute Cᵢⱼ components (space-space).
        
        These encode spatial variations in the chronodynamic field.
        """
        if i == j:
            # Diagonal components
            T_val = self.T_function(tau, x)
            
            # Second spatial derivative
            h = 1e-6
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            T_xx = (self.T_function(tau, x_plus) - 2*T_val + 
                   self.T_function(tau, x_minus)) / h**2
            
            Cii = self.params.S_chrono * T_xx / T_val
        else:
            # Off-diagonal components (mixed derivatives)
            h = 1e-6
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += h
            x_pp[j] += h
            x_pm[i] += h
            x_pm[j] -= h
            x_mp[i] -= h
            x_mp[j] += h
            x_mm[i] -= h
            x_mm[j] -= h
            
            T_xy = (self.T_function(tau, x_pp) - self.T_function(tau, x_pm) -
                   self.T_function(tau, x_mp) + self.T_function(tau, x_mm)) / (4*h**2)
            
            T_val = self.T_function(tau, x)
            Cij = self.params.S_chrono * T_xy / T_val
        
        return Cij
    
    def compute_tensor_divergence(self, tau: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the divergence ∇_μ C^μν of the chronodynamic tensor.
        
        This must vanish for energy-momentum conservation.
        """
        divergence = np.zeros(4)
        h = 1e-6
        
        for mu in range(4):
            for nu in range(4):
                if mu == 0:  # Time derivative
                    C_plus = self.compute_tensor_components(tau + h, x)[mu, nu]
                    C_minus = self.compute_tensor_components(tau - h, x)[mu, nu]
                    div_contrib = (C_plus - C_minus) / (2 * h)
                else:  # Spatial derivatives
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[mu-1] += h
                    x_minus[mu-1] -= h
                    
                    C_plus = self.compute_tensor_components(tau, x_plus)[mu, nu]
                    C_minus = self.compute_tensor_components(tau, x_minus)[mu, nu]
                    div_contrib = (C_plus - C_minus) / (2 * h)
                
                divergence[nu] += div_contrib
        
        return divergence
    
    def compute_trace(self, tau: float, x: np.ndarray) -> float:
        """Compute the trace of the chronodynamic tensor"""
        C = self.compute_tensor_components(tau, x)
        return np.trace(C)
    
    def energy_momentum_source(self, tau: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the effective energy-momentum tensor from chronodynamic effects.
        
        T_eff^μν = -C^μν / (8π)
        """
        C = self.compute_tensor_components(tau, x)
        return -C / (8 * np.pi)
    
    def validate_conservation(self, tau: float, x: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Validate energy-momentum conservation.
        
        Check that ∇_μ T_eff^μν = 0
        """
        divergence = self.compute_tensor_divergence(tau, x)
        max_divergence = np.max(np.abs(divergence))
        
        is_conserved = max_divergence < tolerance
        
        if not is_conserved:
            logger.warning(f"Conservation violation: max_div = {max_divergence}")
        
        return is_conserved


class ChronodynamicEvolution:
    """
    Handles the coupled evolution of scale factor a(τ) and dynamic time T(τ).
    """
    
    def __init__(self, tensor: ChronodynamicTensor):
        self.tensor = tensor
        self.params = tensor.params
    
    def friedmann_equations_modified(self, tau: float, y: np.ndarray) -> np.ndarray:
        """
        Modified Friedmann equations including chronodynamic effects.
        
        Args:
            tau: Conformal time
            y: [a, a'] where a' = da/dτ
            
        Returns:
            [a', a''] derivatives
        """
        a, a_prime = y
        
        # Standard Friedmann contributions
        H_conf_squared = (
            8 * np.pi * self.params.Omega_m / (3 * a) +
            8 * np.pi * self.params.Omega_lambda * a**2 / 3 +
            8 * np.pi * self.params.Omega_r / (3 * a**2)
        )
        
        # Chronodynamic corrections
        x_origin = np.array([0, 0, 0])
        C = self.tensor.compute_tensor_components(tau, x_origin)
        
        # Modified second Friedmann equation
        a_double_prime = -4 * np.pi * a * (
            self.params.Omega_m / a**3 +
            2 * self.params.Omega_lambda * a +
            2 * self.params.Omega_r / a**4
        ) + a * C[0, 0]  # Chronodynamic contribution
        
        return np.array([a_prime, a_double_prime])
    
    def integrate_evolution(self, tau_span: Tuple[float, float], 
                          initial_conditions: np.ndarray,
                          n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Integrate the modified Friedmann equations.
        
        Args:
            tau_span: (tau_start, tau_end)
            initial_conditions: [a_0, a'_0]
            n_points: Number of integration points
            
        Returns:
            Dictionary with tau, a(tau), a'(tau) arrays
        """
        from scipy.integrate import solve_ivp
        
        tau_eval = np.linspace(tau_span[0], tau_span[1], n_points)
        
        solution = solve_ivp(
            self.friedmann_equations_modified,
            tau_span,
            initial_conditions,
            t_eval=tau_eval,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
        
        return {
            'tau': solution.t,
            'a': solution.y[0],
            'a_prime': solution.y[1],
            'H_conf': solution.y[1] / solution.y[0]  # Conformal Hubble
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize with default parameters
    params = CosmologicalParams()
    tensor = ChronodynamicTensor(params, grid_size=64)
    
    # Test tensor computation
    tau = 1.0
    x = np.array([0, 0, 0])
    
    print("Testing Chronodynamic Tensor Implementation")
    print("=" * 50)
    
    # Compute tensor components
    C = tensor.compute_tensor_components(tau, x)
    print(f"Tensor components C_μν at τ={tau}, x={x}:")
    print(C)
    
    # Test conservation
    is_conserved = tensor.validate_conservation(tau, x)
    print(f"Energy-momentum conservation: {is_conserved}")
    
    # Test evolution
    evolution = ChronodynamicEvolution(tensor)
    tau_span = (0.1, 10.0)
    initial_conditions = np.array([0.1, 0.01])  # Small initial scale factor
    
    try:
        result = evolution.integrate_evolution(tau_span, initial_conditions)
        print(f"Evolution computed successfully over τ ∈ [{tau_span[0]}, {tau_span[1]}]")
        print(f"Final scale factor: a(τ_final) = {result['a'][-1]:.6f}")
    except Exception as e:
        print(f"Evolution computation failed: {e}")