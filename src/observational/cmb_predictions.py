#!/usr/bin/env python3
"""
CMB Predictions for Chronodynamic Cosmology
============================================

Computation of Cosmic Microwave Background power spectra
for the Chronodynamic Cosmological Divergence (CCD) model.

Author: Aksel Boursier
Date: August 2025
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CMBConfig:
    """Configuration for CMB calculations"""
    l_max: int = 2500
    k_min: float = 1e-5
    k_max: float = 1.0
    n_k: int = 200
    z_recombination: float = 1090.0
    tau_reionization: float = 0.054

class ChronodynamicTransferFunction:
    """
    Computes transfer functions for chronodynamic perturbations.
    """
    
    def __init__(self, chronodynamic_tensor, config: CMBConfig = None):
        self.tensor = chronodynamic_tensor
        self.config = config or CMBConfig()
        self.params = chronodynamic_tensor.params
        
        self.k_array = np.logspace(
            np.log10(self.config.k_min),
            np.log10(self.config.k_max),
            self.config.n_k
        )
        
        self._precompute_background()
        
        logger.info(f"Initialized ChronodynamicTransferFunction with l_max={self.config.l_max}")

    def _precompute_background(self):
        """
        Solves the Friedmann equations to get a(τ) and H(τ) and creates
        interpolation functions for them.
        """
        logger.info("Pre-computing background cosmology...")
        from chronodynamic_sim.core.chronodynamic_tensor import ChronodynamicEvolution
        
        evolution = ChronodynamicEvolution(self.tensor)
        
        tau_ini = 1e-5
        # A rough estimate of tau_today to ensure the integration range is sufficient
        tau_today = 3 * (2 * 299792.458 / (self.params.H0 * np.sqrt(self.params.Omega_m)) * np.sqrt(1.0 / (1.0 + self.config.z_recombination)))
        tau_span = (tau_ini, tau_today)
        
        a_ini = tau_ini 
        a_prime_ini = a_ini * np.sqrt(
            8 * np.pi * self.params.Omega_m / (3 * a_ini) +
            8 * np.pi * self.params.Omega_r / (3 * a_ini**2)
        )
        initial_conditions = [a_ini, a_prime_ini]

        sol = solve_ivp(
            evolution.friedmann_equations_modified,
            tau_span,
            initial_conditions,
            dense_output=True,
            method='LSODA',
            rtol=1e-7, atol=1e-8
        )

        if not sol.success:
            raise RuntimeError(f"Background cosmology integration failed: {sol.message}")

        self.a_interp_func = sol.sol
        self.H_interp_func = lambda tau: self.a_interp_func(tau, 1)[0] / self.a_interp_func(tau, 0)[0]
        logger.info("Background cosmology pre-computed and interpolated.")

    def _scale_factor(self, tau: float) -> float:
        return self.a_interp_func(tau)[0]

    def _hubble_parameter(self, tau: float) -> float:
        return self.H_interp_func(tau)

    def solve_perturbation_equations(self, k: float) -> Dict[str, np.ndarray]:
        tau_ini = 1e-4
        tau_rec = self._conformal_time_at_recombination()
        
        if tau_ini >= tau_rec:
            raise ValueError(f"Initial time (τ_ini={tau_ini}) is not smaller than recombination time (τ_rec={tau_rec})")

        tau_array = np.logspace(np.log10(tau_ini), np.log10(tau_rec), 500)
        
        initial_conditions = self._adiabatic_initial_conditions(k, tau_ini)
        
        def perturbation_system(tau, y):
            return self._chronodynamic_perturbation_equations(tau, y, k)
        
        solution = solve_ivp(
            perturbation_system,
            (tau_ini, tau_rec),
            initial_conditions,
            t_eval=tau_array,
            method='LSODA',
            rtol=1e-6,
            atol=1e-7
        )
        
        if not solution.success:
            raise RuntimeError(f"Perturbation integration failed for k={k}: {solution.message}")
        
        return {
            'tau': solution.t, 'delta_c': solution.y[0], 'delta_b': solution.y[1],
            'delta_gamma': solution.y[2], 'theta_c': solution.y[3], 'theta_b': solution.y[4],
            'theta_gamma': solution.y[5], 'phi': solution.y[6], 'psi': solution.y[7]
        }
    
    def _conformal_time_at_recombination(self) -> float:
        a_rec = 1.0 / (1.0 + self.config.z_recombination)
        
        def objective(tau):
            return self.a_interp_func(tau)[0] - a_rec

        H0 = self.params.H0 / 299792.458
        tau_guess = 2 / (H0 * np.sqrt(self.params.Omega_m)) * np.sqrt(a_rec)
        
        try:
            tau_rec_solution = fsolve(objective, x0=tau_guess)
            return tau_rec_solution[0]
        except Exception as e:
            logger.error(f"Could not solve for recombination time: {e}")
            return tau_guess

    def _adiabatic_initial_conditions(self, k: float, tau: float) -> np.ndarray:
        A_s = 2.1e-9
        n_s = 0.965
        
        R_k = np.sqrt(A_s * (k / 0.05)**(n_s - 1))
        
        phi = (1/3) * R_k
        delta_gamma = -2 * phi
        delta_b = 0.75 * delta_gamma
        delta_c = delta_b
        theta_gamma = - (k**2 * tau) / 6 * R_k
        theta_b = theta_gamma
        
        return np.array([
            delta_c, delta_b, delta_gamma,
            0, theta_b, theta_gamma, phi, -phi
        ])

    def _chronodynamic_perturbation_equations(self, tau: float, y: np.ndarray, k: float) -> np.ndarray:
        delta_c, delta_b, delta_gamma, theta_c, theta_b, theta_gamma, phi, psi = y
        
        a = self._scale_factor(tau)
        H = self._hubble_parameter(tau)
        
        x_origin = np.array([0, 0, 0])
        C_tensor = self.tensor.compute_tensor_components(tau, x_origin)
        
        S_spatial = self.params.S_chrono * np.trace(C_tensor[1:, 1:]) / 3
        
        z = (1.0 / a) - 1.0
        if z > 1200: xe = 1.0
        elif z < 900: xe = 1e-4
        else: xe = 0.5 * (1 + np.tanh((1050 - z) / 50))
        
        n_H = 1.9e-4
        sigma_T = 4.78e-48
        tau_dot = xe * n_H * (1+z)**3 * sigma_T * a

        phi_dot = -H * phi + (k**2 / (3*a**2)) * (delta_c + delta_b)
        psi_dot = -H * psi
        
        delta_c_dot = -k * theta_c + 3 * phi_dot
        theta_c_dot = -H * theta_c - (k/a) * psi

        delta_b_dot = -k * theta_b + 3 * phi_dot
        theta_b_dot = -H * theta_b - (k/a) * psi + tau_dot * (theta_gamma - theta_b)

        delta_gamma_dot = -(4/3) * k * theta_gamma + 4 * phi_dot
        theta_gamma_dot = (k / 4) * delta_gamma - (k/a) * psi + tau_dot * (theta_b - theta_gamma)

        theta_c_dot += S_spatial * k * psi
        theta_b_dot += S_spatial * k * psi
        theta_gamma_dot += S_spatial * k * psi / 4
        
        return np.array([
            delta_c_dot, delta_b_dot, delta_gamma_dot,
            theta_c_dot, theta_b_dot, theta_gamma_dot,
            phi_dot, psi_dot
        ])

    def _redshift_from_tau(self, tau: float) -> float:
        a = self._scale_factor(tau)
        if a == 0: return np.inf
        return 1.0/a - 1.0

class CMBPredictor:
    """
    Main class for computing CMB power spectra in chronodynamic cosmology.
    """
    
    def __init__(self, chronodynamic_tensor, config: CMBConfig = None):
        self.tensor = chronodynamic_tensor
        self.config = config or CMBConfig()
        self.transfer = ChronodynamicTransferFunction(chronodynamic_tensor, config)
        
        self.l_array = np.arange(2, self.config.l_max + 1)
        
        logger.info("Initialized CMBPredictor")
    
    def compute_power_spectra(self) -> Dict[str, np.ndarray]:
        """
        Compute temperature and polarization power spectra.
        """
        logger.info("Computing CMB power spectra")
        
        C_l_TT = np.zeros(len(self.l_array))
        C_l_TE = np.zeros(len(self.l_array))
        C_l_EE = np.zeros(len(self.l_array))
        
        for i, k in enumerate(self.transfer.k_array):
            if i % 20 == 0:
                logger.info(f"Processing k mode {i+1}/{len(self.transfer.k_array)}")
            
            try:
                perturbations = self.transfer.solve_perturbation_equations(k)
            except Exception as e:
                logger.warning(f"Skipping k={k} due to error: {e}")
                continue
            
            tau_rec = perturbations['tau'][-1]
            delta_gamma_rec = perturbations['delta_gamma'][-1]
            theta_gamma_rec = perturbations['theta_gamma'][-1]
            phi_rec = perturbations['phi'][-1]
            
            S_T = delta_gamma_rec / 4 + phi_rec
            S_E = theta_gamma_rec / k
            
            for j, l in enumerate(self.l_array):
                chi_rec = self._comoving_distance_recombination()
                x = k * chi_rec
                j_l = self._spherical_bessel(l, x)
                P_k = self._primordial_power_spectrum(k)
                dk = np.diff(self.transfer.k_array)[0] if i < len(self.transfer.k_array)-1 else 0.01
                
                C_l_TT[j] += P_k * S_T**2 * j_l**2 * dk
                C_l_TE[j] += P_k * S_T * S_E * j_l**2 * dk  
                C_l_EE[j] += P_k * S_E**2 * j_l**2 * dk
        
        T_CMB = 2.725e6
        C_l_TT *= T_CMB**2 * (2*np.pi)**2
        C_l_TE *= T_CMB**2 * (2*np.pi)**2
        C_l_EE *= T_CMB**2 * (2*np.pi)**2
        
        l_factor = self.l_array * (self.l_array + 1) / (2 * np.pi)
        
        return {
            'l': self.l_array,
            'TT': C_l_TT * l_factor,
            'TE': C_l_TE * l_factor,
            'EE': C_l_EE * l_factor
        }
    
    def _comoving_distance_recombination(self) -> float:
        z_rec = self.config.z_recombination
        H0 = self.tensor.params.H0
        c = 299792.458
        chi_rec = c / H0 * 2 * np.sqrt(1 + z_rec)
        return chi_rec
    
    def _spherical_bessel(self, l: int, x: float) -> float:
        from scipy.special import spherical_jn
        return spherical_jn(l, x)
    
    def _primordial_power_spectrum(self, k: float) -> float:
        A_s = 2.1e-9
        n_s = 0.965
        k_pivot = 0.05
        return A_s * (k / k_pivot)**(n_s - 1)
    
    def compute_chronodynamic_signatures(self) -> Dict[str, np.ndarray]:
        logger.info("Computing chronodynamic signatures")
        
        C_l_standard = self._compute_standard_cmb()
        C_l_ccd = self.compute_power_spectra()
        
        signatures = {
            'l': self.l_array,
            'delta_TT': C_l_ccd['TT'] - C_l_standard['TT'],
            'ratio_TT': C_l_ccd['TT'] / C_l_standard['TT'],
            'delta_TE': C_l_ccd['TE'] - C_l_standard['TE'],
            'ratio_TE': C_l_ccd['TE'] / C_l_standard['TE']
        }
        
        signatures['peak_shifts'] = self._detect_peak_shifts(C_l_ccd, C_l_standard)
        signatures['amplitude_changes'] = self._detect_amplitude_changes(C_l_ccd, C_l_standard)
        
        return signatures
    
    def _compute_standard_cmb(self) -> Dict[str, np.ndarray]:
        l_array = self.l_array
        C_l_TT_std = 6000 * np.exp(-(l_array - 220)**2 / (2 * 50**2))
        C_l_TT_std += 3000 * np.exp(-(l_array - 540)**2 / (2 * 40**2))
        C_l_TT_std += 1500 * np.exp(-(l_array - 800)**2 / (2 * 35**2))
        damping = np.exp(-(l_array / 1000)**2)
        C_l_TT_std *= damping
        C_l_TE_std = 0.3 * C_l_TT_std
        C_l_EE_std = 0.1 * C_l_TT_std
        return {
            'l': l_array, 'TT': C_l_TT_std, 'TE': C_l_TE_std, 'EE': C_l_EE_std
        }
    
    def _detect_peak_shifts(self, C_l_ccd: Dict, C_l_standard: Dict) -> Dict:
        from scipy.signal import find_peaks
        peaks_std, _ = find_peaks(C_l_standard['TT'], height=1000, distance=100)
        peaks_ccd, _ = find_peaks(C_l_ccd['TT'], height=1000, distance=100)
        peak_shifts = []
        if len(peaks_std) > 0 and len(peaks_ccd) > 0:
            for i in range(min(len(peaks_std), len(peaks_ccd))):
                l_std = self.l_array[peaks_std[i]]
                l_ccd = self.l_array[peaks_ccd[i]]
                shift = (l_ccd - l_std) / l_std
                peak_shifts.append(shift)
        return {
            'peaks_standard': peaks_std, 'peaks_ccd': peaks_ccd, 'relative_shifts': peak_shifts
        }
    
    def _detect_amplitude_changes(self, C_l_ccd: Dict, C_l_standard: Dict) -> Dict:
        l_ranges = {
            'low_l': (2, 50), 'first_peak': (150, 300),
            'second_peak': (400, 600), 'damping_tail': (1000, 2000)
        }
        amplitude_changes = {}
        for range_name, (l_min, l_max) in l_ranges.items():
            mask = (self.l_array >= l_min) & (self.l_array <= l_max)
            if np.any(mask):
                mean_std = np.mean(C_l_standard['TT'][mask])
                mean_ccd = np.mean(C_l_ccd['TT'][mask])
                relative_change = (mean_ccd - mean_std) / mean_std
                amplitude_changes[range_name] = relative_change
        return amplitude_changes