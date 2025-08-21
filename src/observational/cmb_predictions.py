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
from scipy.interpolate import interp1d, UnivariateSpline
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CMBConfig:
    """Configuration for CMB calculations"""
    l_max: int = 2500          # Maximum multipole
    k_min: float = 1e-5        # Minimum wavenumber [Mpc⁻¹]
    k_max: float = 1.0         # Maximum wavenumber [Mpc⁻¹] 
    n_k: int = 200             # Number of k modes
    z_recombination: float = 1090.0  # Recombination redshift
    tau_reionization: float = 0.054  # Reionization optical depth
    

class ChronodynamicTransferFunction:
    """
    Computes transfer functions for chronodynamic perturbations.
    
    The chronodynamic tensor C_μν modifies the evolution of density
    and gravitational potential perturbations, leading to distinctive
    signatures in the CMB power spectra.
    """
    
    def __init__(self, chronodynamic_tensor, config: CMBConfig = None):
        self.tensor = chronodynamic_tensor
        self.config = config or CMBConfig()
        self.params = chronodynamic_tensor.params
        
        # Wave number array
        self.k_array = np.logspace(
            np.log10(self.config.k_min),
            np.log10(self.config.k_max),
            self.config.n_k
        )
        
        logger.info(f"Initialized ChronodynamicTransferFunction with l_max={self.config.l_max}")
    
    def solve_perturbation_equations(self, k: float) -> Dict[str, np.ndarray]:
        """
        Solve the modified perturbation equations for a given k mode.
        
        The chronodynamic modifications appear as additional source terms
        in the Einstein-Boltzmann equations.
        
        Args:
            k: Comoving wavenumber [Mpc⁻¹]
            
        Returns:
            Dictionary with perturbation solutions
        """
        # Conformal time array (from early universe to recombination)
        tau_ini = 1e-4  # Initial conformal time
        tau_rec = self._conformal_time_at_recombination()
        tau_array = np.logspace(np.log10(tau_ini), np.log10(tau_rec), 1000)
        
        # Initial conditions (adiabatic)
        initial_conditions = self._adiabatic_initial_conditions(k)
        
        # Solve system
        def perturbation_system(tau, y):
            return self._chronodynamic_perturbation_equations(tau, y, k)
        
        solution = solve_ivp(
            perturbation_system,
            (tau_ini, tau_rec),
            initial_conditions,
            t_eval=tau_array,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        if not solution.success:
            logger.error(f"Perturbation evolution failed for k={k}")
            raise RuntimeError("Perturbation integration failed")
        
        return {
            'tau': solution.t,
            'delta_c': solution.y[0],      # Cold dark matter perturbation
            'delta_b': solution.y[1],      # Baryon perturbation  
            'delta_gamma': solution.y[2],  # Photon perturbation
            'theta_c': solution.y[3],      # CDM velocity
            'theta_b': solution.y[4],      # Baryon velocity
            'theta_gamma': solution.y[5],  # Photon velocity
            'phi': solution.y[6],          # Gravitational potential
            'psi': solution.y[7]           # Curvature perturbation
        }
    
    def _conformal_time_at_recombination(self) -> float:
        """Compute conformal time at recombination"""
        # Simplified calculation - in practice, integrate from scale factor evolution
        z_rec = self.config.z_recombination
        a_rec = 1.0 / (1.0 + z_rec)
        
        # Approximate conformal time (should be computed from Friedmann equations)
        H0 = self.params.H0  # km/s/Mpc
        c = 299792.458  # km/s
        
        # Very rough approximation
        tau_rec = 2 * c / (H0 * np.sqrt(self.params.Omega_m)) * np.sqrt(a_rec)
        return tau_rec
    
    def _adiabatic_initial_conditions(self, k: float) -> np.ndarray:
        """Set adiabatic initial conditions for perturbations"""
        # Primordial curvature perturbation amplitude
        A_s = 2.1e-9  # Scalar amplitude
        n_s = 0.965   # Spectral index
        
        # Initial curvature perturbation
        zeta_k = np.sqrt(A_s * (k / 0.05)**(n_s - 1))
        
        # Adiabatic initial conditions
        delta_c_ini = -1.5 * zeta_k
        delta_b_ini = -1.5 * zeta_k  
        delta_gamma_ini = -2.0 * zeta_k
        theta_c_ini = 0.5 * k * zeta_k
        theta_b_ini = 0.5 * k * zeta_k
        theta_gamma_ini = 0.5 * k * zeta_k
        phi_ini = zeta_k
        psi_ini = zeta_k
        
        return np.array([
            delta_c_ini, delta_b_ini, delta_gamma_ini,
            theta_c_ini, theta_b_ini, theta_gamma_ini,
            phi_ini, psi_ini
        ])
    
    def _chronodynamic_perturbation_equations(self, tau: float, y: np.ndarray, k: float) -> np.ndarray:
        """
        Modified Einstein-Boltzmann equations with chronodynamic corrections.
        
        The chronodynamic tensor C_μν introduces additional source terms
        that modify the standard perturbation evolution.
        """
        # Extract variables
        delta_c, delta_b, delta_gamma, theta_c, theta_b, theta_gamma, phi, psi = y
        
        # Background quantities
        a = self._scale_factor(tau)
        H = self._hubble_parameter(tau)
        
        # Chronodynamic corrections
        x_origin = np.array([0, 0, 0])
        C_tensor = self.tensor.compute_tensor_components(tau, x_origin)
        
        # Chronodynamic source terms
        S_chrono = self.params.S_chrono * C_tensor[0, 0]  # Time-time component
        S_spatial = self.params.S_chrono * np.trace(C_tensor[1:, 1:]) / 3  # Spatial trace
        
        # Modified equations
        # CDM perturbations
        ddelta_c_dtau = -theta_c + 3 * (phi / tau) * (1 + S_chrono)
        dtheta_c_dtau = -H * theta_c + k**2 * psi * (1 + S_spatial)
        
        # Baryon perturbations (with Thomson scattering)
        n_e = self._electron_density(tau)  # Free electron density
        sigma_T = 6.65e-29  # Thomson scattering cross-section [m²]
        tau_dot = n_e * sigma_T * a  # Optical depth derivative
        
        ddelta_b_dtau = -theta_b + 3 * (phi / tau) * (1 + S_chrono)
        dtheta_b_dtau = (-H * theta_b + k**2 * psi * (1 + S_spatial) + 
                        tau_dot * (theta_gamma - theta_b))
        
        # Photon perturbations
        ddelta_gamma_dtau = -4/3 * theta_gamma + 4 * (phi / tau) * (1 + S_chrono)
        dtheta_gamma_dtau = (k**2 * (delta_gamma/4 + psi) * (1 + S_spatial) +
                            tau_dot * (theta_b - theta_gamma))
        
        # Gravitational potentials (modified Poisson equations)
        k2 = k**2
        dphi_dtau = psi - k2 * phi / (3 * H**2) * (1 + S_spatial)
        dpsi_dtau = -H * psi - H * phi * (1 + S_chrono)
        
        return np.array([
            ddelta_c_dtau, ddelta_b_dtau, ddelta_gamma_dtau,
            dtheta_c_dtau, dtheta_b_dtau, dtheta_gamma_dtau,
            dphi_dtau, dpsi_dtau
        ])
    
    def _scale_factor(self, tau: float) -> float:
        """Scale factor as function of conformal time"""
        # This should come from the chronodynamic evolution
        # For now, use approximate solution
        return tau**2 / (2 * self.params.H0)
    
    def _hubble_parameter(self, tau: float) -> float:
        """Hubble parameter H = a'/a in conformal time"""
        # Derivative of scale factor
        return 2.0 / tau  # From a ∝ τ²
    
    def _electron_density(self, tau: float) -> float:
        """Free electron density (simplified recombination history)"""
        z = self._redshift_from_tau(tau)
        
        # Simplified recombination (should use RECFAST or similar)
        if z > 1200:
            x_e = 1.0  # Fully ionized
        elif z < 900:
            x_e = 1e-4  # Fully recombined
        else:
            # Smooth transition
            x_e = 0.5 * (1 + np.tanh((1050 - z) / 50))
        
        # Electron density
        n_H = 1.9e5  # Hydrogen number density today [m⁻³]
        Omega_b = 0.049  # Baryon density
        return x_e * n_H * Omega_b * (1 + z)**3
    
    def _redshift_from_tau(self, tau: float) -> float:
        """Convert conformal time to redshift"""
        a = self._scale_factor(tau)
        return 1.0/a - 1.0


class CMBPredictor:
    """
    Main class for computing CMB power spectra in chronodynamic cosmology.
    """
    
    def __init__(self, chronodynamic_tensor, config: CMBConfig = None):
        self.tensor = chronodynamic_tensor
        self.config = config or CMBConfig()
        self.transfer = ChronodynamicTransferFunction(chronodynamic_tensor, config)
        
        # Multipole array
        self.l_array = np.arange(2, self.config.l_max + 1)
        
        logger.info("Initialized CMBPredictor")
    
    def compute_power_spectra(self) -> Dict[str, np.ndarray]:
        """
        Compute temperature and polarization power spectra.
        
        Returns:
            Dictionary with C_l^TT, C_l^TE, C_l^EE power spectra
        """
        logger.info("Computing CMB power spectra")
        
        # Initialize power spectra
        C_l_TT = np.zeros(len(self.l_array))
        C_l_TE = np.zeros(len(self.l_array))
        C_l_EE = np.zeros(len(self.l_array))
        
        # Loop over k modes
        for i, k in enumerate(self.transfer.k_array):
            if i % 20 == 0:
                logger.info(f"Processing k mode {i+1}/{len(self.transfer.k_array)}")
            
            # Solve perturbation equations
            try:
                perturbations = self.transfer.solve_perturbation_equations(k)
            except Exception as e:
                logger.warning(f"Skipping k={k} due to error: {e}")
                continue
            
            # Compute transfer functions at recombination
            tau_rec = perturbations['tau'][-1]
            delta_gamma_rec = perturbations['delta_gamma'][-1]
            theta_gamma_rec = perturbations['theta_gamma'][-1]
            phi_rec = perturbations['phi'][-1]
            
            # Source functions for temperature and polarization
            S_T = delta_gamma_rec / 4 + phi_rec  # Temperature source
            S_E = theta_gamma_rec / k            # E-mode polarization source
            
            # Spherical Bessel functions for projection
            for j, l in enumerate(self.l_array):
                # Comoving distance to recombination
                chi_rec = self._comoving_distance_recombination()
                x = k * chi_rec
                
                # Spherical Bessel function j_l(x)
                j_l = self._spherical_bessel(l, x)
                
                # Power spectrum contributions
                P_k = self._primordial_power_spectrum(k)
                
                # Integrate over k (simplified - should use proper integration)
                dk = np.diff(self.transfer.k_array)[0] if i < len(self.transfer.k_array)-1 else 0.01
                
                C_l_TT[j] += P_k * S_T**2 * j_l**2 * dk
                C_l_TE[j] += P_k * S_T * S_E * j_l**2 * dk  
                C_l_EE[j] += P_k * S_E**2 * j_l**2 * dk
        
        # Apply normalization and units
        # Convert to μK² units
        T_CMB = 2.725e6  # CMB temperature in μK
        
        C_l_TT *= T_CMB**2 * (2*np.pi)**2
        C_l_TE *= T_CMB**2 * (2*np.pi)**2
        C_l_EE *= T_CMB**2 * (2*np.pi)**2
        
        # Apply l(l+1) normalization
        l_factor = self.l_array * (self.l_array + 1) / (2 * np.pi)
        
        return {
            'l': self.l_array,
            'TT': C_l_TT * l_factor,
            'TE': C_l_TE * l_factor,
            'EE': C_l_EE * l_factor
        }
    
    def _comoving_distance_recombination(self) -> float:
        """Comoving distance to recombination surface"""
        # Simplified calculation
        z_rec = self.config.z_recombination
        H0 = self.tensor.params.H0
        c = 299792.458  # km/s
        
        # Approximate comoving distance
        chi_rec = c / H0 * 2 * np.sqrt(1 + z_rec)  # Very rough approximation
        return chi_rec
    
    def _spherical_bessel(self, l: int, x: float) -> float:
        """Spherical Bessel function j_l(x)"""
        from scipy.special import spherical_jn
        return spherical_jn(l, x)
    
    def _primordial_power_spectrum(self, k: float) -> float:
        """Primordial power spectrum P(k)"""
        # Scale-invariant spectrum with running
        A_s = 2.1e-9
        n_s = 0.965
        k_pivot = 0.05  # Mpc⁻¹
        
        return A_s * (k / k_pivot)**(n_s - 1)
    
    def compute_chronodynamic_signatures(self) -> Dict[str, np.ndarray]:
        """
        Compute specific chronodynamic signatures in CMB.
        
        These are observational predictions unique to the CCD model.
        """
        logger.info("Computing chronodynamic signatures")
        
        # Standard ΛCDM prediction (for comparison)
        # This would require running without chronodynamic corrections
        C_l_standard = self._compute_standard_cmb()
        
        # CCD prediction
        C_l_ccd = self.compute_power_spectra()
        
        # Differences and ratios
        signatures = {
            'l': self.l_array,
            'delta_TT': C_l_ccd['TT'] - C_l_standard['TT'],
            'ratio_TT': C_l_ccd['TT'] / C_l_standard['TT'],
            'delta_TE': C_l_ccd['TE'] - C_l_standard['TE'],
            'ratio_TE': C_l_ccd['TE'] / C_l_standard['TE']
        }
        
        # Characteristic features
        signatures['peak_shifts'] = self._detect_peak_shifts(C_l_ccd, C_l_standard)
        signatures['amplitude_changes'] = self._detect_amplitude_changes(C_l_ccd, C_l_standard)
        
        return signatures
    
    def _compute_standard_cmb(self) -> Dict[str, np.ndarray]:
        """Compute standard ΛCDM CMB prediction for comparison"""
        # This is a placeholder - would implement standard calculation
        # or interface with existing code like CAMB/CLASS
        
        # Mock standard prediction
        l_array = self.l_array
        
        # Approximate ΛCDM TT spectrum shape
        C_l_TT_std = 6000 * np.exp(-(l_array - 220)**2 / (2 * 50**2))  # First peak
        C_l_TT_std += 3000 * np.exp(-(l_array - 540)**2 / (2 * 40**2))  # Second peak
        C_l_TT_std += 1500 * np.exp(-(l_array - 800)**2 / (2 * 35**2))  # Third peak
        
        # Add damping tail
        damping = np.exp(-(l_array / 1000)**2)
        C_l_TT_std *= damping
        
        C_l_TE_std = 0.3 * C_l_TT_std  # Rough TE correlation
        C_l_EE_std = 0.1 * C_l_TT_std  # Rough EE amplitude
        
        return {
            'l': l_array,
            'TT': C_l_TT_std,
            'TE': C_l_TE_std,
            'EE': C_l_EE_std
        }
    
    def _detect_peak_shifts(self, C_l_ccd: Dict, C_l_standard: Dict) -> Dict:
        """Detect shifts in acoustic peak positions"""
        # Find peaks in both spectra
        from scipy.signal import find_peaks
        
        peaks_std, _ = find_peaks(C_l_standard['TT'], height=1000, distance=100)
        peaks_ccd, _ = find_peaks(C_l_ccd['TT'], height=1000, distance=100)
        
        # Compute peak shifts
        peak_shifts = []
        if len(peaks_std) > 0 and len(peaks_ccd) > 0:
            for i in range(min(len(peaks_std), len(peaks_ccd))):
                l_std = self.l_array[peaks_std[i]]
                l_ccd = self.l_array[peaks_ccd[i]]
                shift = (l_ccd - l_std) / l_std
                peak_shifts.append(shift)
        
        return {
            'peaks_standard': peaks_std,
            'peaks_ccd': peaks_ccd,
            'relative_shifts': peak_shifts
        }
    
    def _detect_amplitude_changes(self, C_l_ccd: Dict, C_l_standard: Dict) -> Dict:
        """Detect amplitude changes in different l ranges"""
        # Define l ranges for analysis
        l_ranges = {
            'low_l': (2, 50),      # Large-scale ISW
            'first_peak': (150, 300),  # First acoustic peak
            'second_peak': (400, 600), # Second acoustic peak
            'damping_tail': (1000, 2000)  # Small-scale damping
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


# Example usage and testing
if __name__ == "__main__":
    print("Testing CMB Predictions for Chronodynamic Cosmology")
    print("=" * 60)
    
    # Import required modules
    from ..core.chronodynamic_tensor import CosmologicalParams, ChronodynamicTensor
    
    # Initialize with test parameters
    params = CosmologicalParams(
        H0=67.4, Omega_m=0.315, Omega_lambda=0.685,
        S_chrono=1.0, T0_scale=1.0
    )
    
    tensor = ChronodynamicTensor(params, grid_size=64)
    
    # Test CMB prediction
    config = CMBConfig(l_max=1000, n_k=50)  # Reduced for testing
    predictor = CMBPredictor(tensor, config)
    
    try:
        # Test transfer function
        print("Testing transfer function computation...")
        k_test = 0.1  # Mpc⁻¹
        perturbations = predictor.transfer.solve_perturbation_equations(k_test)
        print(f"Transfer function computed for k={k_test}")
        
        # Test power spectrum computation (simplified)
        print("Testing power spectrum computation...")
        # This would take time for full calculation
        # power_spectra = predictor.compute_power_spectra()
        # print(f"Power spectra computed: l_max={np.max(power_spectra['l'])}")
        
        print("CMB prediction tests completed successfully")
        
    except Exception as e:
        print(f"CMB prediction test failed: {e}")