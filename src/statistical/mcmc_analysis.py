#!/usr/bin/env python3
"""
MCMC Analysis for Chronodynamic Cosmology
==========================================

Bayesian parameter estimation and model comparison for the
Chronodynamic Cosmological Divergence (CCD) model.

Author: Aksel Boursier
Date: August 2025
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy import optimize
from typing import Dict, List, Tuple, Callable, Optional
import logging
from dataclasses import dataclass
import h5py
import json

logger = logging.getLogger(__name__)


@dataclass
class MCMCConfig:
    """Configuration for MCMC analysis"""
    nwalkers: int = 64
    nsteps: int = 10000
    nburn: int = 2000
    thin: int = 1
    progress: bool = True
    parallel: bool = True
    moves: Optional[str] = None  # emcee moves
    
    
@dataclass 
class ParameterPriors:
    """Prior distributions for CCD model parameters"""
    # Cosmological parameters
    Omega_m: Tuple[float, float] = (0.1, 0.5)      # Matter density
    Omega_lambda: Tuple[float, float] = (0.5, 0.9)  # Dark energy density  
    H0_local: Tuple[float, float] = (65.0, 75.0)    # Local Hubble constant
    
    # Chronodynamic parameters
    S_chrono: Tuple[float, float] = (0.1, 2.0)      # Chronodynamic coupling
    T0_scale: Tuple[float, float] = (0.5, 1.5)      # Time scale factor
    q0_decel: Tuple[float, float] = (-1.0, 0.0)     # Deceleration parameter
    j0_jerk: Tuple[float, float] = (-2.0, 2.0)      # Jerk parameter


class ChronodynamicLikelihood:
    """
    Likelihood computation for chronodynamic cosmology.
    
    Compares theoretical predictions with observational data including:
    - CMB power spectra (Planck)
    - Type Ia supernovae (distance-redshift)
    - Baryon Acoustic Oscillations
    - Local H0 measurements
    """
    
    def __init__(self, observational_data: Dict):
        """
        Initialize with observational datasets.
        
        Args:
            observational_data: Dictionary containing observational datasets
        """
        self.data = observational_data
        self.theory_cache = {}  # Cache for expensive computations
        
        logger.info("Initialized ChronodynamicLikelihood")
    
    def log_likelihood(self, theta: np.ndarray) -> float:
        """
        Compute log-likelihood for given parameter vector.
        
        Args:
            theta: Parameter vector [Omega_m, Omega_lambda, H0, S_chrono, T0, q0, j0]
            
        Returns:
            Log-likelihood value
        """
        # Extract parameters
        Omega_m, Omega_lambda, H0, S_chrono, T0_scale, q0, j0 = theta
        
        # Check parameter bounds
        if not self._check_parameter_bounds(theta):
            return -np.inf
        
        # Compute theoretical predictions
        try:
            theory_predictions = self._compute_theory_predictions(theta)
        except Exception as e:
            logger.warning(f"Theory computation failed for θ={theta}: {e}")
            return -np.inf
        
        # Compute likelihood components
        log_like = 0.0
        
        # CMB likelihood
        if 'cmb' in self.data:
            log_like += self._cmb_likelihood(theory_predictions['cmb'], self.data['cmb'])
        
        # Supernovae likelihood
        if 'sne' in self.data:
            log_like += self._sne_likelihood(theory_predictions['sne'], self.data['sne'])
        
        # BAO likelihood
        if 'bao' in self.data:
            log_like += self._bao_likelihood(theory_predictions['bao'], self.data['bao'])
        
        # Local H0 likelihood
        if 'h0_local' in self.data:
            log_like += self._h0_likelihood(H0, self.data['h0_local'])
        
        return log_like
    
    def _check_parameter_bounds(self, theta: np.ndarray) -> bool:
        """Check if parameters are within physical bounds"""
        Omega_m, Omega_lambda, H0, S_chrono, T0_scale, q0, j0 = theta
        
        # Physical constraints
        if Omega_m < 0 or Omega_m > 1:
            return False
        if Omega_lambda < 0 or Omega_lambda > 1:
            return False
        if Omega_m + Omega_lambda > 1.1:  # Allow small curvature
            return False
        if H0 < 50 or H0 > 100:
            return False
        if S_chrono < 0 or S_chrono > 5:
            return False
        if T0_scale <= 0 or T0_scale > 3:
            return False
        
        return True
    
    def _compute_theory_predictions(self, theta: np.ndarray) -> Dict:
        """
        Compute theoretical predictions for given parameters.
        
        This is the computationally expensive part that interfaces
        with the chronodynamic simulation code.
        """
        # Create cache key
        cache_key = tuple(theta)
        if cache_key in self.theory_cache:
            return self.theory_cache[cache_key]
        
        # Import chronodynamic modules
        from ..core.chronodynamic_tensor import CosmologicalParams, ChronodynamicTensor
        from ..observational.cmb_predictions import CMBPredictor
        from ..observational.distance_redshift import DistanceCalculator
        
        # Set up parameters
        Omega_m, Omega_lambda, H0, S_chrono, T0_scale, q0, j0 = theta
        params = CosmologicalParams(
            H0=H0, Omega_m=Omega_m, Omega_lambda=Omega_lambda,
            S_chrono=S_chrono, T0_scale=T0_scale, 
            q0_decel=q0, j0_jerk=j0
        )
        
        # Initialize chronodynamic tensor
        tensor = ChronodynamicTensor(params, grid_size=128)
        
        predictions = {}
        
        # CMB predictions
        if 'cmb' in self.data:
            cmb_predictor = CMBPredictor(tensor)
            predictions['cmb'] = cmb_predictor.compute_power_spectra()
        
        # Distance-redshift predictions
        if 'sne' in self.data:
            distance_calc = DistanceCalculator(tensor)
            z_sne = self.data['sne']['redshift']
            predictions['sne'] = distance_calc.luminosity_distance(z_sne)
        
        # BAO predictions
        if 'bao' in self.data:
            z_bao = self.data['bao']['redshift']
            predictions['bao'] = distance_calc.angular_diameter_distance(z_bao)
        
        # Cache results
        self.theory_cache[cache_key] = predictions
        
        return predictions
    
    def _cmb_likelihood(self, theory_cl: Dict, data_cl: Dict) -> float:
        """Compute CMB likelihood"""
        log_like = 0.0
        
        for spectrum in ['TT', 'TE', 'EE']:
            if spectrum in theory_cl and spectrum in data_cl:
                theory = theory_cl[spectrum]
                data = data_cl[spectrum]
                errors = data_cl[f'{spectrum}_err']
                
                # Gaussian likelihood
                chi2 = np.sum((theory - data)**2 / errors**2)
                log_like -= 0.5 * chi2
        
        return log_like
    
    def _sne_likelihood(self, theory_mu: np.ndarray, data_sne: Dict) -> float:
        """Compute Type Ia supernovae likelihood"""
        data_mu = data_sne['distance_modulus']
        errors = data_sne['errors']
        
        # Include intrinsic scatter
        sigma_int = 0.1  # Intrinsic scatter in magnitude
        total_errors = np.sqrt(errors**2 + sigma_int**2)
        
        chi2 = np.sum((theory_mu - data_mu)**2 / total_errors**2)
        return -0.5 * chi2
    
    def _bao_likelihood(self, theory_da: np.ndarray, data_bao: Dict) -> float:
        """Compute BAO likelihood"""
        data_da = data_bao['angular_diameter_distance']
        errors = data_bao['errors']
        
        chi2 = np.sum((theory_da - data_da)**2 / errors**2)
        return -0.5 * chi2
    
    def _h0_likelihood(self, theory_h0: float, data_h0: Dict) -> float:
        """Compute local H0 measurement likelihood"""
        measured_h0 = data_h0['value']
        error = data_h0['error']
        
        return -0.5 * (theory_h0 - measured_h0)**2 / error**2


class ChronodynamicMCMC:
    """
    Main MCMC analysis class for chronodynamic cosmology.
    """
    
    def __init__(self, 
                 likelihood: ChronodynamicLikelihood,
                 priors: ParameterPriors,
                 config: MCMCConfig = None):
        """
        Initialize MCMC analysis.
        
        Args:
            likelihood: Likelihood function
            priors: Parameter priors
            config: MCMC configuration
        """
        self.likelihood = likelihood
        self.priors = priors
        self.config = config or MCMCConfig()
        
        # Parameter names and bounds
        self.param_names = [
            'Omega_m', 'Omega_lambda', 'H0_local', 
            'S_chrono', 'T0_scale', 'q0_decel', 'j0_jerk'
        ]
        self.ndim = len(self.param_names)
        
        # MCMC state
        self.sampler = None
        self.chain = None
        self.log_prob = None
        
        logger.info(f"Initialized ChronodynamicMCMC with {self.ndim} parameters")
    
    def log_prior(self, theta: np.ndarray) -> float:
        """Compute log-prior probability"""
        Omega_m, Omega_lambda, H0, S_chrono, T0_scale, q0, j0 = theta
        
        log_prior = 0.0
        
        # Flat priors within bounds
        if not (self.priors.Omega_m[0] <= Omega_m <= self.priors.Omega_m[1]):
            return -np.inf
        if not (self.priors.Omega_lambda[0] <= Omega_lambda <= self.priors.Omega_lambda[1]):
            return -np.inf
        if not (self.priors.H0_local[0] <= H0 <= self.priors.H0_local[1]):
            return -np.inf
        if not (self.priors.S_chrono[0] <= S_chrono <= self.priors.S_chrono[1]):
            return -np.inf
        if not (self.priors.T0_scale[0] <= T0_scale <= self.priors.T0_scale[1]):
            return -np.inf
        if not (self.priors.q0_decel[0] <= q0 <= self.priors.q0_decel[1]):
            return -np.inf
        if not (self.priors.j0_jerk[0] <= j0 <= self.priors.j0_jerk[1]):
            return -np.inf
        
        return log_prior
    
    def log_posterior(self, theta: np.ndarray) -> float:
        """Compute log-posterior probability"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.likelihood.log_likelihood(theta)
        return lp + ll
    
    def initialize_walkers(self) -> np.ndarray:
        """Initialize walker positions"""
        # Start walkers in small ball around maximum likelihood estimate
        initial_guess = self._find_max_likelihood()
        
        # Add small random perturbations
        pos = []
        for i in range(self.config.nwalkers):
            walker_pos = initial_guess + 1e-4 * np.random.randn(self.ndim)
            pos.append(walker_pos)
        
        return np.array(pos)
    
    def _find_max_likelihood(self) -> np.ndarray:
        """Find maximum likelihood estimate as starting point"""
        # Initial guess (roughly ΛCDM values with chronodynamic additions)
        initial = np.array([0.315, 0.685, 67.4, 1.0, 1.0, -0.55, 1.0])
        
        # Minimize negative log-likelihood
        def neg_log_posterior(theta):
            return -self.log_posterior(theta)
        
        try:
            result = optimize.minimize(
                neg_log_posterior,
                initial,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
            if result.success:
                logger.info(f"Found ML estimate: {result.x}")
                return result.x
            else:
                logger.warning("ML optimization failed, using initial guess")
                return initial
        
        except Exception as e:
            logger.warning(f"ML optimization error: {e}")
            return initial
    
    def run_mcmc(self, save_chain: bool = True, filename: str = None) -> Dict:
        """
        Run MCMC sampling.
        
        Args:
            save_chain: Whether to save chain to file
            filename: Output filename for chain
            
        Returns:
            Dictionary with sampling results
        """
        logger.info("Starting MCMC sampling")
        
        # Initialize sampler
        self.sampler = emcee.EnsembleSampler(
            self.config.nwalkers,
            self.ndim,
            self.log_posterior,
            moves=emcee.moves.StretchMove() if self.config.moves is None else self.config.moves
        )
        
        # Initialize walkers
        pos = self.initialize_walkers()
        
        # Burn-in phase
        logger.info(f"Running burn-in for {self.config.nburn} steps")
        pos, _, _ = self.sampler.run_mcmc(
            pos, 
            self.config.nburn, 
            progress=self.config.progress
        )
        self.sampler.reset()
        
        # Production run
        logger.info(f"Running production for {self.config.nsteps} steps")
        self.sampler.run_mcmc(
            pos, 
            self.config.nsteps, 
            progress=self.config.progress,
            thin=self.config.thin
        )
        
        # Extract results
        self.chain = self.sampler.get_chain()
        self.log_prob = self.sampler.get_log_prob()
        
        # Compute statistics
        results = self._compute_statistics()
        
        # Save chain if requested
        if save_chain:
            output_file = filename or 'chronodynamic_mcmc_chain.h5'
            self._save_chain(output_file)
        
        logger.info("MCMC sampling completed")
        return results
    
    def _compute_statistics(self) -> Dict:
        """Compute summary statistics from MCMC chain"""
        # Flatten chain
        flat_chain = self.chain.reshape(-1, self.ndim)
        flat_log_prob = self.log_prob.flatten()
        
        # Parameter estimates (median and credible intervals)
        percentiles = [16, 50, 84]  # 1σ credible intervals
        param_stats = {}
        
        for i, param_name in enumerate(self.param_names):
            values = flat_chain[:, i]
            p16, p50, p84 = np.percentile(values, percentiles)
            
            param_stats[param_name] = {
                'median': p50,
                'mean': np.mean(values),
                'std': np.std(values),
                'lower_1sigma': p50 - p16,
                'upper_1sigma': p84 - p50,
                'credible_interval_68': [p16, p84]
            }
        
        # Convergence diagnostics
        convergence = self._compute_convergence_diagnostics()
        
        # Model evidence (thermodynamic integration approximation)
        log_evidence = self._estimate_log_evidence()
        
        return {
            'parameter_stats': param_stats,
            'convergence': convergence,
            'log_evidence': log_evidence,
            'acceptance_fraction': np.mean(self.sampler.acceptance_fraction),
            'autocorr_time': self._safe_autocorr_time(),
            'effective_sample_size': self._compute_effective_sample_size(flat_chain)
        }
    
    def _compute_convergence_diagnostics(self) -> Dict:
        """Compute Gelman-Rubin diagnostic and other convergence tests"""
        if self.chain.shape[0] < 100:  # Need sufficient samples
            return {'gelman_rubin': np.nan, 'converged': False}
        
        # Split chains into first and second half
        n_samples = self.chain.shape[0]
        mid = n_samples // 2
        
        chain1 = self.chain[:mid]
        chain2 = self.chain[mid:]
        
        # Compute Gelman-Rubin statistic for each parameter
        R_hat = []
        for i in range(self.ndim):
            # Within-chain variance
            W = 0.5 * (np.var(chain1[:, :, i], axis=0).mean() + 
                      np.var(chain2[:, :, i], axis=0).mean())
            
            # Between-chain variance
            mean1 = np.mean(chain1[:, :, i], axis=0)
            mean2 = np.mean(chain2[:, :, i], axis=0)
            B = 0.5 * self.config.nwalkers * np.var([mean1.mean(), mean2.mean()])
            
            # Gelman-Rubin statistic
            if W > 0:
                R = np.sqrt((W + B) / W)
                R_hat.append(R)
            else:
                R_hat.append(np.inf)
        
        max_R_hat = np.max(R_hat)
        converged = max_R_hat < 1.1  # Standard convergence criterion
        
        return {
            'gelman_rubin': R_hat,
            'max_gelman_rubin': max_R_hat,
            'converged': converged
        }
    
    def _safe_autocorr_time(self) -> np.ndarray:
        """Compute autocorrelation time with error handling"""
        try:
            return self.sampler.get_autocorr_time(quiet=True)
        except Exception:
            return np.full(self.ndim, np.nan)
    
    def _compute_effective_sample_size(self, flat_chain: np.ndarray) -> np.ndarray:
        """Compute effective sample size for each parameter"""
        n_samples = len(flat_chain)
        autocorr_time = self._safe_autocorr_time()
        
        eff_samples = []
        for i, tau in enumerate(autocorr_time):
            if np.isfinite(tau) and tau > 0:
                eff_n = n_samples / (2 * tau)
            else:
                eff_n = n_samples  # Conservative estimate
            eff_samples.append(eff_n)
        
        return np.array(eff_samples)
    
    def _estimate_log_evidence(self) -> float:
        """Estimate log-evidence using thermodynamic integration"""
        # Simplified evidence estimation
        # Full implementation would use nested sampling or thermodynamic integration
        flat_log_prob = self.log_prob.flatten()
        max_log_prob = np.max(flat_log_prob)
        
        # Rough approximation: log(Z) ≈ max(log(L)) + log(prior_volume)
        prior_volume = 1.0  # Compute actual prior volume
        for param_name in self.param_names:
            bounds = getattr(self.priors, param_name)
            prior_volume *= (bounds[1] - bounds[0])
        
        log_evidence = max_log_prob + np.log(prior_volume)
        return log_evidence
    
    def _save_chain(self, filename: str):
        """Save MCMC chain to HDF5 file"""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('chain', data=self.chain)
            f.create_dataset('log_prob', data=self.log_prob)
            f.create_dataset('acceptance_fraction', data=self.sampler.acceptance_fraction)
            
            # Save metadata
            f.attrs['nwalkers'] = self.config.nwalkers
            f.attrs['nsteps'] = self.config.nsteps
            f.attrs['ndim'] = self.ndim
            f.attrs['param_names'] = [name.encode() for name in self.param_names]
        
        logger.info(f"Chain saved to {filename}")
    
    def plot_corner(self, save_fig: bool = True, filename: str = None) -> plt.Figure:
        """Create corner plot of posterior distributions"""
        flat_chain = self.chain.reshape(-1, self.ndim)
        
        fig = corner.corner(
            flat_chain,
            labels=self.param_names,
            truths=None,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={'fontsize': 12}
        )
        
        if save_fig:
            output_file = filename or 'chronodynamic_corner_plot.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Corner plot saved to {output_file}")
        
        return fig
    
    def plot_chains(self, save_fig: bool = True, filename: str = None) -> plt.Figure:
        """Plot MCMC chains for convergence assessment"""
        fig, axes = plt.subplots(self.ndim, figsize=(12, 2*self.ndim))
        
        for i in range(self.ndim):
            ax = axes[i] if self.ndim > 1 else axes
            
            # Plot all walker chains
            for walker in range(self.config.nwalkers):
                ax.plot(self.chain[:, walker, i], alpha=0.3, color='blue', linewidth=0.5)
            
            ax.set_ylabel(self.param_names[i])
            ax.set_xlabel('Step')
        
        plt.tight_layout()
        
        if save_fig:
            output_file = filename or 'chronodynamic_chains.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Chain plots saved to {output_file}")
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("Testing Chronodynamic MCMC Analysis")
    print("=" * 50)
    
    # Create mock observational data
    mock_data = {
        'cmb': {
            'TT': np.random.normal(0, 1, 100),
            'TT_err': np.ones(100) * 0.1
        },
        'h0_local': {
            'value': 73.0,
            'error': 1.5
        }
    }
    
    # Initialize components
    likelihood = ChronodynamicLikelihood(mock_data)
    priors = ParameterPriors()
    config = MCMCConfig(nwalkers=32, nsteps=1000, nburn=200)
    
    # Test likelihood computation
    test_params = np.array([0.315, 0.685, 67.4, 1.0, 1.0, -0.55, 1.0])
    try:
        log_like = likelihood.log_likelihood(test_params)
        print(f"Test likelihood: {log_like}")
    except Exception as e:
        print(f"Likelihood test failed: {e}")
    
    # Test MCMC initialization
    try:
        mcmc = ChronodynamicMCMC(likelihood, priors, config)
        print(f"MCMC initialized with {mcmc.ndim} parameters")
    except Exception as e:
        print(f"MCMC initialization failed: {e}")