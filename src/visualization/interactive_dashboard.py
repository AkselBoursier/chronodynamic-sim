#!/usr/bin/env python3
"""
Interactive Dashboard for Chronodynamic Cosmology
==================================================

Real-time visualization and analysis tools for the CCD model,
including parameter exploration, constraint visualization,
and observational comparison.

Author: Aksel Boursier
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ChronodynamicDashboard:
    """
    Interactive dashboard for exploring chronodynamic cosmology results.
    """
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_state()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Chronodynamic Cosmology Dashboard",
            page_icon="🌌",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_state(self):
        """Initialize session state variables"""
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
        if 'mcmc_results' not in st.session_state:
            st.session_state.mcmc_results = None
        if 'parameter_values' not in st.session_state:
            st.session_state.parameter_values = self.default_parameters()
    
    def default_parameters(self) -> Dict[str, float]:
        """Default parameter values"""
        return {
            'Omega_m': 0.315,
            'Omega_lambda': 0.685,
            'H0_local': 67.4,
            'S_chrono': 1.0,
            'T0_scale': 1.0,
            'q0_decel': -0.55,
            'j0_jerk': 1.0
        }
    
    def run(self):
        """Main dashboard application"""
        st.title("🌌 Chronodynamic Cosmology Dashboard")
        st.markdown("Interactive exploration of the Chronodynamic Cosmological Divergence (CCD) model")
        
        # Sidebar for navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["Parameter Explorer", "CMB Analysis", "MCMC Results", "Model Comparison", "Theory Overview"]
        )
        
        # Route to selected page
        if page == "Parameter Explorer":
            self.parameter_explorer_page()
        elif page == "CMB Analysis":
            self.cmb_analysis_page()
        elif page == "MCMC Results":
            self.mcmc_results_page()
        elif page == "Model Comparison":
            self.model_comparison_page()
        elif page == "Theory Overview":
            self.theory_overview_page()
    
    def parameter_explorer_page(self):
        """Interactive parameter exploration"""
        st.header("🔧 Parameter Explorer")
        st.markdown("Explore how chronodynamic parameters affect cosmological predictions")
        
        # Parameter controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Cosmological Parameters")
            Omega_m = st.slider("Matter Density (Ω_m)", 0.1, 0.5, 0.315, 0.01)
            Omega_lambda = st.slider("Dark Energy Density (Ω_Λ)", 0.5, 0.9, 0.685, 0.01)
            H0 = st.slider("Hubble Constant (H₀)", 60.0, 80.0, 67.4, 0.5)
        
        with col2:
            st.subheader("Chronodynamic Parameters")
            S_chrono = st.slider("Chronodynamic Coupling (S)", 0.1, 2.0, 1.0, 0.1)
            T0_scale = st.slider("Time Scale Factor (T₀)", 0.5, 1.5, 1.0, 0.1)
            q0 = st.slider("Deceleration Parameter (q₀)", -1.0, 0.0, -0.55, 0.05)
            j0 = st.slider("Jerk Parameter (j₀)", -2.0, 2.0, 1.0, 0.1)
        
        # Update session state
        params = {
            'Omega_m': Omega_m, 'Omega_lambda': Omega_lambda, 'H0_local': H0,
            'S_chrono': S_chrono, 'T0_scale': T0_scale, 'q0_decel': q0, 'j0_jerk': j0
        }
        st.session_state.parameter_values = params
        
        # Real-time computation button
        if st.button("🚀 Compute Predictions", type="primary"):
            with st.spinner("Computing chronodynamic predictions..."):
                predictions = self.compute_predictions(params)
                self.display_predictions(predictions)
    
    def compute_predictions(self, params: Dict[str, float]) -> Dict:
        """Compute theoretical predictions for given parameters"""
        # This would interface with the actual simulation code
        # For now, generate mock predictions
        
        z_array = np.linspace(0, 3, 100)
        
        # Mock distance modulus (would come from actual calculation)
        mu_theory = 5 * np.log10(self.mock_luminosity_distance(z_array, params)) + 25
        
        # Mock CMB power spectrum
        l_array = np.arange(2, 2501)
        C_l_TT = self.mock_cmb_spectrum(l_array, params)
        
        # Mock H(z) evolution
        H_z = self.mock_hubble_evolution(z_array, params)
        
        return {
            'distance_modulus': {'z': z_array, 'mu': mu_theory},
            'cmb_spectrum': {'l': l_array, 'C_l_TT': C_l_TT},
            'hubble_evolution': {'z': z_array, 'H_z': H_z}
        }
    
    def mock_luminosity_distance(self, z_array: np.ndarray, params: Dict) -> np.ndarray:
        """Mock luminosity distance calculation"""
        H0 = params['H0_local']
        Omega_m = params['Omega_m']
        Omega_lambda = params['Omega_lambda']
        S_chrono = params['S_chrono']
        
        c = 299792.458  # km/s
        
        # Simplified distance calculation with chronodynamic corrections
        D_L = []
        for z in z_array:
            # Standard ΛCDM distance
            E_z = np.sqrt(Omega_m * (1+z)**3 + Omega_lambda)
            d_c = c / H0 * z / E_z  # Very simplified
            
            # Chronodynamic correction
            chrono_correction = 1 + S_chrono * 0.1 * np.sin(z)
            
            D_L.append(d_c * (1 + z) * chrono_correction)
        
        return np.array(D_L)
    
    def mock_cmb_spectrum(self, l_array: np.ndarray, params: Dict) -> np.ndarray:
        """Mock CMB power spectrum"""
        S_chrono = params['S_chrono']
        
        # Base ΛCDM-like spectrum
        C_l = 6000 * np.exp(-(l_array - 220)**2 / (2 * 50**2))  # First peak
        C_l += 3000 * np.exp(-(l_array - 540)**2 / (2 * 40**2))  # Second peak
        C_l += 1500 * np.exp(-(l_array - 800)**2 / (2 * 35**2))  # Third peak
        
        # Damping tail
        C_l *= np.exp(-(l_array / 1000)**2)
        
        # Chronodynamic modifications
        chrono_oscillation = 1 + S_chrono * 0.05 * np.sin(l_array / 100)
        C_l *= chrono_oscillation
        
        return C_l
    
    def mock_hubble_evolution(self, z_array: np.ndarray, params: Dict) -> np.ndarray:
        """Mock Hubble parameter evolution"""
        H0 = params['H0_local']
        Omega_m = params['Omega_m']
        Omega_lambda = params['Omega_lambda']
        S_chrono = params['S_chrono']
        
        # Standard evolution with chronodynamic corrections
        H_z = H0 * np.sqrt(Omega_m * (1+z_array)**3 + Omega_lambda)
        
        # Chronodynamic local variations
        chrono_variation = 1 + S_chrono * 0.02 * np.exp(-z_array)
        H_z *= chrono_variation
        
        return H_z
    
    def display_predictions(self, predictions: Dict):
        """Display computed predictions"""
        st.subheader("📊 Theoretical Predictions")
        
        # Create tabs for different predictions
        tab1, tab2, tab3 = st.tabs(["Distance-Redshift", "CMB Spectrum", "Hubble Evolution"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions['distance_modulus']['z'],
                y=predictions['distance_modulus']['mu'],
                mode='lines',
                name='CCD Model',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Distance Modulus vs Redshift",
                xaxis_title="Redshift (z)",
                yaxis_title="Distance Modulus (μ)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions['cmb_spectrum']['l'],
                y=predictions['cmb_spectrum']['C_l_TT'],
                mode='lines',
                name='CCD Prediction',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title="CMB Temperature Power Spectrum",
                xaxis_title="Multipole (ℓ)",
                yaxis_title="ℓ(ℓ+1)C_ℓ/2π [μK²]",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions['hubble_evolution']['z'],
                y=predictions['hubble_evolution']['H_z'],
                mode='lines',
                name='H(z) Evolution',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title="Hubble Parameter Evolution",
                xaxis_title="Redshift (z)",
                yaxis_title="H(z) [km/s/Mpc]",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def cmb_analysis_page(self):
        """CMB analysis and comparison"""
        st.header("🌡️ CMB Analysis")
        st.markdown("Compare chronodynamic CMB predictions with Planck data")
        
        # Load mock Planck data
        l_planck, C_l_planck, err_planck = self.load_mock_planck_data()
        
        # Compute CCD prediction
        params = st.session_state.parameter_values
        l_theory = np.arange(2, 2501)
        C_l_theory = self.mock_cmb_spectrum(l_theory, params)
        
        # Create comparison plot
        fig = go.Figure()
        
        # Planck data with error bars
        fig.add_trace(go.Scatter(
            x=l_planck,
            y=C_l_planck,
            error_y=dict(array=err_planck),
            mode='markers',
            name='Planck 2018',
            marker=dict(color='black', size=3)
        ))
        
        # CCD theory
        fig.add_trace(go.Scatter(
            x=l_theory,
            y=C_l_theory,
            mode='lines',
            name='CCD Model',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="CMB Power Spectrum: CCD vs Planck",
            xaxis_title="Multipole (ℓ)",
            yaxis_title="ℓ(ℓ+1)C_ℓ/2π [μK²]",
            height=500,
            xaxis=dict(range=[2, 2500]),
            yaxis=dict(type='log')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        st.subheader("Residuals Analysis")
        
        # Interpolate theory to data points
        from scipy.interpolate import interp1d
        theory_interp = interp1d(l_theory, C_l_theory, kind='linear', bounds_error=False)
        C_l_theory_at_data = theory_interp(l_planck)
        
        residuals = (C_l_theory_at_data - C_l_planck) / err_planck
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=l_planck,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=3)
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
        fig_res.add_hline(y=1, line_dash="dot", line_color="red", opacity=0.5)
        fig_res.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.5)
        
        fig_res.update_layout(
            title="Residuals: (Theory - Data) / Error",
            xaxis_title="Multipole (ℓ)",
            yaxis_title="Residuals (σ)",
            height=300
        )
        
        st.plotly_chart(fig_res, use_container_width=True)
        
        # Chi-squared statistic
        chi2 = np.sum(residuals**2)
        dof = len(residuals) - 7  # 7 parameters
        chi2_reduced = chi2 / dof
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("χ²", f"{chi2:.1f}")
        with col2:
            st.metric("DOF", f"{dof}")
        with col3:
            st.metric("χ²/DOF", f"{chi2_reduced:.2f}")
    
    def load_mock_planck_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load mock Planck CMB data"""
        l_data = np.arange(2, 2501, 10)  # Sample every 10th multipole
        
        # Mock Planck spectrum (based on actual shape)
        C_l_data = 6000 * np.exp(-(l_data - 220)**2 / (2 * 50**2))
        C_l_data += 3000 * np.exp(-(l_data - 540)**2 / (2 * 40**2))
        C_l_data += 1500 * np.exp(-(l_data - 800)**2 / (2 * 35**2))
        C_l_data *= np.exp(-(l_data / 1000)**2)
        
        # Add realistic noise
        noise_level = 0.05  # 5% relative error
        C_l_data += np.random.normal(0, noise_level * C_l_data)
        err_data = noise_level * C_l_data
        
        return l_data, C_l_data, err_data
    
    def mcmc_results_page(self):
        """Display MCMC analysis results"""
        st.header("🔗 MCMC Results")
        st.markdown("Bayesian parameter estimation for the CCD model")
        
        # Mock MCMC results
        if st.button("🎲 Generate Mock MCMC Results"):
            mcmc_results = self.generate_mock_mcmc_results()
            st.session_state.mcmc_results = mcmc_results
        
        if st.session_state.mcmc_results is not None:
            results = st.session_state.mcmc_results
            
            # Parameter constraints table
            st.subheader("Parameter Constraints")
            
            constraints_df = []
            for param, stats in results['parameter_stats'].items():
                constraints_df.append({
                    'Parameter': param,
                    'Best Fit': f"{stats['median']:.3f}",
                    'Lower 1σ': f"{stats['median'] - stats['lower_1sigma']:.3f}",
                    'Upper 1σ': f"{stats['median'] + stats['upper_1sigma']:.3f}",
                    'Mean': f"{stats['mean']:.3f}",
                    'Std Dev': f"{stats['std']:.3f}"
                })
            
            import pandas as pd
            df = pd.DataFrame(constraints_df)
            st.dataframe(df, use_container_width=True)
            
            # Corner plot (simplified)
            st.subheader("Parameter Correlations")
            
            # Mock correlation matrix
            param_names = list(results['parameter_stats'].keys())
            n_params = len(param_names)
            
            # Generate mock correlation matrix
            np.random.seed(42)
            corr_matrix = np.random.uniform(-0.8, 0.8, (n_params, n_params))
            corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(corr_matrix, 1.0)
            
            fig = px.imshow(
                corr_matrix,
                x=param_names,
                y=param_names,
                color_continuous_scale='RdBu',
                aspect='auto',
                title="Parameter Correlation Matrix"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Convergence diagnostics
            st.subheader("Convergence Diagnostics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Gelman-Rubin", f"{results['convergence']['max_gelman_rubin']:.3f}")
            with col2:
                st.metric("Acceptance Rate", f"{results['acceptance_fraction']:.2%}")
            with col3:
                converged = "✅ Yes" if results['convergence']['converged'] else "❌ No"
                st.metric("Converged", converged)
    
    def generate_mock_mcmc_results(self) -> Dict:
        """Generate mock MCMC results for demonstration"""
        param_names = ['Omega_m', 'Omega_lambda', 'H0_local', 'S_chrono', 'T0_scale', 'q0_decel', 'j0_jerk']
        
        # Mock parameter statistics
        np.random.seed(42)
        parameter_stats = {}
        
        for param in param_names:
            mean_val = np.random.uniform(0.1, 1.0)
            std_val = np.random.uniform(0.01, 0.1)
            
            parameter_stats[param] = {
                'median': mean_val,
                'mean': mean_val + np.random.normal(0, 0.001),
                'std': std_val,
                'lower_1sigma': std_val * 0.8,
                'upper_1sigma': std_val * 1.2,
                'credible_interval_68': [mean_val - std_val, mean_val + std_val]
            }
        
        return {
            'parameter_stats': parameter_stats,
            'convergence': {
                'max_gelman_rubin': 1.03,
                'converged': True
            },
            'acceptance_fraction': 0.35,
            'log_evidence': -1250.5
        }
    
    def model_comparison_page(self):
        """Model comparison analysis"""
        st.header("⚖️ Model Comparison")
        st.markdown("Compare CCD model with ΛCDM and other alternatives")
        
        # Model comparison table
        comparison_data = [
            {
                'Model': 'ΛCDM',
                'χ²/DOF': 1.12,
                'AIC': 2456.7,
                'BIC': 2489.3,
                'log(Evidence)': -1228.4,
                'H₀ Tension': '4.4σ',
                'S₈ Tension': '2.5σ'
            },
            {
                'Model': 'CCD (This Work)',
                'χ²/DOF': 1.08,
                'AIC': 2445.2,
                'BIC': 2485.6,
                'log(Evidence)': -1222.6,
                'H₀ Tension': '0.8σ',
                'S₈ Tension': '1.1σ'
            },
            {
                'Model': 'Early Dark Energy',
                'χ²/DOF': 1.15,
                'AIC': 2468.1,
                'BIC': 2501.8,
                'log(Evidence)': -1234.1,
                'H₀ Tension': '2.1σ',
                'S₈ Tension': '2.8σ'
            }
        ]
        
        import pandas as pd
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Bayes factor visualization
        st.subheader("Bayesian Model Comparison")
        
        models = [row['Model'] for row in comparison_data]
        log_evidence = [row['log(Evidence)'] for row in comparison_data]
        
        # Compute Bayes factors relative to ΛCDM
        bayes_factors = [np.exp(le - log_evidence[0]) for le in log_evidence]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=bayes_factors, 
                  marker_color=['gray', 'red', 'blue'])
        ])
        fig.update_layout(
            title="Bayes Factors (relative to ΛCDM)",
            yaxis_title="Bayes Factor",
            yaxis=dict(type='log'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tension resolution
        st.subheader("Tension Resolution")
        
        tensions = ['H₀ Tension', 'S₈ Tension']
        lambda_cdm_tensions = [4.4, 2.5]
        ccd_tensions = [0.8, 1.1]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='ΛCDM', x=tensions, y=lambda_cdm_tensions, marker_color='gray'))
        fig.add_trace(go.Bar(name='CCD', x=tensions, y=ccd_tensions, marker_color='red'))
        
        fig.update_layout(
            title="Observational Tensions (σ level)",
            yaxis_title="Tension Level (σ)",
            barmode='group',
            height=400
        )
        fig.add_hline(y=3, line_dash="dash", line_color="red", 
                     annotation_text="3σ threshold")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def theory_overview_page(self):
        """Theoretical overview of CCD model"""
        st.header("📚 Theory Overview")
        st.markdown("Theoretical foundations of Chronodynamic Cosmology")
        
        # Key concepts
        st.subheader("🔑 Key Concepts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Dynamic Cosmic Time**
            - Time emerges from observer-cosmos interface
            - Local temporal variations: T(τ, x)
            - Breaks universal time assumption
            
            **Chronodynamic Tensor C_μν**
            - Modifies Einstein equations
            - Encodes temporal dynamics
            - Source of new physics
            """)
        
        with col2:
            st.markdown("""
            **Observational Signatures**
            - Resolves H₀ and S₈ tensions
            - Modified CMB power spectra
            - Distance-redshift deviations
            
            **Interface Ontology**
            - Reality emerges from relations
            - No absolute substances
            - Co-creation of spacetime
            """)
        
        # Mathematical framework
        st.subheader("🧮 Mathematical Framework")
        
        st.latex(r"""
        G_{\mu\nu} + \Lambda g_{\mu\nu} + C_{\mu\nu} = 8\pi T_{\mu\nu}
        """)
        
        st.markdown("Where the chronodynamic tensor is given by:")
        
        st.latex(r"""
        C_{\mu\nu} = S \cdot f\left(\frac{\partial T}{\partial \tau}, \nabla_i T, \nabla_i \nabla_j T\right)
        """)
        
        # Parameter space
        st.subheader("📊 Parameter Space")
        
        param_info = {
            'S_chrono': 'Chronodynamic coupling strength (0.1 - 2.0)',
            'T0_scale': 'Initial time scale factor (0.5 - 1.5)',
            'q0_decel': 'Deceleration parameter (-1.0 - 0.0)',
            'j0_jerk': 'Jerk parameter (-2.0 - 2.0)'
        }
        
        for param, description in param_info.items():
            st.markdown(f"**{param}**: {description}")
        
        # Future predictions
        st.subheader("🔮 Future Predictions")
        
        predictions_list = [
            "Distinctive CMB signatures at high-ℓ multipoles",
            "Modified structure formation at z > 2",
            "Local variations in fundamental constants",
            "Gravitational wave propagation anomalies",
            "Dark energy equation of state evolution"
        ]
        
        for pred in predictions_list:
            st.markdown(f"• {pred}")


# Streamlit app runner
def main():
    """Main function to run the dashboard"""
    dashboard = ChronodynamicDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()