#!/usr/bin/env python3
"""
fit_like.py - quick chi-square against simple SNe and BAO files.

Datasets expected:
- SNe CSV with columns: z, mu, sigma_mu
  * We compare to mu_th(z) = 5 log10(D_L_model(z)) + M, and marginalize analytically over M.
- BAO CSV with columns: z, H_obs, sigma_H
  * We compare to H_obs vs s_H * H_model(z), and marginalize analytically over s_H.

Model:
- Same covariant minimal CCD used elsewhere.
- Parameters read from JSON (same format as params_example.json / params_dynamic*.json).
- We integrate backward from a=1 to cover the data redshifts.

Outputs:
- Prints chi2_SNe, chi2_BAO, chi2_tot (after analytic marginalization of M and s_H).
- Writes a small report in outputs/fit_report.txt

Note: Units here are arbitrary (c=1, 8πG=1). Nuisance parameters absorb overall scales.
"""

import argparse
import numpy as np
import csv
import os, json

try:
    from ccd_models import A_of_T, A_T, V_of_T, V_T
except Exception:
    def A_of_T(T, pars): return float(pars.get("A0", 1.0))
    def A_T(T, pars): return 0.0
    def V_of_T(T, pars):
        Lam = float(pars.get("Lambda", 0.65)); m2 = float(pars.get("m2", 0.0))
        return Lam + 0.5*m2*T*T
    def V_T(T, pars):
        m2 = float(pars.get("m2", 0.0)); return m2*T

def rho_T(A, Tdot, V): return 0.5*A*(Tdot**2) + V

def H_of(a, rho_m0, A, Tdot, V):
    return np.sqrt(max((rho_m0/(a**3) + rho_T(A,Tdot,V))/3.0, 0.0))

def rhs(t, y, pars):
    a, T, Tdot = y
    A = A_of_T(T, pars["A_params"])
    V = V_of_T(T, pars["V_params"])
    H = H_of(a, float(pars["rho_m0"]), A, Tdot, V)
    try:
        AT = A_T(T, pars["A_params"])
    except Exception:
        AT = 0.0
    VT = V_T(T, pars["V_params"])
    Tddot = -(3.0*H*Tdot) - 0.5*(AT/max(A,1e-30))*(Tdot**2) - (VT/max(A,1e-30))
    adot = H*a
    return np.array([adot, Tdot, Tddot], dtype=float)

def rk4_step(t, y, dt, pars):
    k1 = rhs(t, y, pars)
    k2 = rhs(t + 0.5*dt, y + 0.5*dt*k1, pars)
    k3 = rhs(t + 0.5*dt, y + 0.5*dt*k2, pars)
    k4 = rhs(t + dt, y + dt*k3, pars)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_background(pars, zmax, nz=2000):
    a0, T0, Tdot0 = float(pars["a0"]), float(pars["T0"]), float(pars["Tdot0"])
    rho_m0 = float(pars["rho_m0"])
    dt = -abs(float(pars.get("dt", 0.05)))
    a_min = 1.0/(1.0 + max(zmax, 0.0))

    t, y = 0.0, np.array([a0, T0, Tdot0], dtype=float)
    rows = []
    A = A_of_T(y[1], pars["A_params"]); V = V_of_T(y[1], pars["V_params"])
    H = H_of(y[0], rho_m0, A, y[2], V)
    rows.append((t, y[0], H, y[1], y[2], rho_T(A,y[2],V)))
    safety = 0
    while y[0] > a_min and safety < 1000000:
        y = rk4_step(t, y, dt, pars); t += dt
        A = A_of_T(y[1], pars["A_params"]); V = V_of_T(y[1], pars["V_params"])
        H = H_of(y[0], rho_m0, A, y[2], V)
        rows.append((t, y[0], H, y[1], y[2], rho_T(A,y[2],V)))
        safety += 1

    bg = np.array(rows)
    a = bg[:,1]; Hm = bg[:,2]
    z = 1.0/a - 1.0
    order = np.argsort(z)
    z_sorted, H_sorted = z[order], Hm[order]

    z_grid = np.linspace(0.0, z_sorted.max(), nz)
    H_grid = np.interp(z_grid, z_sorted, H_sorted)

    invH = 1.0/np.maximum(H_grid, 1e-30)
    chi = np.concatenate([[0.0], np.cumsum(0.5*(invH[1:]+invH[:-1])*np.diff(z_grid))])
    DL = (1.0+z_grid)*chi
    return z_grid, H_grid, DL

def chi2_SNe(z_data, mu_data, sigma_mu, z_grid, DL_grid):
    mu_th_woM = 5.0*np.log10(np.interp(z_data, z_grid, DL_grid))
    w = 1.0/np.maximum(sigma_mu**2, 1e-30)
    M_hat = np.sum(w*(mu_data - mu_th_woM))/np.sum(w)
    chi2 = np.sum(w*(mu_data - (mu_th_woM + M_hat))**2)
    return chi2, M_hat

def chi2_BAO(z_data, Hobs, sigma_H, z_grid, H_grid):
    H_th = np.interp(z_data, z_grid, H_grid)
    w = 1.0/np.maximum(sigma_H**2, 1e-30)
    s_hat = np.sum(w*Hobs*H_th)/np.sum(w*(H_th**2))
    chi2 = np.sum(w*(Hobs - s_hat*H_th)**2)
    return chi2, s_hat

def main():
    import argparse, os, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="params_example.json")
    ap.add_argument("--sne_csv", type=str, default=None)
    ap.add_argument("--bao_csv", type=str, default=None)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--nz", type=int, default=2000)
    args = ap.parse_args()

    here = os.path.dirname(__file__) or "."
    with open(os.path.join(here, args.params), "r") as fp:
        pars = json.load(fp)

    z_grid, H_grid, DL_grid = integrate_background(pars, args.zmax, args.nz)

    chi2_sne = chi2_bao = np.nan
    M_hat = s_hat = np.nan

    if args.sne_csv:
        S = np.genfromtxt(os.path.join(here, args.sne_csv), delimiter=",", names=True, dtype=None, encoding=None)
        zS, muS, sigS = S["z"], S["mu"], S["sigma_mu"]
        mask = (zS >= z_grid.min()) & (zS <= z_grid.max())
        chi2_sne, M_hat = chi2_SNe(zS[mask], muS[mask], sigS[mask], z_grid, DL_grid)

    if args.bao_csv:
        B = np.genfromtxt(os.path.join(here, args.bao_csv), delimiter=",", names=True, dtype=None, encoding=None)
        zB, HB, sigB = B["z"], B["H_obs"], B["sigma_H"]
        mask = (zB >= z_grid.min()) & (zB <= z_grid.max())
        chi2_bao, s_hat = chi2_BAO(zB[mask], HB[mask], sigB[mask], z_grid, H_grid)

    chi2_tot = 0.0
    parts = []
    if not np.isnan(chi2_sne):
        chi2_tot += chi2_sne; parts.append(f"SNe chi2={chi2_sne:.3f} (M_hat={M_hat:.3f})")
    if not np.isnan(chi2_bao):
        chi2_tot += chi2_bao; parts.append(f"BAO chi2={chi2_bao:.3f} (s_H_hat={s_hat:.5f})")

    outdir = os.path.join(here, "outputs"); os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "fit_report.txt"), "w") as f:
        f.write("fit_like.py result\n")
        f.write(f"params: {args.params}\n")
        f.write(f"z range: {z_grid.min():.4f}-{z_grid.max():.4f}\n")
        if parts:
            f.write(" + ".join(parts) + "\n")
            f.write(f"TOTAL chi2={chi2_tot:.3f}\n")
        else:
            f.write("No datasets provided. Nothing to fit.\n")

    print("Done.")
    if parts:
        print(" | ".join(parts))
        print(f"TOTAL chi2 = {chi2_tot:.3f}")
    else:
        print("No datasets provided. Wrote model grids only.")

if __name__ == "__main__":
    main()
