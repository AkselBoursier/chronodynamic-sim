#!/usr/bin/env python3
"""
run_observables.py
Compute ΔH/H(z) and ΔD_L/D_L(z) vs ΛCDM for the CCD interfacial model.
"""

import os, json, argparse, csv
import numpy as np   # <-- bien ici

# ---------- Model functions ----------
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

# ---------- Background ----------
def rho_T(A, Tdot, V):
    return 0.5*A*(Tdot**2) + V

def H_of(a, rho_m0, A, Tdot, V):
    return np.sqrt(max((rho_m0/(a**3) + rho_T(A,Tdot,V))/3.0, 0.0))

def rhs(t, y, pars):
    a, T, Tdot = y
    Apars, Vpars = pars["A_params"], pars["V_params"]
    A = A_of_T(T, Apars)
    V = V_of_T(T, Vpars)
    H = H_of(a, float(pars["rho_m0"]), A, Tdot, V)
    AT = A_T(T, Apars)
    VT = V_T(T, Vpars)
    Tddot = -(3.0*H*Tdot) - 0.5*(AT/max(A,1e-30))*(Tdot**2) - (VT/max(A,1e-30))
    adot = H * a
    return np.array([adot, Tdot, Tddot], dtype=float)

def rk4_step(t, y, dt, pars):
    k1 = rhs(t, y, pars)
    k2 = rhs(t + 0.5*dt, y + 0.5*dt*k1, pars)
    k3 = rhs(t + 0.5*dt, y + 0.5*dt*k2, pars)
    k4 = rhs(t + dt, y + dt*k3, pars)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="params_example.json")
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--nz", type=int, default=800)
    args = ap.parse_args()

    here = os.path.dirname(__file__) or "."
    with open(os.path.join(here, args.params), "r") as fp:
        pars = json.load(fp)

    a0, T0, Tdot0 = float(pars["a0"]), float(pars["T0"]), float(pars["Tdot0"])
    rho_m0 = float(pars["rho_m0"])
    dt = -abs(float(pars.get("dt", 0.05)))
    a_min = 1.0/(1.0 + args.zmax)

    t, y = 0.0, np.array([a0, T0, Tdot0], dtype=float)
    rows = []
    A = A_of_T(T0, pars["A_params"]); V = V_of_T(T0, pars["V_params"])
    H = H_of(a0, rho_m0, A, Tdot0, V)
    rows.append((t, a0, H, T0, Tdot0, rho_T(A,Tdot0,V)))

    while y[0] > a_min:
        y = rk4_step(t, y, dt, pars); t += dt
        A = A_of_T(y[1], pars["A_params"]); V = V_of_T(y[1], pars["V_params"])
        H = H_of(y[0], rho_m0, A, y[2], V)
        rows.append((t, y[0], H, y[1], y[2], rho_T(A,y[2],V)))

    bg = np.array(rows)
    a, Hm = bg[:,1], bg[:,2]
    rho_T_today = bg[0,5]
    z = 1.0/a - 1.0
    order = np.argsort(z)
    z_sorted, Hm_sorted = z[order], Hm[order]

    def H_LCDM(a): return np.sqrt((rho_m0/(a**3) + rho_T_today)/3.0)
    Hr_sorted = H_LCDM(a[order])

    z_grid = np.linspace(z_sorted.min(), z_sorted.max(), args.nz)
    Hm_grid = np.interp(z_grid, z_sorted, Hm_sorted)
    Hr_grid = np.interp(z_grid, z_sorted, Hr_sorted)

    dH_frac = (Hm_grid - Hr_grid)/Hr_grid

    def cum_int(zg, Hg):
        invH = 1.0/np.maximum(Hg, 1e-30)
        return np.concatenate([[0.0], np.cumsum(0.5*(invH[1:]+invH[:-1])*np.diff(zg))])

    chi_m, chi_r = cum_int(z_grid,Hm_grid), cum_int(z_grid,Hr_grid)
    DL_m, DL_r = (1+z_grid)*chi_m, (1+z_grid)*chi_r
    dDL_frac = (DL_m - DL_r)/DL_r

    outdir = os.path.join(here,"outputs"); os.makedirs(outdir,exist_ok=True)
    with open(os.path.join(outdir,"DeltaH.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["z","H_model","H_LCDM","DeltaH/H"])
        for zi,hm,hr,d in zip(z_grid,Hm_grid,Hr_grid,dH_frac):
            w.writerow([zi,hm,hr,d])
    with open(os.path.join(outdir,"DeltaDL.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["z","DL_model","DL_LCDM","DeltaDL/DL"])
        for zi,dlm,dlr,d in zip(z_grid,DL_m,DL_r,dDL_frac):
            w.writerow([zi,dlm,dlr,d])

    print("Wrote outputs to", outdir)

if __name__=="__main__":
    main()
