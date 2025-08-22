# ccd_perturb.py
# Intégration de l’ODE des perturbations deltaT_k(t) sur fond CCD minimal.

import csv, os, math
from typing import Dict, Tuple, List
from ccd_models import A_of_T, A_T, A_TT, V_T, V_TT

def rk4_step(func, t, y, dt, args):
    k1 = func(t, y, *args)
    y2 = [yi + 0.5*dt*k1i for yi, k1i in zip(y, k1)]
    k2 = func(t+0.5*dt, y2, *args)
    y3 = [yi + 0.5*dt*k2i for yi, k2i in zip(y, k2)]
    k3 = func(t+0.5*dt, y3, *args)
    y4 = [yi + dt*k3i for yi, k3i in zip(y, k3)]
    k4 = func(t+dt, y4, *args)
    return [yi + (dt/6.0)*(k1i + 2*k2i + 2*k3i + k4i) for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)]

def load_background(bg_csv: str):
    rows = []
    with open(bg_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "t": float(row["t"]),
                "a": float(row["a"]),
                "H": float(row["H"]),
                "T": float(row["T"]),
                "Tdot": float(row["Tdot"])
            })
    return rows

def meff2_at(T, Tdot, H, pars: Dict):
    A = A_of_T(T, pars["A_params"])
    AT = A_T(T, pars["A_params"])
    ATT = A_TT(T, pars["A_params"])
    VT = V_T(T, pars["V_params"])
    VTT = V_TT(T, pars["V_params"])
    Tddot = -(3.0*H*Tdot) - 0.5*(AT/max(A,1e-30))*(Tdot**2) - (VT/max(A,1e-30))
    X0 = -(Tdot**2)
    return ( VTT - 0.5*ATT*X0 + AT*(Tddot + 3.0*H*Tdot) )/max(A,1e-30)

def ode_deltaT(t, y, a, H, T, Tdot, pars: Dict, k: float):
    # y = [deltaT, deltaTdot]
    deltaT, deltaTdot = y
    A = A_of_T(T, pars["A_params"])
    AT = A_T(T, pars["A_params"])
    cs2 = 1.0
    gamma = (AT/max(A,1e-30)) * Tdot
    m2 = meff2_at(T, Tdot, H, pars)
    d1 = deltaTdot
    d2 = -(3.0*H + gamma)*deltaTdot - ( (k**2)/(a**2) + m2 )*deltaT
    return [d1, d2]

def integrate_mode(bg_rows, pars: Dict, k: float, outpath: str):
    # Conditions initiales simples : deltaT ~ 1e-6, deltaTdot ~ 0
    deltaT = 1e-6
    deltaTdot = 0.0
    with open(outpath, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["t","deltaT","deltaTdot"])
        for i in range(len(bg_rows)-1):
            r = bg_rows[i]
            rp = bg_rows[i+1]
            t, tp = r["t"], rp["t"]
            dt = tp - t
            y = [deltaT, deltaTdot]
            y = rk4_step(ode_deltaT, t, y, dt, (r["a"], r["H"], r["T"], r["Tdot"], pars, k))
            deltaT, deltaTdot = y
            w.writerow([tp, deltaT, deltaTdot])

def run_perturbations(bg_csv: str, pars: Dict, ks, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    rows = load_background(bg_csv)
    for k in ks:
        outpath = os.path.join(outdir, f"perturb_k{k}.csv")
        integrate_mode(rows, pars, k, outpath)

if __name__ == "__main__":
    import json, os
    here = os.path.dirname(__file__) or "."
    with open(os.path.join(here, "params_example.json"), "r") as fp:
        pars = json.load(fp)
    bg_csv = os.path.join(here, "outputs", "background.csv")
    ks = pars.get("k_modes", [0.1])
    run_perturbations(bg_csv, pars, ks, outdir=os.path.join(here, "outputs"))
