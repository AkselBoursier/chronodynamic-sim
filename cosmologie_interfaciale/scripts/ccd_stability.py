# ccd_stability.py
# Calcul de c_s^2, m_eff^2 et vérifications de stabilité à partir du fond.

import csv, os, math
from typing import Dict

from ccd_models import A_of_T, A_T, A_TT, V_T, V_TT

def compute_stability_row(a, H, T, Tdot, pars: Dict):
    A = A_of_T(T, pars["A_params"])
    AT = A_T(T, pars["A_params"])
    ATT = A_TT(T, pars["A_params"])
    VT = V_T(T, pars["V_params"])
    VTT = V_TT(T, pars["V_params"])

    # Équation maîtresse (pour Tddot)
    # A (Tddot + 3 H Tdot) + 0.5 A_T Tdot^2 + V_T = 0
    Tddot = -(3.0*H*Tdot) - 0.5*(AT/max(A,1e-30))*(Tdot**2) - (VT/max(A,1e-30))

    X0 = -(Tdot**2)
    cs2 = 1.0  # version minimale
    meff2 = ( VTT - 0.5*ATT*X0 + AT*(Tddot + 3.0*H*Tdot) )/max(A,1e-30)

    no_ghost = (A > 0.0)
    grad_stable = (cs2 > 0.0)
    tachyon_free = (meff2 >= 0.0)  # critère strict; relâcher si besoin

    return A, cs2, meff2, no_ghost, grad_stable, tachyon_free

def run_stability(background_csv: str, pars: Dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "stability.csv")
    with open(background_csv, "r") as f, open(outpath, "w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.writer(g)
        w.writerow(["t","A","cs2","m_eff2","no_ghost","grad_stable","tachyon_free"])
        for row in r:
            t = float(row["t"]); a = float(row["a"]); H = float(row["H"])
            T = float(row["T"]); Tdot = float(row["Tdot"])
            A, cs2, meff2, ng, gs, tf = compute_stability_row(a,H,T,Tdot,pars)
            w.writerow([t, A, cs2, meff2, int(ng), int(gs), int(tf)])

if __name__ == "__main__":
    import json
    here = os.path.dirname(__file__) or "."
    with open(os.path.join(here, "params_example.json"), "r") as fp:
        pars = json.load(fp)
    bg_csv = os.path.join(here, "outputs", "background.csv")
    run_stability(bg_csv, pars, outdir=os.path.join(here, "outputs"))
