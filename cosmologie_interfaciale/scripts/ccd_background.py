# ccd_background.py
# Intégration du fond FLRW pour CCD interfaciale covariante minimale.

from typing import Dict, Tuple, List
import csv, os, math

from ccd_models import A_of_T, A_T, V_of_T

# Unités naturelles: 8πG = 1 ⇒ 3 H^2 = ρ_m + ρ_T
# ρ_m(a) = ρ_m0 / a^3

def rho_T(A: float, Tdot: float, V: float) -> float:
    return 0.5 * A * (Tdot**2) + V

def H_of(a: float, rho_m0: float, A: float, Tdot: float, V: float) -> float:
    rho_m = rho_m0 / (a**3)
    return math.sqrt(max( (rho_m + rho_T(A, Tdot, V)) / 3.0, 0.0 ))

def background_rhs(t: float, y: List[float], pars: Dict) -> List[float]:
    # y = [a, T, Tdot]
    a, T, Tdot = y
    Apars = pars["A_params"]
    Vpars = pars["V_params"]
    rho_m0 = float(pars["rho_m0"])

    A = A_of_T(T, Apars)
    V = V_of_T(T, Vpars)
    H = H_of(a, rho_m0, A, Tdot, V)

    # Équation maîtresse : A (Tddot + 3H Tdot) + 0.5 A_T Tdot^2 + V_T = 0
    # ⇒ Tddot = -(3H Tdot) - (0.5 A_T/A) Tdot^2 - V_T/A
    AT = 0.0
    try:
        from ccd_models import A_T, V_T
        AT = A_T(T, Apars)
        VT = V_T(T, Vpars)
    except Exception:
        VT = 0.0

    Tddot = -(3.0*H*Tdot) - 0.5*(AT/max(A,1e-30))*(Tdot**2) - (VT/max(A,1e-30))

    adot = H * a
    return [adot, Tdot, Tddot]

def rk4_step(f, t, y, dt, pars):
    k1 = f(t, y, pars)
    y2 = [yi + 0.5*dt*k1i for yi, k1i in zip(y, k1)]
    k2 = f(t + 0.5*dt, y2, pars)
    y3 = [yi + 0.5*dt*k2i for yi, k2i in zip(y, k2)]
    k3 = f(t + 0.5*dt, y3, pars)
    y4 = [yi + dt*k3i for yi, k3i in zip(y, k3)]
    k4 = f(t + dt, y4, pars)
    return [yi + (dt/6.0)*(k1i + 2*k2i + 2*k3i + k4i) for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)]

def integrate_background(pars: Dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    a0 = float(pars.get("a0", 1.0))
    T0 = float(pars.get("T0", 0.0))
    Tdot0 = float(pars.get("Tdot0", 0.0))
    t0 = float(pars.get("t_start", 0.0))
    t1 = float(pars.get("t_end", 100.0))
    dt = float(pars.get("dt", 0.05))

    y = [a0, T0, Tdot0]
    t = t0

    with open(os.path.join(outdir, "background.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "a", "H", "T", "Tdot", "rho_m", "rho_T"])
        while t <= t1:
            A = A_of_T(y[1], pars["A_params"])
            V = V_of_T(y[1], pars["V_params"])
            H = H_of(y[0], float(pars["rho_m0"]), A, y[2], V)
            rho_m = float(pars["rho_m0"])/(y[0]**3)
            rho_t = 0.5*A*(y[2]**2) + V
            w.writerow([t, y[0], H, y[1], y[2], rho_m, rho_t])
            y = rk4_step(background_rhs, t, y, dt, pars)
            t += dt

if __name__ == "__main__":
    import json
    import os
    here = os.path.dirname(__file__) or "."
    with open(os.path.join(here, "params_example.json"), "r") as fp:
        pars = json.load(fp)
    integrate_background(pars, outdir=os.path.join(here, "outputs"))
