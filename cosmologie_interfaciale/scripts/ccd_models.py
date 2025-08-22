# ccd_models.py
# Définitions de A(T), V(T) et dérivées. Version minimale : A(T)=const, V(T)=Lambda+0.5 m2 T^2

from typing import Dict, Tuple

def A_of_T(T: float, pars: Dict) -> float:
    t = pars.get("type", "constant")
    if t == "constant":
        return float(pars.get("A0", 1.0))
    elif t == "exp":
        A0 = float(pars.get("A0", 1.0))
        alpha = float(pars.get("alpha", 0.0))
        return A0 * (2.718281828459045 ** (alpha * T))
    else:
        return float(pars.get("A0", 1.0))

def A_T(T: float, pars: Dict) -> float:
    t = pars.get("type", "constant")
    if t == "constant":
        return 0.0
    elif t == "exp":
        A0 = float(pars.get("A0", 1.0))
        alpha = float(pars.get("alpha", 0.0))
        return alpha * A0 * (2.718281828459045 ** (alpha * T))
    else:
        return 0.0

def A_TT(T: float, pars: Dict) -> float:
    t = pars.get("type", "constant")
    if t == "constant":
        return 0.0
    elif t == "exp":
        A0 = float(pars.get("A0", 1.0))
        alpha = float(pars.get("alpha", 0.0))
        return (alpha**2) * A0 * (2.718281828459045 ** (alpha * T))
    else:
        return 0.0

def V_of_T(T: float, pars: Dict) -> float:
    t = pars.get("type", "quadratic")
    if t == "quadratic":
        Lam = float(pars.get("Lambda", 0.65))
        m2 = float(pars.get("m2", 0.0))
        return Lam + 0.5 * m2 * (T**2)
    elif t == "constant":
        return float(pars.get("Lambda", 0.65))
    else:
        # défaut : quadratic
        Lam = float(pars.get("Lambda", 0.65))
        m2 = float(pars.get("m2", 0.0))
        return Lam + 0.5 * m2 * (T**2)

def V_T(T: float, pars: Dict) -> float:
    t = pars.get("type", "quadratic")
    if t == "quadratic":
        m2 = float(pars.get("m2", 0.0))
        return m2 * T
    elif t == "constant":
        return 0.0
    else:
        m2 = float(pars.get("m2", 0.0))
        return m2 * T

def V_TT(T: float, pars: Dict) -> float:
    t = pars.get("type", "quadratic")
    if t == "quadratic":
        return float(pars.get("m2", 0.0))
    elif t == "constant":
        return 0.0
    else:
        return float(pars.get("m2", 0.0))
