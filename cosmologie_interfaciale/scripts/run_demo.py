# run_demo.py
# Exécution bout en bout : fond → stabilité → perturbations

import os, json

from ccd_background import integrate_background
from ccd_stability import run_stability
from ccd_perturb import run_perturbations

def main():
    here = os.path.dirname(__file__) or "."
    with open(os.path.join(here, "params_dynamic2.json"), "r") as fp:
        pars = json.load(fp)

    outdir = os.path.join(here, "outputs")
    integrate_background(pars, outdir=outdir)
    run_stability(os.path.join(outdir, "background.csv"), pars, outdir=outdir)
    run_perturbations(os.path.join(outdir, "background.csv"), pars, pars.get("k_modes",[0.1]), outdir=outdir)

    print("Terminé. Fichiers écrits dans:", outdir)

if __name__ == "__main__":
    main()
