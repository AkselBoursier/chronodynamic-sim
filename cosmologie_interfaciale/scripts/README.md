# CCD Interfaciale — Scripts de référence

## Contenu
- `params_example.json` : paramètres du modèle (A(T), V(T), H0, Omega_m0, etc.).
- `ccd_models.py` : définitions de A(T), V(T) et de leurs dérivées.
- `ccd_background.py` : intégration du fond FLRW (a(t), T(t), Tdot(t)).
- `ccd_stability.py` : calcul de c_s^2, m_eff^2 et vérifications no-ghost.
- `ccd_perturb.py` : intégration de l’ODE des perturbations δT_k(t).
- `run_demo.py` : exemple d’exécution de bout en bout.

## Hypothèses
- Unités naturelles : c = 1, 8πG = 1 ⇒ 3H^2 = ρ_m + ρ_T.
- Fond matière seule (ρ_m ∝ a^−3).
- Version covariante minimale : L = √−g [ R/(16πG) + A(T) X − 2 V(T) ].

## Usage rapide
1. Éditez `params_example.json` si nécessaire.
2. Exécutez : `python run_demo.py`
3. Vous verrez un résumé de stabilité et des sorties `.csv` simples dans `./outputs/`.

## Sorties
- `outputs/background.csv` : t, a, H, T, Tdot.
- `outputs/stability.csv` : t, A, cs2, m_eff2.
- `outputs/perturb_kEllipsis.csv` : t, deltaT, deltaTdot pour chaque k.

## Personnalisation
- Modifiez `ccd_models.py` pour changer A(T), V(T).
- Pour tester d’autres scénarios, éditez `params_example.json` (plage temporelle, conditions initiales, liste des modes k).

## Références internes (formalisme)
- Écriture covariante minimale, équations de fond et de stabilité conformément au document “Équations interfaciales — version covariante minimale”.
