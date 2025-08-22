# Stabilité linéaire — CCD interfaciale (autour de FLRW)

Objectif: donner les conditions analytiques directement testables numériquement pour le modèle covariant minimal
S = ∫ d⁴x √(−g) [ (1/16πG) R + A(T)·X − 2 V(T) ] + S_matière, avec X ≡ g^{μν} ∂_μ T ∂_ν T.

## 1) Arrière-plan FLRW
- Métrique: ds² = −dt² + a²(t) d→x².
- Champ homogène: T = T₀(t), X₀ = −Ṫ₀².
- Densité et pression: ρ_T = ½ A(T₀) Ṫ₀² + V(T₀),  p_T = ½ A(T₀) Ṫ₀² − V(T₀).
- Équations de Friedmann: 3H² = 8πG (ρ_m + ρ_T),  −2Ḣ = 8πG (ρ_m + p_m + ρ_T + p_T).
- Équation de fond ("maîtresse"): A(T₀)(T̈₀ + 3H Ṫ₀) + ½ A_T(T₀) Ṫ₀² + V_T(T₀) = 0.

## 2) Perturbation scalaire du champ (découplage métrique sub-horizon)
Poser T = T₀(t) + δT(t,→x), travailler à premier ordre et négliger les perturbations métriques pour k ≫ aH.

Équation linéarisée au premier ordre:
δT̈ + (3H + Γ) δṪ + [ c_s² (k²/a²) + m_eff² ] δT = 0,
avec
- Γ ≡ (A_T/A) Ṫ₀  (terme de friction additionnelle),
- c_s² = 1  (car K_X = A(T), K_XX = 0),
- m_eff² ≡ [ V_TT − ½ A_TT X₀ + A_T ( T̈₀ + 3H Ṫ₀ ) ] / A,  où X₀ = −Ṫ₀².

Commentaires:
- c_s² = 1 élimine les instabilités de gradient.
- Le terme Γ ≥ 0 si A_T Ṫ₀ ≥ 0, ce qui amortit les perturbations.

## 3) Action quadratique (forme utile pour code)
À k ≫ aH, l’action au second ordre en δT est, à un facteur global près:
S² ≈ ∫ dt d³x a³ [ ½ A(T₀) (δṪ)² − ½ A(T₀) (∇δT)²/a² − ½ A(T₀) m_eff² δT² ].

Conditions de santé du secteur scalaire:
1) **No-ghost**: A(T₀) > 0.
2) **Gradient stable**: c_s² = 1 > 0 (automatique ici).
3) **Tachyon évité**: m_eff² ≥ 0 (ou, plus faible: m_eff² ≳ −O(H²)).

## 4) Variante canonique (facultative)
Redéfinir φ par dφ/dT = √A(T). Alors X_φ = A X et (dans la limite k ≫ aH):
δφ ≡ √A · δT,  c_s² = 1,  m_φ² = (1/A)[ V_TT − ½ (A_T/A) V_T ]  (approx. quasi-Minkowski).
Cette base est utile pour des solveurs standard de champ canonique.

## 5) Procédure de test numérique (checklist)
Entrées: {A(T), V(T)} et solution de fond {a(t), T₀(t)}.
Étapes:
1. Intégrer l’équation de fond pour T₀(t) et H(t).
2. Évaluer A(T₀), A_T, A_TT, V_T, V_TT, Ṫ₀, T̈₀.
3. Calculer Γ, c_s², m_eff².
4. Vérifier: A>0, c_s²>0, m_eff² ≥ 0 (ou ≥ −η H², η≲O(1)).
5. (Option) Évoluer δT_k avec l’ODE ci-dessus pour un faisceau de k comobiles.

## 6) Remarques observationnelles rapides
- c_GW = c (pas de couplage non minimal): contraintes LIGO/Virgo respectées.
- Secteur scalaire standard: pas de modification de propagation des ondes gravitationnelles.
- Les effets cosmo viennent de l’énergie/pression de fond (w_T) et de la dynamique de m_eff².

Ce document fournit toutes les expressions à coder pour un test de stabilité en fond FLRW du modèle interfacial minimal. 

