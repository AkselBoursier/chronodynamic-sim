# Équations interfaciales — version covariante minimale

## 1) Variables et scalaires covariants
- Champ‑« horloge » : T(x^μ), scalaire.
- Invariant cinétique : X ≡ g^{μν} ∂_μ T ∂_ν T.
- (Option) « Rythme interfacial » opérationnel : ℛ ≡ √(−X) pour FLRW avec T = T(t).

## 2) Action canonique (covariante)
S = ∫ d^4x √(−g) [ (1/16πG) R + A(T)·X − 2 V(T) ] + S_matière[g_{μν}, ψ].
- A(T) > 0 garantit un signe cinétique correct (pas de fantôme).
- Cette forme remplace toute dépendance non covariante du type f(t) ou h(t)·R^2(t). Les fonctions sont désormais des fonctions scalaires de T.
- Option sûre : pour préserver c_GW = c, on évite les couplages non minimaux susceptibles de modifier la vitesse des ondes gravitationnelles.

## 3) Équations de champ
### 3.1 Einstein
G_{μν} = 8πG [ T^{matière}_{μν} + T^{(T)}_{μν} ],
T^{(T)}_{μν} = A(T) ∂_μ T ∂_ν T − g_{μν} [ ½ A(T) X + V(T) ].

### 3.2 Dynamique scalaire (interface)
∇_μ [ A(T) ∂^μ T ] − ½ A_T(T) X − V_T(T) = 0.

## 4) FLRW homogène (ds² = −dt² + a²(t) d→x², T = T(t))
- X = −Ṫ².
- Énergie et pression effectives du champ interfacial :
ρ_T = ½ A(T) Ṫ² + V(T),   p_T = ½ A(T) Ṫ² − V(T).

### 4.1 Équations de Friedmann modifiées
3H² = 8πG ( ρ_matière + ρ_T ),
−2Ḣ = 8πG ( ρ_matière + p_matière + ρ_T + p_T ).

### 4.2 Équation maîtresse (conservation interfaciale)
ṙho_T + 3H (ρ_T + p_T) = 0  ⇔  A(T) ( T¨ + 3H T˙ ) + ½ A_T(T) T˙² + V_T(T) = 0.
C’est la version covariante de la loi de cohérence rythmique précédente.

## 5) Correspondance avec l’ancienne écriture
Ancienne écriture non covariante √(−g)[ f(t) + h(t) R²(t) ] → **termes effectifs** :
- f(t) ↦ partie potentielle V(T) et composante de pression négative.
- h(t)R²(t) ↦ contributions cinétiques via A(T)·X et via le choix de solution T(t) (le « rythme » est désormais géométrisé par X).

## 6) Propriétés clés
- Covariance générale : aucun t nu, uniquement des scalaires T, X.
- Limite RG : si ∂T → 0 et V → Λ/(8πG), on récupère G_{μν} + Λ g_{μν} = 8πG T^{matière}_{μν}.
- Causalité et hyperbolicité : équation de type onde pour T avec friction cosmique 3H T˙.
- Ondes gravitationnelles : avec couplage minimal, c_GW = c.

## 7) Lecture interfaciale (pont conceptuel)
- T = variable d’interface temporelle.
- A(T) = tension de l’interface (gain cinétique).
- V(T) = pulsation/latence de l’interface.
- ρ_T, p_T = contenu effectif dû à l’interface dans les équations cosmologiques.

## 8) Prochaines vérifications logiques
1. Stabilité linéaire : modes scalaire et tensorielles autour de FLRW.
2. Contraintes locales : PPN, Ḡ/G, vitesse des ondes gravitationnelles.
3. Dégénérescence avec ΛCDM : régimes où w_eff ≈ −1 et écarts contrôlés pour z ≳ 1.

> Cette version remplace explicitement les dépendances non covariantes par une écriture standard de type k‑essence minimale, prête pour l’analyse de stabilité et l’implémentation numérique.

