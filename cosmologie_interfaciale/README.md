# README — Dossier Cosmologie Chronodynamique (CCD)

## Objectif du dossier
Ce dossier rassemble les briques nécessaires pour comprendre et développer la **Cosmologie Chronodynamique (CCD)**. Il est structuré pour être lisible par n’importe quel développeur, chercheur ou intelligence artificielle. Chaque document remplit une fonction précise (fondements théoriques, formalisme mathématique, stabilité, comparaisons, variantes covariantes).

## Structure des fichiers

### 1. **Cosmologie Chronodynamique — Fondements Théoriques**
- Contenu : Version longue type *thèse de physique théorique*.
- Objectif : donner le contexte (tensions H₀, S₈, anomalies JWST, etc.), introduire les piliers conceptuels (causal sets de Sorkin, Rovelli, Wheeler, Merleau‑Ponty), puis formaliser la CCD et ses prédictions.
- Usage : base conceptuelle et philosophique ; document de référence pour publication.

### 2. **Lagrangien Chronodynamique**
- Contenu : équations lagrangiennes détaillées, tenseur C_{μν}, équations de Friedmann modifiées, équation maîtresse.
- Objectif : formaliser mathématiquement la dynamique du champ temporel T(x^μ).
- Usage : cœur du formalisme ; point d’entrée pour implémentation dans codes cosmologiques (CLASS, CAMB).

### 3. **Conditions de Stabilité Causale**
- Contenu : contraintes analytiques pour garantir absence de modes tachyoniques, respect de la causalité, conservation de l’énergie‑impulsion.
- Objectif : vérifier la consistance interne du modèle.
- Usage : check de stabilité théorique et perturbative.

### 4. **CCD vs Causal Set Theory**
- Contenu : comparaison entre CCD et la théorie des ensembles causaux de Sorkin.
- Objectif : situer la CCD par rapport aux approches discrètes de l’espace‑temps.
- Usage : repère conceptuel pour collaborations externes.

### 5. **Équations Interfaciales (version covariante minimale)**
- Contenu : reformulation covariante du modèle en termes de champ scalaire T(x), invariant X, fonctions A(T) et V(T).
- Objectif : éliminer les dépendances non‑covariantes, rendre l’action utilisable pour simulations et stabilité.
- Usage : version « prête à coder », structure de type k‑essence.

### 6. **Stabilité Linéaire — CCD Interfaciale**
- Contenu : dérivation des équations de perturbations δT, conditions no‑ghost, c_s², m_eff², action quadratique.
- Objectif : fournir critères de santé du modèle autour d’un fond FLRW.
- Usage : mode d’emploi pour tests numériques et simulations.

## Logique globale
- **Fondements théoriques** → cadrage conceptuel et philosophique.
- **Lagrangien** → écriture formelle complète (première version non minimale, semi‑covariante).
- **Équations interfaciales** → version covariante minimale, utilisable en code.
- **Conditions de stabilité** + **Stabilité linéaire** → garanties de cohérence et de consistance dynamique.
- **Comparaison causets** → positionnement vis‑à‑vis d’autres approches.

## Pour les développeurs
- Point d’entrée recommandé pour coder : **Équations Interfaciales (covariante minimale)**.
- Tester la stabilité : utiliser **Stabilité Linéaire** (ODE à implémenter pour δT_k(t)).
- Vérifier cohérence : appliquer **Conditions de Stabilité Causale**.
- Pour documentation ou vulgarisation : se référer à **Fondements Théoriques**.

## Notes finales
- Les deux variantes (ancienne version semi‑covariante avec f(t), h(t), et version covariante minimale avec A(T), V(T)) sont présentes. La seconde est à privilégier pour le développement effectif.
- Le dossier est pensé comme un écosystème : chaque fichier correspond à une « couche » (concept, formalisme, stabilité, comparaisons).
- Toute implémentation numérique doit partir de la version covariante minimale afin de respecter covariance et éviter les pathologies.

---

**En résumé** : ce dossier fournit tout le nécessaire pour comprendre, tester et implémenter la CCD, du socle conceptuel jusqu’aux équations prêtes à coder.

