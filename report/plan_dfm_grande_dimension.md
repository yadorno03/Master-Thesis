# Plan : faire fonctionner le DFM en dimension ≥ 500, puis sur le S&P 500

**Statut : plan, rien n'est lancé.** Rédigé le 7 novembre 2026, après les diagnostics du
run d = 2048 ([08](08_grande_dimension_d2048.md)).
- Ce plan fusionne deux propositions écrites le même jour.
- Les règles de décision de la section 4 sont à fixer dans un addendum de
  [protocol_preregistration.md](protocol_preregistration.md) avant tout calcul.

**Deux jalons :**
1. une génération fiable à d = 500 sur des simulations ;
2. une génération fiable pour environ 500 actions réelles.

Le second jalon ne demande pas seulement plus de dimensions. Il faut aussi gérer les
queues épaisses, la volatilité qui varie et le peu de données.

## 1. Point de départ

**Ce qui marche.** À d = 128, le DFM et le guidage Refined CDG-ML + dernier pas exact
fonctionnent ([07](07_notebook_quatre_sources_d128.md)).

**Ce qui casse à d = 2048.** Les tirages du DFM n'ont que 31 % à 58 % de la vraie
variance des facteurs. Le reste du modèle est bon :
- les charges V sont justes : 15 facteurs sur 16 avec un cosinus > 0,993 ; le 16ᵉ a une
  variance de 0,005 et ne peut pas être identifié ;
- les variances idiosyncratiques c sont justes : corrélation de 0,999 avec la vérité ;
- l'échantillonneur n'est pas en cause : avec le vrai score et le même schéma d'Euler,
  les variances des facteurs sont justes (0,95 à 1,05).

**La cause.** La perte d'entraînement voit à peine la variance des facteurs.
- Diviser par deux toutes les variances des facteurs, le reste du score étant exact, ne
  coûte que +0,40 de perte sur 534,5, soit 0,07 %. L'écart-type de la perte d'un batch
  est d'environ 34.
- Le score dans la direction d'un facteur est de l'ordre de 1/variance. Plus le rapport
  entre variance d'un facteur et variance idiosyncratique (λ/c) est grand, moins ces
  directions comptent dans la perte.
- Dans la population de Chen et al. (2025), λ grandit avec d alors que c reste vers
  0,05. Le rapport λ/c du plus gros facteur vaut environ 50 à d = 128, environ 200 à
  d = 500 et environ 830 à d = 2048.

**Ce qui a déjà été testé à d = 2048** (une graine, 1 000 epochs sauf mention) :

| Variante | Rapport de variance des facteurs (cible : 1) | Verdict |
| --- | --- | --- |
| Largeur 64 (run Modal) | 0,31–0,58 | référence |
| Largeur 64, +1 500 epochs | 0,41–0,61 | petit gain |
| Entrées de g multipliées par un même facteur d'échelle | 0,44–0,59 | meilleure perte, facteurs presque inchangés |
| Largeur 256 | 0,53–0,78 | mieux |
| 3 couches de 512, epoch 500 | 0,74–1,00 | presque bon… |
| 3 couches de 512, epoch 1 000 | 0,47–0,91 | …puis recule, alors que la perte baisse |
| Erreur sur les facteurs pondérée par leur variance (2 versions) | instable, puis 0,03–0,10 | échec : poids jusqu'à environ 830 fois aux petits t, gradients trop bruités |

**Ce qu'on en retient :**
- la perte de validation ne sélectionne pas les bons facteurs ;
- les variances des facteurs oscillent d'un checkpoint à l'autre ;
- à d = 128, la phase 2 ([04](04_phase2_budgets_dfm.md)) n'avait trouvé que des gains
  modestes en ajoutant des epochs ou de la largeur.

Les scripts, logs et checkpoints sont dans
`artifacts/benchmark/d2048_k16/diagnostics_architecture_20261107/` (local).

## 2. Ce que disent les données réelles

Ces chiffres viennent du panel préparé le 29 octobre :
`artifacts/market_data_cleaning/20261029T104817954744Z/findings.md`.
- **Le panel :**
  - 800 actions ;
  - entraînement 2010–2019 (2 516 jours), validation 2020–2022 (756 jours), test
    2023–2025 (750 jours) ;
  - n/d = 3,15 sur l'entraînement.
- **Le spectre (matrice de corrélation des rendements d'entraînement) :**
  - le premier facteur, le marché, a une valeur propre de 275,5, soit 34 % de la
    variance ;
  - les suivants valent 28,5, 14,6, 11,6 et 9,7 ;
  - après 10 facteurs, la variance résiduelle moyenne est d'environ 0,54.
- **Conséquence.**
  - Seul le facteur de marché est dans le régime difficile : λ/c ≈ 510 à d = 800, soit
    environ 300 à d = 500.
  - Les autres facteurs ont un rapport entre 5 et 50, comparable à notre d = 128, où
    tout marche.
  - Or c'est justement le facteur de marché qui pilote le portefeuille équipondéré et
    ses queues.
- **Peu de données.** Avec 2 516 jours, les simulations doivent aussi être faites avec
  N ≈ 2 500 observations, pas seulement 12 000.
- **Univers.** Il a été construit avec les composants disponibles au moment de la
  récupération des données, d'où un biais du survivant (section 8).
- **Événements.** Le seuil à 5 % de l'entraînement donne 10,6 % de jours en événement
  sur la validation (période Covid). Les périodes calmes et de stress sont donc à
  évaluer séparément.

## 3. Règles de conduite des expériences

- **Échelle de dimension** :
  - d = 128, 256 et 500, avec un nombre de facteurs fixe à l'intérieur de chaque
    comparaison ;
  - puis une confirmation à d = 1 024, et d = 2 048 comme test de résistance.
- **Données** : N = 12 000 et N = 2 500 observations.
- **Graines** :
  - un premier tri sur une graine ;
  - puis les meilleures pistes et la référence confirmées sur 3 graines au moins.
- **Comparaisons équitables** : même budget d'entraînement, et temps de calcul réel
  rapporté.
- **Données réservées** : un jeu d'évaluation final, jamais utilisé pour choisir un
  modèle ou un checkpoint.
- **Checkpoints complets et reprenables**, avec l'état de l'optimiseur, ce qui n'est
  pas le cas aujourd'hui.
- **Chaque nouveau DFM demande un nouvel entraînement de H.**

## 4. Critères de succès (à fixer avant de lancer)

**Étape gaussienne, d = 500 :**
1. **Facteurs** : variance de chaque facteur identifiable (λ ≥ 5 c̄) entre 90 % et
   110 % de la vérité, en moyenne sur les graines. Avec 3 000 tirages, le bruit
   d'échantillonnage sur ce rapport est d'environ ±3 %.
2. **Covariance** :
   - erreur nettement plus basse que le DFM actuel ;
   - au moins aussi bonne que l'estimateur empirique (PCA sur les mêmes données), le
     critère « Diff contre Emp » de Chen et al. (2025) ;
   - comparable à la référence gaussienne ajustée (idée I0).
3. **Distances** (sliced W1, KS des portefeuilles) proches du niveau Monte Carlo à
   taille égale, et de la référence gaussienne ajustée.
4. **Queues** : pas de dégradation des probabilités de queue de plusieurs portefeuilles
   fixés à l'avance : équipondéré, sectoriels, exposés à un facteur.
5. **Stabilité** : mêmes conclusions d'une graine à l'autre et d'une résolution
   d'échantillonnage à l'autre.
6. **Pas de régression à d = 128.**

**Conditionnel** (après l'étape gaussienne), événements à 5 % et à 1 % :
- le rejet du DFM est à moins de 2 fois le niveau Monte Carlo de la vérité ;
- Refined CDG-ML + dernier pas exact est au niveau du bruit face au rejet du DFM.

Ce sont des objectifs proposés, pas des capacités établies.

**Ce qu'on mesure, au-delà de la perte :**
- la variance des facteurs et la variance résiduelle, séparément ;
- la covariance et les premières valeurs propres ;
- le sliced W1 ;
- l'erreur du score selon le temps de diffusion, sur les simulations gaussiennes ;
- les probabilités de queue des portefeuilles fixés à l'avance.

## 5. Les idées, par ordre de priorité

Toutes les premières améliorations restent **dans la forme factorielle du score**
(formule (18) de Chen et al. (2025)). C'est ce qui garde le guidage par coordonnées
réduites utilisable.

| # | Idée | Ce qu'elle corrige | Effort | Écart à Chen et al. (2025) |
| --- | --- | --- | --- | --- |
| I0 | Référence gaussienne ajustée | donne une cible atteignable, sépare estimation et échantillonnage | faible | aucun (référence) |
| I1 | Choisir le checkpoint sur un critère factoriel | sélection aveugle aux facteurs | faible | aucun |
| I2 | Préconditionner les données : réduire les directions des facteurs | perte aveugle aux facteurs | faible | prétraitement seulement |
| I3 | Partie gaussienne explicite dans g, plus une correction neuronale | précision demandée à g | moyen | modéré |
| I4 | Entrées de g standardisées par facteur et par temps | échelle des entrées | faible | léger |
| I5 | Réseau g plus fort dans la forme (18) | capacité | faible à moyen | aucun |
| I6 | Optimisation : moyenne mobile, baisse du pas, écrêtage, V gelé | oscillations, bruit des gradients | faible | léger |
| I7 | Plus de pas près de la fin de l'échantillonnage | variance résiduelle gonflée de 20 à 30 % | faible | aucun |
| I8 | Pondération ciblée sur les temps intermédiaires | perte aveugle aux facteurs | moyen | modéré |
| I9 | U-Net comme référence inconditionnelle | savoir si l'écart vient de l'architecture | moyen à élevé | aucun |

### Priorité 1

**I0. Une référence gaussienne ajustée.**
- On ajuste un modèle gaussien « facteurs + diagonale » sur les données d'entraînement
  : moyenne, V̂ et λ̂ par PCA, c par les résidus.
- On compare deux façons d'en tirer des échantillons :
  - des tirages directs ;
  - des tirages par notre échantillonneur, avec le score analytique de ce modèle.
- Les tirages directs montrent ce qui est atteignable avec N observations (l'erreur
  d'estimation seule). L'écart entre les deux mesure l'erreur de l'échantillonneur.
- Sur données réelles, cette référence devient le modèle à battre.

**I1. Un critère factoriel pour choisir le checkpoint.**
- Pendant l'entraînement, générer régulièrement quelques milliers de tirages.
- Comparer leur covariance, dans les coordonnées PCA, à celle de l'échantillon de
  validation (pas au jeu d'évaluation final).
- Choisir le checkpoint sur ce critère plutôt que sur la perte de validation.
- C'est un préalable : sans lui, un bon modèle peut être perdu au moment de la
  sélection.

**I2. Préconditionner les données.**
- **Idée.** Avant l'entraînement, réduire l'échelle des directions des facteurs pour
  ramener λ/c au niveau de d = 128.
  - Avec V̂ et λ̂ estimés par PCA sur les données d'entraînement, on pose
    r̃ = P r, avec P = I + V̂ (diag(s) − I) V̂ᵀ et s_j = √(ρ c̄ / λ̂_j) pour les
    facteurs où λ̂_j > ρ c̄.
  - ρ est le rapport visé, par exemple 20.
  - Sur données réelles, seul le facteur de marché serait réduit.
- **Utilisation.** On entraîne le DFM sur r̃, puis on revient aux rendements par
  r = P⁻¹ r̃.
- **Compatibilité avec le pipeline.** L'événement reste linéaire : w′r ≤ C équivaut à
  (P⁻ᵀw)′ r̃ ≤ C.
  - H, le guidage et le dernier pas exact fonctionnent donc sans changement dans les
    coordonnées r̃.
  - Le DFM reste celui de Chen et al. (2025) : seul le prétraitement change.
- **Limite.** Dans r̃, le bruit idiosyncratique n'est plus exactement diagonal.
  - L'écart reste confiné au sous-espace des facteurs.
  - Il est de l'ordre de la dispersion des variances idiosyncratiques.
- **Test** : ρ ∈ {10, 20, 50}.

**I3. Une partie gaussienne explicite dans g, plus une correction neuronale.**
- On écrit g(u, t) = g_lin(u, t) + MLP(u, t), où :
  - g_lin est l'espérance a posteriori des facteurs du modèle gaussien ajusté (I0) ;
  - ses paramètres restent entraînables ;
  - le MLP part de zéro et n'apprend que les écarts au cas gaussien.
- La partie difficile, une application affine très précise, devient connue en forme
  fermée. C'est l'esprit du préconditionnement des réseaux de diffusion de Karras et
  al. (2022).
- **Attention.** Sur la population gaussienne, le problème devient presque trivial. La
  valeur de I3 se juge donc sur les populations non gaussiennes (section 6).

**I4. Des entrées de g standardisées par facteur et par temps.**
- Standardiser ou blanchir les coordonnées projetées Vᵀ D_t r, séparément pour chaque
  temps de diffusion, avec des statistiques calculées sur l'entraînement.
- La formule extérieure du score ne change pas.
- **Attente modérée.** Une version plus simple, avec le même facteur d'échelle pour
  toutes les coordonnées, a amélioré la perte mais pas les facteurs (section 1). La
  version par facteur n'est pas testée et reste peu coûteuse.

### Priorité 2

**I5. Un réseau g plus fort, dans la forme (18).**
- Blocs résiduels, activation lisse (SiLU), encodage du temps plus riche.
- Largeur 128, puis 256 ou 512.
- D'abord un changement à la fois.
- 3 couches de 512 a approché la bonne variance à l'epoch 500, avant de reculer.

**I6. L'optimisation.**
- Une moyenne mobile exponentielle des poids, au taux de 0,999 comme dans le code
  publié des auteurs. Elle vise les oscillations observées.
- Une baisse progressive du pas d'apprentissage.
- L'écrêtage des gradients.
- Des batchs plus grands.
- V gelé après la PCA au début, puisque V et c issus de la PCA sont déjà excellents.

**I7. Plus de pas près de la fin de l'échantillonnage.**
- Avec le vrai score, Euler à 400 pas gonfle la variance résiduelle d'environ 30 % à
  d = 2048 :
  - environ 9 points viennent de l'arrêt à t0 = 0,005 ;
  - le reste vient du pas d'Euler.
- Comparer la grille uniforme actuelle à une grille qui met plus de pas près du faible
  bruit, ou intégrer exactement la partie linéaire.
- Valider d'abord avec le score analytique, pour ne pas confondre erreur numérique et
  erreur d'apprentissage.
- Le dernier pas exact du guidage doit rester compatible avec la nouvelle grille.

**I8. Une pondération ciblée sur les temps intermédiaires.**
- La pondération naïve a échoué : poids d'environ 830 aux petits t, où le bruit domine.
- Variante :
  - pondérer seulement les temps où la variance des facteurs se décide (t ≈ 0,3 à 4) ;
  - plafonner le poids, par exemple à 10.
- Expérimental.

### Priorité 3 : seulement comme référence

**I9. Une comparaison avec un U-Net.**
- Elle sert à savoir si une architecture plus large apporte un vrai gain. Elle ne sert
  pas au pipeline conditionnel :
  - un U-Net n'a ni V ni c, donc H devrait travailler sur toutes les coordonnées ;
  - le guidage demanderait une adaptation à part.
- Deux variantes possibles :
  - **le U-Net des auteurs** : 1,6 milliard de paramètres à d = 2048, GPU de 40 à 80 Go,
    code d'entraînement sur la branche `review/dfm-06-author-training` ;
  - **un U-Net local de taille moyenne**, en traitant explicitement l'ordre des actifs
    et le remplissage nécessaire pour former une image. Il est plus léger, mais n'a pas
    la légitimité du code publié.
- À ne lancer qu'après les priorités 1 et 2.

## 6. Populations de test, du plus simple au plus réaliste

- **P1. Échelle de dimension**, avec la population gaussienne de Chen et al. (2025) :
  d ∈ {128, 256, 500, 1 024, 2 048}.
- **P2. Population calibrée sur le panel réel.**
  - Charges, covariance des facteurs et variances résiduelles sont estimées par PCA sur
    les rendements d'entraînement, avec d ≈ 500.
  - Les tirages restent gaussiens. La vérité est donc connue, avec le vrai spectre : un
    très gros facteur de marché, puis des facteurs modestes.
- **P3. Plus de réalisme.** On ajoute :
  - des facteurs à queues épaisses (loi de Student) ;
  - des mélanges de volatilité ;
  - une dépendance entre résidus, par exemple par secteur.

  On teste ainsi si l'hypothèse de résidus diagonaux et gaussiens devient la limite
  suivante. La vérité conditionnelle s'obtient par rejet exact sur la population : c'est
  peu coûteux, car il n'y a pas de diffusion à simuler.
- **Nombre de facteurs** choisi par validation, plutôt que k = 16 d'office.
- **Pour chaque population** : N = 2 500 et N = 12 000.

## 7. Étapes

| Étape | Contenu | Calcul indicatif |
| --- | --- | --- |
| 0 | Addendum au protocole ; critère I1 ; checkpoints reprenables ; référence I0 et DFM actuel sur P1 à d = 128, 256 et 500 | quelques heures-conteneur CPU (Modal) |
| 1 | **Première expérience**, voir ci-dessous | environ 5 à 15 min par entraînement à d = 500, en parallèle |
| 2 | Confirmation sur 3 graines des meilleures pistes ; I5, I6, I7 si besoin ; confirmation à d = 1 024 | quelques heures-conteneur |
| 3 | Recette retenue sur P2 et P3, avec tout le pipeline conditionnel (nouvel H 40k / 50, événements à 5 % et 1 %) | quelques heures-conteneur |
| 4 | Données réelles, voir section 8 | à évaluer |

**Première expérience** : à d = 500, avec N = 2 500 et N = 12 000, sur une graine.
Chaque variante est choisie avec le critère I1 :
1. le DFM actuel ;
2. la référence gaussienne ajustée (I0) ;
3. le DFM sur données préconditionnées (I2) ;
4. le DFM avec partie gaussienne et correction neuronale (I3) ;
5. le DFM avec entrées standardisées par facteur (I4).

On confirme ensuite les deux meilleures variantes et la référence sur 3 graines.

## 8. Passage aux données réelles

- **Partir du panel existant** (`artifacts/market_data_cleaning/20261029T104817954744Z/`) :
  - réconcilier ses découpages chronologiques avec l'ancien adaptateur de prétraitement
    (`return_preprocessing.py`, `equity_preprocessing.py`) ;
  - enregistrer une seule transformation ajustée sur l'entraînement, avec la convention
    d'unités des rendements et l'ordre des actifs.
- **Univers.**
  - Commencer avec une cohorte fixe et documentée.
  - Avant de parler d'une étude historique du S&P 500, obtenir les dates d'entrée et de
    sortie de l'indice et les radiations, et définir l'univers selon la méthodologie
    officielle de l'indice.
  - Le panel actuel a un biais du survivant et un biais vers les historiques longs.
- **Modèles de comparaison** :
  - gaussien avec covariance rétrécie ;
  - gaussien à facteurs (I0) ;
  - à facteurs avec queues épaisses.
- **Évaluation.**
  - Séparer les périodes calmes et de stress.
  - Faire le rejet du DFM contre le guidage, comme aujourd'hui.
  - Confronter aux jours de stress observés hors échantillon.
- **Guidage.** Utiliser Refined CDG-ML seulement : CDG-ML sans raffinements s'effondre
  en grande dimension.
- **Plus tard.** Conditionner sur la volatilité récente, seulement quand le modèle
  inconditionnel marche.

## 9. Risques et questions ouvertes

- **Rédaction.** I2 et I3 sont nos choix. Il faudra les présenter comme tels, sans les
  attribuer à Chen et al. (2025).
- **Rendements non indépendants.** Le DFM suppose des rendements indépendants et de
  même loi ; ce n'est pas le cas sur dix ans (volatilité groupée, régimes).
- **Estimation de V̂ et λ̂ (I0, I2, I3).** Avec n/d ≈ 3 à 5, elle est raisonnable pour le
  facteur de marché, moins pour les petits facteurs. À vérifier sur P2.
- **Une seule graine** pour tous les résultats de la section 1 : les ordres de grandeur
  sont fiables, pas les petites différences.
