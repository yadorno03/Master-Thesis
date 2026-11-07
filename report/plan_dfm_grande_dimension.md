# Plan : faire fonctionner le DFM en dimension ≥ 500, puis sur le S&P 500

**Statut : plan, rien n'est lancé.** Rédigé le 7 novembre 2026, après les diagnostics du
run d = 2048 ([08](08_grande_dimension_d2048.md)). Les règles de décision de la
section 3 sont à fixer dans un addendum de
[protocol_preregistration.md](protocol_preregistration.md) avant tout calcul.

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
  d = 512 et environ 830 à d = 2048.

**Ce qui a déjà été testé à d = 2048** (une graine, 1 000 epochs sauf mention) :

| Variante | Rapport de variance des facteurs (cible : 1) | Verdict |
| --- | --- | --- |
| Largeur 64 (run Modal) | 0,31–0,58 | référence |
| Largeur 64, +1 500 epochs | 0,41–0,61 | petit gain |
| Entrées de g mises à l'échelle | 0,44–0,59 | meilleure perte, facteurs presque inchangés |
| Largeur 256 | 0,53–0,78 | mieux |
| 3 couches de 512, epoch 500 | 0,74–1,00 | presque bon… |
| 3 couches de 512, epoch 1 000 | 0,47–0,91 | …puis recule, alors que la perte baisse |
| Erreur sur les facteurs pondérée par leur variance (2 versions) | instable, puis 0,03–0,10 | échec : poids jusqu'à environ 830 fois aux petits t, gradients trop bruités |

**Deux leçons pour la suite :**
- la perte de validation ne sélectionne pas les bons facteurs ;
- les variances des facteurs oscillent d'un checkpoint à l'autre.

Les scripts, logs et checkpoints sont dans
`artifacts/benchmark/d2048_k16/diagnostics_architecture_20261107/` (local).

## 2. Pourquoi c'est le bon problème pour le S&P 500

- **Le S&P 500 est dans le même régime difficile.** Le facteur de marché grandit avec le
  nombre d'actifs.
  - Ordre de grandeur, à vérifier sur les données : s'il explique 25 à 40 % de la
    variance de rendements standardisés, sa valeur propre vaut environ 125 à 200 pour
    500 actifs.
  - Avec une variance idiosyncratique d'environ 0,6, on obtient λ/c ≈ 200 à 350,
    entre notre d = 512 et notre d = 2048.
- **L'historique est court.** Dix ans de données quotidiennes donnent environ 2 500
  observations, contre 12 000 dans nos simulations.
- **Les rendements ne sont pas gaussiens :** queues épaisses, volatilité qui varie,
  changements de régime. Un modèle factoriel gaussien estimé par PCA donne une loi
  conditionnelle analytique : c'est la référence que le DFM doit battre.
- **Il n'y a pas de vérité sur données réelles.** Il faut donc valider sur des
  simulations réalistes avant de passer au S&P 500.

## 3. Critères de succès (à fixer avant de lancer)

Configuration : d = 512, k = 16, N = 2 500 et N = 12 000 observations, 3 graines.

1. **Facteurs** : rapport de variance entre 0,85 et 1,15 pour chaque facteur
   identifiable (λ ≥ 5 c̄), en moyenne sur les graines.
2. **Covariance** : erreur au plus égale à celle de l'estimateur empirique (PCA sur les
   mêmes données d'entraînement). C'est le critère « Diff contre Emp » de Chen et
   al. (2025).
3. **Portefeuille** : KS au plus 1,5 fois le niveau Monte Carlo.
4. **Conditionnel**, événements à 5 % et à 1 % :
   - le rejet du DFM est à moins de 2 fois le niveau Monte Carlo de la vérité, en
     sliced W1 ;
   - Refined CDG-ML + dernier pas exact est au niveau du bruit face au rejet du DFM.
5. **Pas de régression à d = 128.**

## 4. Les idées, par ordre de priorité

| # | Idée | Ce qu'elle corrige | Effort | Risque | Écart à Chen et al. (2025) |
| --- | --- | --- | --- | --- | --- |
| I1 | Choisir le checkpoint sur un critère factoriel | sélection aveugle aux facteurs | faible | faible | aucun |
| I2 | Préconditionner les données : réduire les directions des facteurs | perte aveugle aux facteurs | faible | moyen | prétraitement seulement |
| I3 | Réseau g plus fort dans la forme (18), avec moyenne mobile des poids | capacité, oscillations | faible à moyen | faible | aucun |
| I4 | Meilleur échantillonneur | variance idiosyncratique gonflée de 20 à 30 % | faible | faible | aucun |
| I5 | Partie linéaire explicite dans g | précision demandée à g | moyen | moyen | modéré |
| I6 | V gelé après la PCA, batchs plus grands | bruit des gradients | faible | faible | léger |
| I7 | Pondération ciblée sur les temps intermédiaires | perte aveugle aux facteurs | moyen | élevé | modéré |
| I8 | Leur U-Net comme simple référence inconditionnelle | savoir si l'écart vient de l'architecture | moyen (GPU) | faible | aucun |

### Priorité 1 : peu coûteux, gain probable

**I1. Un critère factoriel pour choisir le checkpoint.**
- Pendant l'entraînement, générer régulièrement quelques milliers de tirages.
- Comparer leur covariance, dans les coordonnées PCA, à celle de l'échantillon de
  validation (rapports de variance des facteurs, erreur de covariance), sans utiliser
  la vérité.
- Choisir le checkpoint sur ce critère, et non plus sur la perte de validation.
- C'est un préalable : sans lui, un bon modèle peut être perdu au moment de la
  sélection.

**I2. Préconditionner les données.**
- **Idée.** Avant l'entraînement, réduire l'échelle des directions des facteurs pour
  ramener λ/c au niveau de d = 128, où tout marche.
  - Avec V̂ et λ̂ estimés par PCA sur les données d'entraînement, on pose
    r̃ = P r, avec P = I + V̂ (diag(s) − I) V̂ᵀ et s_j = √(ρ c̄ / λ̂_j) pour les
    facteurs où λ̂_j > ρ c̄.
  - ρ est le rapport visé, par exemple 20.
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
- **Test** : ρ ∈ {10, 20, 50} à d = 512 et d = 2048.

**I3. Un réseau g plus fort, dans la forme (18).**
- On garde V, c, D_t et la connexion −D_t r, et on ne change que g :
  - blocs résiduels ;
  - activation SiLU ;
  - encodage sinusoïdal du temps ;
  - largeur 256 à 512, 3 à 4 blocs.
- On ajoute une **moyenne mobile exponentielle des poids** (taux 0,999, comme dans le
  code publié des auteurs). Elle vise directement les oscillations observées avec 3
  couches de 512.
- La théorie de Chen et al. (2025) autorise n'importe quel réseau g.
- **Test** : g actuel contre g résiduel, avec et sans moyenne mobile.

**I4. Un meilleur échantillonneur.**
- Avec le vrai score, Euler à 400 pas gonfle la variance idiosyncratique d'environ 30 %
  à d = 2048, et d'environ 20 % à d = 128 :
  - environ 9 points viennent de l'arrêt à t0 = 0,005 ;
  - le reste vient du pas d'Euler.
- Pistes :
  - intégrer exactement la partie linéaire de l'équation d'Ornstein-Uhlenbeck ;
  - utiliser une grille de temps logarithmique ;
  - débruiter au dernier pas (formule de Tweedie).
- **Point à vérifier.** Le dernier pas exact garantit l'événement sur l'état final. Un
  débruitage final ne doit pas le casser.
- C'est un effet secondaire, mais il pèse sur l'erreur de covariance à toutes les
  dimensions.

### Priorité 2 : si la priorité 1 ne suffit pas

**I5. Une partie linéaire explicite dans g.**
- On écrit g(u, t) = g_lin(u, t) + MLP(u, t), où :
  - g_lin est l'espérance a posteriori des facteurs pour un modèle gaussien, calculée
    à partir des estimations PCA (moyenne, covariance des facteurs, c) ;
  - les paramètres de g_lin restent entraînables ;
  - le MLP part de zéro et n'apprend que les écarts au cas gaussien.
- La partie difficile du problème, une application affine très précise, devient
  connue en forme fermée.
- **Attention.** Sur la population gaussienne, le problème devient presque trivial. Il
  faut donc tester sur des populations non gaussiennes (section 5).

**I6. V gelé après la PCA, batchs plus grands.**
- V et c issus de la PCA sont déjà excellents. Les geler (ou geler seulement V) enlève
  du bruit et des paramètres.
- Des batchs de 1 024 au lieu de 256 réduisent le bruit des gradients dans les
  directions des facteurs.

**I7. Une pondération ciblée sur les temps intermédiaires.**
- La pondération naïve a échoué : poids d'environ 830 aux petits t, où le bruit domine.
- Variante :
  - pondérer seulement les temps où la variance des facteurs se décide (t ≈ 0,3 à 4) ;
  - plafonner le poids, par exemple à 10 ;
  - ou tirer plus souvent ces temps-là.
- Expérimental.

### Priorité 3 : seulement comme référence

**I8. Le U-Net des auteurs comme référence inconditionnelle.**
- Il a 1,6 milliard de paramètres à d = 2048 et demande un GPU de 40 à 80 Go.
- Il sert seulement à savoir si l'écart vient de notre architecture.
- Il n'est pas utilisable dans le pipeline conditionnel :
  - il n'a ni V ni c, donc H devrait travailler sur toutes les coordonnées ;
  - le guidage n'y serait qu'approché.
- Le code d'entraînement existe sur la branche `review/dfm-06-author-training`.
- Un U-Net maison n'est pas recommandé : il aurait les mêmes défauts, sans la
  légitimité du code publié.

## 5. Populations de test, du plus simple au plus réaliste

- **P1. Échelle de dimension**, avec la population gaussienne de Chen et al. (2025) :
  d ∈ {128, 256, 512, 1 024, 2 048}. Elle sert à situer où le DFM casse, puis à
  vérifier chaque amélioration.
- **P2. Population calibrée sur le S&P 500.**
  - Charges, covariance des facteurs et variances idiosyncratiques sont estimées par
    PCA sur les vrais rendements, avec d ≈ 500 et k choisi selon Bai et al. (2002).
  - Les tirages restent gaussiens. La vérité est donc connue, avec un spectre de
    valeurs propres réaliste.
- **P3. Même population avec des facteurs et des résidus de Student** (4 à 5 degrés de
  liberté).
  - On teste ainsi les queues épaisses et la partie non linéaire de g.
  - La vérité conditionnelle s'obtient par rejet exact sur la population : c'est peu
    coûteux, car il n'y a pas de diffusion à simuler.
- **Pour chaque population** : N = 2 500 (comme l'historique du S&P 500) et
  N = 12 000.

## 6. Étapes

| Étape | Contenu | Calcul indicatif |
| --- | --- | --- |
| 0 | Addendum au protocole ; critère I1 ; référence sur P1 à d = 512, 3 graines | quelques heures-conteneur CPU (Modal) |
| 1 | I2, I3, I4 seuls puis combinés, à d = 512 et d = 2048 | environ 5 à 15 min par entraînement à d = 512, en parallèle sur Modal |
| 2 | Si les critères ne sont pas atteints : I5, I6, I7 | idem |
| 3 | Recette retenue sur P2 et P3, avec tout le pipeline conditionnel (H 40k / 50, événements à 5 % et 1 %) | quelques heures-conteneur |
| 4 | S&P 500 réel, voir section 7 | à évaluer |

**Ce que je ferais en premier :** I1, puis I2 et I3 (avec la moyenne mobile) à
d = 512, sur 3 graines. Ce sont les idées les moins chères, et chacune vise une cause
mesurée.

## 7. Passage au S&P 500 réel

- **Données.**
  - Composition sans biais du survivant.
  - Rendements quotidiens log, standardisés actif par actif.
  - Winsorisation à 2,5 % de chaque côté, comme Chen et al. (2025).
  - Fenêtres glissantes.
  - Le code de prétraitement existe déjà : `equity_preprocessing.py`,
    `return_preprocessing.py`.
- **Nombre de facteurs.** Critère de Bai et al. (2002), avec un test de robustesse sur
  k ∈ {5, 8, 10, 16}.
- **Évaluation sans vérité :**
  - rejet du DFM contre guidage, comme aujourd'hui ;
  - comparaison au modèle factoriel gaussien estimé par PCA, dont la loi
    conditionnelle est analytique ;
  - confrontation aux jours de stress observés hors échantillon.
- **Guidage.** Utiliser Refined CDG-ML seulement : CDG-ML sans raffinements s'effondre
  en grande dimension.

## 8. Risques et questions ouvertes

- **Rédaction.** I2 et I5 sont nos choix. Il faudra les présenter comme tels, sans les
  attribuer à Chen et al. (2025).
- **Rendements non indépendants.** Le DFM suppose des rendements indépendants et de
  même loi ; ce n'est pas le cas sur dix ans. Les fenêtres glissantes et la
  dépendance au régime sont hors du périmètre de départ.
- **Estimation de V̂ et λ̂ (I2, I5).** Avec N = 2 500 et d = 500, elle est raisonnable,
  mais à vérifier sur P2.
- **Une seule graine** pour tous les résultats de la section 1 : les ordres de grandeur
  sont fiables, pas les petites différences.
