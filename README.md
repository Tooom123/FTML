# FTML 2026 - Projet

Le sujet complet est dans `subject.pdf`.

## Rapport

- `rendu.tex` / `rendu.pdf` : rapport qui répond aux 7 exercices. Les exercices 1 à 3
  (maths) y sont traités en entier. Les exercices 4 à 7 (code) y sont résumés
  (démarche, résultats, graphiques), le détail est dans les notebooks correspondants.
- `figures/` : graphiques utilisés dans `rendu.pdf`, extraits des notebooks des
  exercices 4 à 7.

## Dossiers

- `exercice1/`, `exercice2/`, `exercice3/` : scripts `simulation.py` (exercices de
  maths, vérifiés par simulation).
- `exercice4/`, `exercice5/` : notebooks de régression et de classification, avec
  les données fournies (`.npy`).
- `exercice6/`, `exercice7/` : notebooks d'application supervisée et non supervisée.
  Les datasets ne sont pas dans le repo (trop volumineux, exclus par
  `.gitignore`) : ils sont à télécharger depuis les liens Kaggle donnés dans les
  notebooks, dans `exercice6/dataset smartphones/` et `exercice7/dataset/`.

Chaque dossier `exerciceN/` contient un `requirements.txt` pour les librairies
utilisées.
