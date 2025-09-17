## Exercices sur Git & GitHub

### Exercice 1 : pull request et peer review sur le depo du cours

Fait sur le repo de la classe `class_repo`

### Exercice 2 : gérer les branches et merge conflicts dans le dossier math_docs

### Exercice 3 : collaborer sur le dossier math_docs (pas encore fait)

### Exercice 4 : créer des github actions pour math_docs 

disponible dans .github/workflows/

Workflows ajoutés :
- `format-maths-docs-markdown.yml` : formate et génère une table des matières pour les fichiers Markdown sous `data_science_practice/1_git_&github/`.
- `check-maths-docs.yml` : vérifie que `data_science_practice/1_git_&github/math_docs/README.md` existe et n'est pas vide (échoue sinon).
- `welcome.yml` : envoie un message de bienvenue aux nouveaux contributeurs qui ouvrent une issue. (dans le dépôt principal)

Note sur `TOC.md` :
- Le workflow `format-maths-docs-markdown.yml` génère `TOC.md` dans `data_science_practice/1_git_&github/` et tente de le committer/pusher automatiquement si son contenu change.
- Pour que le push fonctionne, le workflow doit avoir `permissions: contents: write` et `actions/checkout` doit persister les credentials (ou alors pousser vers une branche et ouvrir une PR).

### Exercice 5 : contribuer à sklearn

