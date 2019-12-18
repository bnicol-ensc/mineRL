# Projet Transpromotion MineRL
## Programmation d'un agent RL pour la réalisation du challenge *MineRLNavigateDense-v0*
Equipe d'étudiants :
 - 1A :
   - Guilhem Le Moigne
   - Victor Leroy
   - Mael Lucas
 - 2A :
   - Guillaume Grosse
   - Mateo Mahaut
   - Benjamin Nicol
---
## Fonctionnement du code
[*Code issu du template de la competition MineRL*](https://github.com/minerllabs/competition_submission_starter_template)

Pour faire fonctionner le code il vous faudra lancer les fichiers executables presents dans le dossier *utility* selon le lancement desire veuillez vous reporter au tableau ci dessous :
- *evaluation_locally.sh* : Lance uniquement l'evaluation du modele
- *train_locally.sh* : Lance l'integralite du programme (verification du telechargement, entrainement du modele, et evaluation)
- *verify_or_download_data.py* : Verifie lse dependances de donnees et les telecharge si necessaire

### Structure de fichier
```
.
├── aicrowd.json           # Submission meta information like your username
├── apt.txt                # Packages to be installed inside docker image
├── config-file.txt        # NEAT configuration file
├── data                   # The downloaded data, the path to directory is also available as `MINERL_DATA_ROOT` env variable
├── requirements.txt       # Python packages to be installed
├── run.py                 # Run training or testing phase code
├── saved                  # Trained models are saved inside this directory
├── test.py                # IMPORTANT: Your testing/inference phase code, must include main() method
├── train                  # Your trained model MUST be saved inside this directory, must include main() method
├── train.py               # IMPORTANT: Your training phase code
└── utility                # The utility scripts to provide smoother experience to you.
    ├── debug_build.sh
    ├── docker_run.sh
    ├── environ.sh
    ├── evaluation_locally.sh
    ├── parser.py
    ├── train_locally.sh
    └── verify_or_download_data.sh
```