#RUN CA EN AMONT :
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#puis :
#./setup.ps1

# Arrêter en cas d’erreur
$ErrorActionPreference = "Stop"

conda deactivate
conda env remove --name RL_Book
conda create --name RL_Book python=3.8
conda activate RL_Book

python.exe -m pip install --upgrade pip

Write-Host "Installation des dépendances Python..."
#pip install -r requirements.txt
conda env update --file=environment-full.yml

Write-Host "Setup termine avec succes."