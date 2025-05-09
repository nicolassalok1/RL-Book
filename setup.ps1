#RUN CA EN AMONT :
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#puis :
#./setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "Installation des dépendances Python..."
#
conda env create -f environment.yml
conda activate RL_Book
pip install -r requirements.txt


Write-Host "Setup termine avec succes."