# Projet égalisation d'histogramme sur GPU
Ce projet à été réalisé dans le cadre de l'UE GPGPU du Master 1 d'informatique de Limoges
Deux étudiants ont travaillés sur ce projet:
- Celestin MARCHAND
- Florian AUBERVAL


Pour compiler ce programme:
```sh
nvcc -o hist ./gpu/main.cu ./gpu/histoGPU.cu ./utils/image.cpp
```
Pour lancer la répartition de l'histogramme sur une image png:
```sh
./hist -f ./img/chateau.png
```

Le benchmarking du programme est disponnible dans le fichier benchmark.ipynb