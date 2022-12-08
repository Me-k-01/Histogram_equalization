# Projet égalisation d'histogramme sur GPU
Ce projet à été réalisé dans le cadre de l'UE GPGPU du Master 1 d'informatique de Limoges
Deux étudiants ont travaillés sur ce projet:
- Celestin MARCHAND
- Florian AUBERVAL


Pour compiler ce programme:
```sh
nvcc -o hist ./gpu/main.cu ./gpu/histoGPU.cu image.cpp
```
Pour cancer la répartition de l'histogramme sur une image png:
```sh
./hist -f ./img/chateau.png
```