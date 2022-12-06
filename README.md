# Histogram_equalization
Projet de TP GPGPU, Master 1


Compiler le programme:
```sh
nvcc -o hist main.cu image.cpp histoGPU.cu
```
Lancer le programme sur une image png:
```sh
./hist -f ./img/chateau.png
```