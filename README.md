# Histogram_equalization
Projet de TP GPGPU, Master 1


Compiler le programme:
```sh
nvcc -o hist main.cu image.cpp histoCPU.cpp histoGPU.cu
```
Lancer le programme sur une image png:
```sh
./hist -f /path/to/image.png
```