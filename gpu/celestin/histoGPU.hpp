#ifndef __HISTOGRAMME_GPU__
#define __HISTOGRAMME_GPU__

#include <vector>

#include "../../image.hpp"

// fonction d'appel au fonction gpu
void gpuCall(const Image & f_ImageIn, int nbEchantillon, Image & f_ImageOut);

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
__global__ void rgb2hsv(const unsigned char f_PixelTable[], unsigned long f_sizeTable, float f_HueTable[],float f_SaturationTable[],float f_ValueTable[]);

// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
__global__ void hsv2rgb(const float f_HueTable[],const float f_SaturationTable[],const float f_ValueTable[], unsigned long f_sizeTable, unsigned char f_PixelTable[]);


// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
__global__ void histogram(const float f_ValueTable[], unsigned long sizeTable, const unsigned int f_NbEchantillon, unsigned int f_HistoTable[]);

// À partir de l’histogramme, applique la fonction de répartition r(l)
__global__ void repart(const unsigned int f_HistoTable[], unsigned long sizeTable, unsigned int f_RepartionTable[]);

// À partir de la répartition précédente, “étaler” l’histogramme.
__global__ void equalization(const unsigned int f_RepartionTable[], unsigned long sizeTableRepartition, float f_ValueTable[], unsigned long sizeValueTable);

#endif // __HISTOGRAMME_GPU__