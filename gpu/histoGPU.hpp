#ifndef __HISTOGRAMME_GPU__
#define __HISTOGRAMME_GPU__

#include <vector>

#include "../image.hpp"

enum kernelToTest {
    RGB2HSV,
    HISTOGRAM,
    REPART,
    EQUALIZATION,
    HSV2RGB
};


// fonction d'appel au fonction gpu
void gpuCall(Image & f_ImageIn, const int f_nbEchantillon);

// fonction d'appel au fonction gpu pour tests
void gpuCallTest(Image & f_Image, const int f_nbEchantillon, dim3 f_bloc, dim3 f_grille, kernelToTest f_kernel);

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
__global__ void rgb2hsv(const unsigned char f_PixelTable[], const unsigned int f_sizeTable, float f_HueTable[], float f_SaturationTable[], float f_ValueTable[]);

// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
__global__ void hsv2rgb(const float f_HueTable[], const float f_SaturationTable[],const float f_ValueTable[], const unsigned int f_sizeTable, unsigned char f_PixelTable[]);


// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
__global__ void histogram(const float f_ValueTable[], unsigned int f_sizeTable, const unsigned int f_NbEchantillon, unsigned int f_HistoTable[]);

// À partir de l’histogramme, applique la fonction de répartition r(l)
__global__ void repart(const unsigned int f_HistoTable[], const unsigned int f_sizeTable, unsigned int f_RepartionTable[]);

// À partir de la répartition précédente, “étaler” l’histogramme.
__global__ void equalization(const unsigned int f_RepartionTable[], const unsigned int f_sizeTableRepartition, float f_ValueTable[], const unsigned int sizeValueTable);


#endif // __HISTOGRAMME_GPU__