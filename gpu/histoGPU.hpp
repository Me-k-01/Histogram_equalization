#ifndef __HISTOGRAMME_GPU__
#define __HISTOGRAMME_GPU__

#include <vector>

#include "../utils/image.hpp"

#define HISTO_SIZE 256

enum kernelToTest {
    RGB2HSV,
    RGB2HSV_MINIMUMDIVERGENCE,
    RGB2HSV_COORDINATEDOUTPUTS,
    HISTOGRAM,
    HISTOGRAM_WITHSHAREDMEMORY,
    HISTOGRAM_WITHSHAREDMEMORYANDHARCODEDSIZE,
    HISTOGRAM_WITHMINIMUMCALCULATIONDEPENCIES,
    REPART,
    REPART_WITHSHAREDMEMORY,
    REPART_WITHSHAREDMEMORYANDHARCODEDSIZE,
    EQUALIZATION,
    EQUALIZATION_CONSTANTCOEFFICIENT,
    HSV2RGB
};

// variables et constantes utilisable dans les fonctions __global__ et __device__
__constant__ float deviceCoefficientEqualization = 1.0f;

// fonction d'appel au fonction gpu
void gpuCall(Image & f_ImageIn, const int f_nbEchantillon);

// fonction d'appel aux fonctions gpu pour benchmark
void gpuCallBenchmark(Image & f_Image, const int f_nbEchantillon, dim3 f_bloc, dim3 f_grille, kernelToTest f_kernel);


// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
__global__ void rgb2hsv(const unsigned char f_PixelTable[], const unsigned int f_sizeTable, float f_HueTable[], float f_SaturationTable[], float f_ValueTable[]);
// amélioration qui evite les branches en utilisant le résultat des test logique directement dans la formule (sous forme d'entier) 
__global__ void rgb2hsvWithMinimumDivergence(const unsigned char f_PixelTable[], const unsigned int f_sizeTable, float f_HueTable[], float f_SaturationTable[], float f_ValueTable[]);
// amélioration pour faire les accès mémoires en simultanées  
__global__ void rgb2hsvWithCoordinatedOutputs(const unsigned char f_PixelTable[], const unsigned int f_sizeTable, float f_HueTable[], float f_SaturationTable[], float f_ValueTable[]);


// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
__global__ void histogram(const float f_ValueTable[], unsigned int f_sizeTable, const unsigned int f_NbEchantillon, unsigned int f_HistoTable[]);
// amélioration avec des histogrammes partiels intermédiaire
__global__ void histogramWithSharedMemory(const float f_ValueTable[], unsigned int f_sizeTable, const unsigned int f_nbEchantillon, unsigned int f_HistoTable[]);
// amélioration avec des histogrammes partiels intermédiaire mais la taille est hardcodée
__global__ void histogramWithSharedMemoryAndHarcodedsSize(const float f_ValueTable[], unsigned int f_sizeTable, unsigned int f_HistoTable[]);
// amélioration qui enlève au maximum les dépendances de calcules
__global__ void histogramWithMinimumDependencies(const float f_ValueTable[], unsigned int f_sizeTable, const unsigned int f_nbEchantillon, unsigned int f_HistoTable[]);

// À partir de l’histogramme, applique la fonction de répartition r(l)
__global__ void repart(const unsigned int f_HistoTable[], const unsigned int f_sizeTable, unsigned int f_RepartionTable[]);
// répartition avec l'utilisation de la shared memory pour le tableau histogramme
__global__ void repartWithSharedMemory(const unsigned int f_HistoTable[], const unsigned int f_sizeTable, unsigned int f_RepartionTable[]);
// répartition avec l'utilisation de la shared memory pour le tableau histogramme mais la taille est hardcodée
__global__ void repartWithSharedMemoryAndHarcodedsSize(const unsigned int f_HistoTable[], unsigned int f_RepartionTable[]); 
 

// À partir de la répartition précédente, “étaler” l’histogramme.
__global__ void equalization(const unsigned int f_RepartionTable[], const unsigned int f_sizeTableRepartition, float f_ValueTable[], const unsigned int sizeValueTable);
// Version de la fonction "equalization" sans calcule du coeficiant de répartition car il est passé en mémoire constante.
__global__ void equalizationWithConstantCoefficient(const unsigned int f_RepartionTable[], const unsigned int f_sizeTableRepartition, float f_ValueTable[], const unsigned int sizeValueTable);


// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
__global__ void hsv2rgb(const float f_HueTable[], const float f_SaturationTable[],const float f_ValueTable[], const unsigned int f_sizeTable, unsigned char f_PixelTable[]);


#endif // __HISTOGRAMME_GPU__