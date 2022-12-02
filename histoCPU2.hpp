#ifndef __HISTOGRAMME_CPU__
#define __HISTOGRAMME_CPU__

#include <vector>

#include "image.hpp"

// Version séquentiel de l'algorithme 

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
void rgb2hsv(const Image & f_image, std::vector<float> & f_Htable,  std::vector<float> &  f_Stable,  std::vector<float> &  f_Vtable);
void rgb2hsv(const Image & f_Image, float f_HueTable[],float f_SaturationTable[],float f_ValueTable[]);

// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
void hsv2rgb(const std::vector<float> & f_Htable, const std::vector<float> & f_Stable, const std::vector<float> & f_Vtable, std::vector<unsigned char> & f_pixeltable);
void hsv2rgb(const float f_HueTable[],const float f_SaturationTable[],const float f_ValueTable[], unsigned long sizeTable, unsigned char f_PixelTable[]);


// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
void histogram(const std::vector<float> & f_ValueTable, const unsigned int f_NbEchantillon, std::vector<unsigned int> & f_HistoTable);
void histogram(const float f_ValueTable[], unsigned long sizeTable, const unsigned int f_NbEchantillon, unsigned int f_HistoTable[]);

// À partir de l’histogramme, applique la fonction de répartition r(l)
void repart(const std::vector<unsigned int> & f_HistoTable, std::vector<unsigned int> & f_RepartionTable);
void repart(const unsigned int f_HistoTable[], unsigned long sizeTable, unsigned int f_RepartionTable[]);

// À partir de la répartition précédente, “étaler” l’histogramme.
void equalization(const std::vector<unsigned int> & f_RepartionTable, std::vector<float> & f_ValueTable);
void equalization(const unsigned int f_RepartionTable[], unsigned long sizeTableRepartition, float f_ValueTable[], unsigned long sizeValueTable);

#endif // __HISTOGRAMME_CPU__