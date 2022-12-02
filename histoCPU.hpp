#ifndef __HISTOGRAMME_CPU__
#define __HISTOGRAMME_CPU__
  
#include "image.hpp"

// Version séquentiel de l'algorithme 

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
void rgb2hsv(const Image * const img, unsigned char * hue, unsigned char * sat, unsigned char * val);

// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
void hsv2rgb(unsigned char * hue, unsigned char * sat, unsigned char * val, const Image * const img);

// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
void histogram(unsigned char * imgVal, size_t imgSize, unsigned int * histArray);

// À partir de l’histogramme, applique la fonction de répartition r(l) 
int repart(unsigned int * histArray, size_t l);

// À partir de la répartition précédente, “étaler” l’histogramme.
void equalization(unsigned char * imgVal, size_t imgSize, unsigned int * histArray);

#endif // __HISTOGRAMME_CPU__