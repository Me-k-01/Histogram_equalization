#ifndef __HISTOGRAMME_CPU__
#define __HISTOGRAMME_CPU__
  
// Version séquentiel de l'algorithme 

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
void rgb2hsv();

// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
void hsv2rgb();

// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
void histogram();

// À partir de l’histogramme, applique la fonction de répartition r(l)
void repart();

// À partir de la répartition précédente, “étaler” l’histogramme.
void equalization();

#endif // __HISTOGRAMME_CPU__