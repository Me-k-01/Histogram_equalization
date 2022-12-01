#include "histoCPU.hpp"
#include <iostream>
#include <cmath>

float max(const float a, const float b, const float c){
    return std::max(std::max(a, b), c);
} 
float min(const float a, const float b, const float c){
    return std::min(std::min(a, b), c);
} 

void rgb2hsv(const Image * const img, unsigned char * hue, unsigned char * sat, unsigned char * val) {
    for (int y = 0; y < img->_height; y++) {
        for (int x = 0; x < img->_width; x++) {
            const size_t i = (x + img->_width * y) * img->_nbChannels;
            const float r = img->_pixels[i]/255.f;
            const float g = img->_pixels[i+1]/255.f;
            const float b = img->_pixels[i+2]/255.f; 
            const float colMax = max(r, g, b);
            const float colMin = min(r, g, b);
            const float delta = colMax - colMin;

            // arccos en degrés
            //float hueCalc = acos((r - .5f * g - .5f * b) / sqrt(r*r + g*g + b*b - r*g - r*b - g*b)) * (180.f/M_PI);
            //std::cout<<hueCalc<<std::endl;

            float hueCalc; 
            if (delta == 0)
                hueCalc = 0; // Nuance de gris, donc pas de hue à définir
            else if (colMax == r)
                hueCalc = (g - b) / delta;
            else if (colMax == g)
                hueCalc = (b - r) / delta + 2;
            else //if (colMax == b)
                hueCalc = (r - g) / delta + 4;
            // Le Hue correspond à une roue de couleur [0, 6] 
            hueCalc = hueCalc > 0? std::fmod(hueCalc, 6): std::fmod(hueCalc, 6) + 6;

            hue[i] = hueCalc * 255.f / 6.f;  
            val[i] = colMax * 255.f;
            sat[i] = colMax == 0? 0 : (1.f - colMin/colMax) * 255.f; 
            //sat[i] = val[i] == 0? 0 : (delta/val[i]) * 255.f;
            //std::cout<< sat[i] <<std::endl;
        }
    }
} 

void hsv2rgb(unsigned char * hue, unsigned char * sat, unsigned char * val, const Image * const img) {
    for (int y = 0; y < img->_height; y++) {
        for (int x = 0; x < img->_width; x++) { 
            const size_t i = x + img->_width * y; 
            const float h = hue[i]/255.f;
            const float s = sat[i]/255.f;
            const float v = val[i]/255.f; 
            //std::cout << s;
            const float hDecimal = (h - (int)h);

            const float alpha = v * (1-s);
            const float beta = v * (1-hDecimal) * s;
            const float gamma = v * (1-(1-hDecimal)) * s;

 
            float r; float g; float b;
            if (0 <= h && h < 1) {
                r = v;
                g = gamma;
                b = alpha;
            } else if (1 <= h && h < 2) {
                r = beta;
                g = v;
                b = alpha;
            } else if (2 <= h && h < 3) {
                r = alpha;
                g = v;
                b = gamma;
            } else if (3 <= h && h < 4) {
                r = alpha;
                g = beta;
                b = v;
            } else if (4 <= h && h < 5) {
                r = gamma;
                g = alpha;
                b = v;
            } else if (5 <= h && h < 6) {
                r = v;
                g = alpha;
                b = beta;
            }
            const size_t j = i*3;
            img->_pixels[j] = r * 255.f;
            img->_pixels[j+1] = g * 255.f;
            img->_pixels[j+2] = b * 255.f; 
        }
    } 
} 