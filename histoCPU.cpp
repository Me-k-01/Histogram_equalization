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
    for (int x = 0; x < img->_width; x++) {
        for (int y = 0; y < img->_height; y++) { 
            const size_t i = x + img->_width * y;
            const size_t j = i * img->_nbChannels;
            const float r = img->_pixels[j]/255.f;
            const float g = img->_pixels[j+1]/255.f;
            const float b = img->_pixels[j+2]/255.f; 
            const float colorMax = max(r, g, b);
            const float colorMin = min(r, g, b);
            const float delta = colorMax - colorMin;

            float hueCalc; 
            if (delta == 0)
                hueCalc = 0; // Nuance de gris, donc pas de hue à définir
            else if (colorMax == r)
                hueCalc = (g - b) / delta;
            else if (colorMax == g)
                hueCalc = (b - r) / delta + 2;
            else //if (colMax == b)
                hueCalc = (r - g) / delta + 4;
            // Le Hue correspond à une roue de couleur [0, 6] 
            hueCalc = hueCalc >= 0? std::fmod(hueCalc, 6): std::fmod(hueCalc, 6) + 6;
            //sat[i] = val[i] == 0 ? 0 : (delta/val[i]) * 255.f;
            hue[i] = (unsigned char)(hueCalc * 255.f) / 6.f;  
            val[i] = (unsigned char)colorMax * 255.f;
            sat[i] = colorMax == 0? 0 : (unsigned char)(1.f - colorMin/colorMax) * 255.f; 
        }
    }
} 

void hsv2rgb(unsigned char * hue, unsigned char * sat, unsigned char * val, const Image * const img) {
    for (int x = 0; x < img->_width; x++) {  
        for (int y = 0; y < img->_height; y++) { 
            const size_t i = x + img->_width * y; 
            const size_t j = i*img->_nbChannels;
            const float h = hue[i] * 360.f / 255.f;
            const float s = sat[i]/255.f;
            const float v = sat[i]/255.f;  

 
            const float colorMax = 255 * v; // Chroma
            const float colorMin = colorMax * (1 - s);
            const float colorAdd = (colorMax-colorMin) * (1 - std::abs(
                fmod((h / 60.f), 2) - 1 
            ));
            const float colorInter = colorAdd + colorMin;

            float r; float g; float b;
            if (0 <= h && h < 60) {
                r = colorMax;
                g = colorInter;
                b = colorMin;
            } else if (60 <= h && h < 120) {
                r = colorInter;
                g = colorMax;
                b = colorMin;
            } else if (120 <= h && h < 180) {
                r = colorMin;
                g = colorMax;
                b = colorInter;
            } else if (180 <= h && h < 240) {
                r = colorMin;
                g = colorInter;
                b = colorMax;
            } else if (240 <= h && h < 300) {
                r = colorInter;
                g = colorMin;
                b = colorMax;
            } else if (300 <= h && h < 360) {
                r = colorMax;
                g = colorMin;
                b = colorInter;
            }
 
            img->_pixels[j] = (unsigned char)r;
            img->_pixels[j+1] = (unsigned char)g;
            img->_pixels[j+2] = (unsigned char)b; 
        }
    } 
} 