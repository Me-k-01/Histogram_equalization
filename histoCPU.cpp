#include "histoCPU.hpp"
#include <iostream>
#include <cmath>

float max(float a, float b, float c){
    return std::max(std::max(a, b), c);
} 
float min(float a, float b, float c){
    return std::min(std::min(a, b), c);
} 

void rgb2hsv(Image * img, unsigned char * hue, unsigned char * sat, unsigned char * val) {
    for (int y = 0; y < img->_height; y++) {
        for (int x = 0; x < img->_width; x++) {
            const size_t i = (x + img->_width * y) * 3;
            float r = img->_pixels[i]/255.f;
            float g = img->_pixels[i+1]/255.f;
            float b = img->_pixels[i+2]/255.f;
            float cmax = max(r, g, b);
            float cmin = min(r, g, b);
            float d = cmax - cmin;

            // arccos en degrés
            float res = acos((r - .5f * g - .5f * b) / sqrt(r*r + g*g + b*b - r*g - r*b - g*b)) * (180.f/M_PI);
            if (g >= b) 
                hue[i] = res;
            else 
                hue[i] = 360.f - res;

            val[i] = cmax;
            sat[i] = cmax == 0? 0 : 1 - cmin/cmax;
        }
    }
    
} 
