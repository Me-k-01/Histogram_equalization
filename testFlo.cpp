#include <iostream>
#include <cmath>

float max(const float a, const float b, const float c){
    return std::max(std::max(a, b), c);
} 
float min(const float a, const float b, const float c){
    return std::min(std::min(a, b), c);
} 


void toHSV(unsigned char * rgb, unsigned char * hsvOut) {

    const float r = rgb[0]/255.f;
    const float g = rgb[1]/255.f;
    const float b = rgb[2]/255.f; 
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
 
    hsvOut[0] = (hueCalc * 255.f) / 6.f;  
    hsvOut[2] = colorMax * 255.f;
    hsvOut[1] = colorMax == 0? 0 : (1.f - colorMin/colorMax) * 255.f; 
    //sat[i] = val[i] == 0? 0 : (delta/val[i]) * 255.f;
    //std::cout<< val[i] <<std::endl;
 
}
void toRGB(unsigned char * hsv, unsigned char * rgbOut) { 
    const float h = hsv[0] * 360.f / 255.f;
    const float s = hsv[1]/255.f;
    const float v = hsv[2]/255.f;  

    /*
    const float colorMax = 255 * v;
    const float colorMin = colorMax * (1 - s);
    const float colorAdd = (colorMax-colorMin) * (1 - std::abs(
        (((int)(h / 60.f)) % 2) - 1 
    ));
    */
    const float colorMax = 255 * v; // Chroma
    const float colorMin = colorMax * (1 - s);
    const float colorAdd = (colorMax-colorMin) * (1 - std::abs(
        (((int)(h / 60.f)) % 2) - 1 
    ));
    
    float r; float g; float b;
    if (0 <= h && h < 60) {
        r = colorMax;
        g = colorAdd + colorMin;
        b = colorMin;
    } else if (60 <= h && h < 120) {
        r = colorAdd + colorMin;
        g = colorMax;
        b = colorMin;
    } else if (120 <= h && h < 180) {
        std::cout << "here: " << colorAdd << std::endl;
        r = colorMin;
        g = colorMax;
        b = colorAdd + colorMin;
    } else if (180 <= h && h < 240) {
        r = colorMin;
        g = colorAdd + colorMin;
        b = colorMax;
    } else if (240 <= h && h < 300) {
        r = colorAdd + colorMin;
        g = colorMin;
        b = colorMax;
    } else if (300 <= h && h < 360) {
        r = colorMax;
        g = colorMin;
        b = colorAdd + colorMin;
    }

    rgbOut[0] = r;
    rgbOut[1] = g;
    rgbOut[2] = b; 
}
/*
void toRGB(unsigned char * hsv, unsigned char * rgbOut) { 
    const float h = (hsv[0]/255.f) * 6.f;
    const float s = hsv[1]/255.f;
    const float v = hsv[2]/255.f;  
    const float hDecimal = (h - (int)h);
    //std::cout << hDecimal << std::endl;

    const float colorA = v * (1-s);
    const float colorB = v * (1-hDecimal) * s;
    const float colorC = v * (1-(1-hDecimal)) * s; 


    float r; float g; float b;
    if (0 <= h && h < 1) {
        r = v;
        g = colorC;
        b = colorA;
    } else if (1 <= h && h < 2) {
        r = colorB;
        g = v;
        b = colorA;
    } else if (2 <= h && h < 3) {
        r = colorA;
        g = v;
        b = colorC;
    } else if (3 <= h && h < 4) {
        r = colorA;
        g = colorB;
        b = v;
    } else if (4 <= h && h < 5) {
        r = colorC;
        g = colorA;
        b = v;
    } else if (5 <= h && h < 6) {
        r = v;
        g = colorA;
        b = colorB;
    } 
    rgbOut[0] = r * 255.f;
    rgbOut[1] = g * 255.f;
    rgbOut[2] = b * 255.f; 
}*/

int main() {
    unsigned char rgb[3] = {(unsigned char) 0, (unsigned char) 0, (unsigned char) 0};
    unsigned char hsv[3] ; 
    // TEST 0 0 0
    toHSV(rgb, hsv);
    std::cout << "Pour: " << (int)rgb[0] << " , " << (int)rgb[1] << " , " << (int)rgb[2] << std::endl;
    std::cout << "H: " << (int)hsv[0] << std::endl;
    std::cout << "S: " << (int)hsv[1] << std::endl;
    std::cout << "V: " << (int)hsv[2] << std::endl;

    toRGB(hsv, rgb); 
    std::cout  << std::endl;
    std::cout << "R: " << (int)rgb[0] << std::endl;
    std::cout << "G: " << (int)rgb[1] << std::endl;
    std::cout << "B: " << (int)rgb[2] << std::endl;

    // test Rouge
    rgb[0] = (unsigned char) 255;
    rgb[1] = (unsigned char) 0;
    rgb[2] = (unsigned char) 0;
    toHSV(rgb, hsv);
    std::cout << std::endl << "Pour: " << (int)rgb[0] << " , " << (int)rgb[1] << " , " << (int)rgb[2] << std::endl;
    std::cout << "H: " << (int)hsv[0] << std::endl;
    std::cout << "S: " << (int)hsv[1] << std::endl;
    std::cout << "V: " << (int)hsv[2] << std::endl;
    toRGB(hsv, rgb);
    std::cout  << std::endl;
    std::cout << "R: " << (int)rgb[0] << std::endl;
    std::cout << "G: " << (int)rgb[1] << std::endl;
    std::cout << "B: " << (int)rgb[2] << std::endl;

    // test Vert
    rgb[0] = (unsigned char) 0;
    rgb[1] = (unsigned char) 255;
    rgb[2] = (unsigned char) 0;
    toHSV(rgb, hsv);
    std::cout << std::endl << "Pour: " << (int)rgb[0] << " , " << (int)rgb[1] << " , " << (int)rgb[2] << std::endl;
    std::cout << "H: " << (int)hsv[0] << std::endl;
    std::cout << "S: " << (int)hsv[1] << std::endl;
    std::cout << "V: " << (int)hsv[2] << std::endl;
    toRGB(hsv, rgb);
    std::cout  << std::endl;
    std::cout << "R: " << (int)rgb[0] << std::endl;
    std::cout << "G: " << (int)rgb[1] << std::endl;
    std::cout << "B: " << (int)rgb[2] << std::endl;

    // test Bleu
    rgb[0] = (unsigned char) 0;
    rgb[1] = (unsigned char) 0;
    rgb[2] = (unsigned char) 255;
    toHSV(rgb, hsv);
    std::cout << std::endl << "Pour: " << (int)rgb[0] << " , " << (int)rgb[1] << " , " << (int)rgb[2] << std::endl;
    std::cout << "H: " << (int)hsv[0] << std::endl;
    std::cout << "S: " << (int)hsv[1] << std::endl;
    std::cout << "V: " << (int)hsv[2] << std::endl;
    toRGB(hsv, rgb);
    std::cout  << std::endl;
    std::cout << "R: " << (int)rgb[0] << std::endl;
    std::cout << "G: " << (int)rgb[1] << std::endl;
    std::cout << "B: " << (int)rgb[2] << std::endl;
    

    // test gris foncé
    rgb[0] = (unsigned char) 50;
    rgb[1] = (unsigned char) 50;
    rgb[2] = (unsigned char) 50;
    toHSV(rgb, hsv);
    std::cout << std::endl << "Pour: " << (int)rgb[0] << " , " << (int)rgb[1] << " , " << (int)rgb[2] << std::endl;
    std::cout << "H: " << (int)hsv[0] << std::endl;
    std::cout << "S: " << (int)hsv[1] << std::endl;
    std::cout << "V: " << (int)hsv[2] << std::endl;
    toRGB(hsv, rgb);
    std::cout  << std::endl;
    std::cout << "R: " << (int)rgb[0] << std::endl;
    std::cout << "G: " << (int)rgb[1] << std::endl;
    std::cout << "B: " << (int)rgb[2] << std::endl;

    // test gris foncé
    rgb[0] = (unsigned char) 31;
    rgb[1] = (unsigned char) 170;
    rgb[2] = (unsigned char) 97;
    toHSV(rgb, hsv);
    std::cout << std::endl << "Pour: " << (int)rgb[0] << " , " << (int)rgb[1] << " , " << (int)rgb[2] << std::endl;
    std::cout << "H: " << (int)hsv[0] << std::endl;
    std::cout << "S: " << (int)hsv[1] << std::endl;
    std::cout << "V: " << (int)hsv[2] << std::endl;
    toRGB(hsv, rgb);
    std::cout  << std::endl;
    std::cout << "R: " << (int)rgb[0] << std::endl;
    std::cout << "G: " << (int)rgb[1] << std::endl;
    std::cout << "B: " << (int)rgb[2] << std::endl;
    return 0;
}