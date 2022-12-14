#include "histoCPU.hpp"

#include <iostream>
#include <algorithm>
#include <math.h>

#define NOMINMAX //ligne rendu obligatoire si l'on veut utiliser les fonction min et max de la bibliothèque algorithm car "windows.h" définit deux macros : max et min

#include "../../utils/chronoCPU.hpp"


#define PI 3.14159265f

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents

//version found at https://www.rapidtables.com/convert/color/hsv-to-rgb.html
/*
float rgb2hsv(const Image & f_Image, std::vector<float> & f_HueTable,  std::vector<float> &  f_SaturationTable,  std::vector<float> &  f_ValueTable){

    ChronoCPU timeresult;

    int imagesize = f_Image._height*f_Image._width*f_Image._nbChannels;
    //on s'assure que les vecteur sont vides
    f_HueTable.clear();
    f_SaturationTable.clear();
    f_ValueTable.clear();

    timeresult.start();

    for (int i = 0; i < imagesize; i+=3)
    {
        int tableindice = i /3;

        float red = (float)f_Image._pixels[i]/255.0f;
        float green = (float)f_Image._pixels[i+1]/255.0f;
        float blue = (float)f_Image._pixels[i+2]/255.0f;

        float colormax = std::max(red,std::max(green,blue));
        float colormin = std::min(red,std::min(green,blue));
        float colordelta = colormax - colormin;

        if(colordelta > 0){
            if(colormax == red){
                f_HueTable.push_back(60 * ((green - blue)/colordelta));
            }
            else if (colormax == green)
            {
                f_HueTable.push_back(60 * (((blue - red)/colordelta) + 2.0f));
            }
            else{
                f_HueTable.push_back(60 * (((red - green)/colordelta) + 4.0f));
            }
            if(f_HueTable.at(tableindice) < 0){
                f_HueTable.at(tableindice) += 360.0;
            }
        }
        else{
            f_HueTable.push_back(0);
        }


        if(colormax > 0){
            f_SaturationTable.push_back(colordelta / colormax);
        }
        else{
            f_SaturationTable.push_back(0);
        }

        f_ValueTable.push_back(colormax);

    }
    timeresult.stop();
    return timeresult.elapsedTime();
}
*/

//version found at https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html#:~:text=Converting%20RGB%20to%20HSV&text=H%20%3D%20360%20%2D%20cos%2D1,cosine%20is%20calculated%20in%20degrees.
float rgb2hsv(const Image & f_Image, std::vector<float> & f_HueTable,  std::vector<float> &  f_SaturationTable,  std::vector<float> &  f_ValueTable){

    ChronoCPU timeresult;

    int imagesize = f_Image._height*f_Image._width*f_Image._nbChannels;
    //on s'assure que les vecteur sont vides
    f_HueTable.clear();
    f_SaturationTable.clear();
    f_ValueTable.clear();

    timeresult.start();

    for (int i = 0; i < imagesize; i+=3)
    {
        float red = (float)f_Image._pixels[i];
        float green = (float)f_Image._pixels[i+1];
        float blue = (float)f_Image._pixels[i+2];

        float colormax = std::max(red, std::max(green, blue));
        float colormin = std::min(red, std::min(green, blue));

        f_ValueTable.emplace_back(colormax/255.0f);
        
        if(colormax > 0){
            f_SaturationTable.emplace_back(1.0f-(colormin/colormax));
        }
        else
        {
            f_SaturationTable.emplace_back(0.0f);
        }

        if(colormax - colormin > 0){

            float hue = (std::acos((red - (green/2.0f + blue/2.0f))/std::sqrt(red*red + green*green + blue*blue - (red*green + red*blue + green*blue))))*180/PI;
            if( blue > green){
                f_HueTable.emplace_back(360.0f - hue);
            }
            else{
                f_HueTable.emplace_back(hue);
            }
        }
        else{
                f_HueTable.emplace_back(0.0f);
        }
    }

    timeresult.stop();

    return timeresult.elapsedTime();
}

//version avec des tableaux c pour une meilleur comparaison avec GPU
float rgb2hsv(const Image & f_Image, float f_HueTable[],float f_SaturationTable[],float f_ValueTable[]){
    
    ChronoCPU timeresult;

    int imagesize = f_Image._height*f_Image._width;

    timeresult.start();
    for (int i = 0; i < imagesize; i++)
    {
        float red = (float)f_Image._pixels[i*3];
        float green = (float)f_Image._pixels[i*3+1];
        float blue = (float)f_Image._pixels[i*3+2];

        float colormax = std::max(red, std::max(green, blue));
        float colormin = std::min(red, std::min(green, blue));

        f_ValueTable[i] = colormax/255.0f;
        
        if(colormax > 0){
            f_SaturationTable[i] = 1.0f-(colormin/colormax);
        }
        else
        {
            f_SaturationTable[i] = 0.0f;
        }

        if(colormax - colormin > 0){

            float hue = (std::acos((red - (green/2.0f + blue/2.0f))/std::sqrt(red*red + green*green + blue*blue - (red*green + red*blue + green*blue))))*180/PI;
            if( blue > green){
                f_HueTable[i] = 360.0f - hue;
            }
            else{
                f_HueTable[i] = hue;
            }
        }
        else{
                f_HueTable[i] = 0.0f;
        }
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}


// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
//version found at https://www.rapidtables.com/convert/color/hsv-to-rgb.html
/*
float hsv2rgb(const std::vector<float> & f_HueTable, const std::vector<float> & f_SaturationTable, const std::vector<float> & f_ValueTable, std::vector<unsigned char> & f_PixelTable){
    
    ChronoCPU timeresult;

    int tablesize = f_HueTable.size(); 
    f_PixelTable.clear();

    timeresult.start();

    for (int i = 0; i < tablesize; i++)
    {
        float h = f_HueTable[i];

        float c = f_ValueTable[i] * f_SaturationTable[i];
        float x = c *(1.0f - std::abs((std::fmod((h/60.0f), 2.0f) -1.0f)));
        float m = f_ValueTable[i] - c;

        float red=0.0f, green=0.0f, blue =0.0f;
        
        if(h < 60){
            red = c;
            green = x;
            blue = 0;
        }
        else if (h < 120){
            red = x;
            green = c;
            blue = 0;
        }
        else if (h < 180){
            red = 0;
            green = c;
            blue = x;
        }
        else if (h < 240){
            red = 0;
            green = x;
            blue = c;
        }
        else if (h < 300){
            red = x;
            green = 0;
            blue = c;
        }
        else{
            red = c;
            green = 0;
            blue = x;
        }
    
        f_PixelTable.push_back((unsigned char)((red + m) * 255));
        f_PixelTable.push_back((unsigned char)((green + m) * 255));
        f_PixelTable.push_back((unsigned char)((blue + m) * 255));

    }
    timeresult.stop();
    return timeresult.elapsedTime();
}
*/

//version found at https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html#:~:text=Converting%20RGB%20to%20HSV&text=H%20%3D%20360%20%2D%20cos%2D1,cosine%20is%20calculated%20in%20degrees.
float hsv2rgb(const std::vector<float> & f_HueTable, const std::vector<float> & f_SaturationTable, const std::vector<float> & f_ValueTable, std::vector<unsigned char> & f_PixelTable){
    
    ChronoCPU timeresult;
    int tablesize = f_HueTable.size(); 
    f_PixelTable.clear();

    timeresult.start();

    for (int i = 0; i < tablesize; i++)
    {
        float colormax = 255.0f * f_ValueTable.at(i);
        float colormin = colormax*(1.0f-f_SaturationTable.at(i));

        float h = f_HueTable.at(i);

        float z = (colormax - colormin)* (1.0f - abs(fmod(h/60.0f,2.0f) -1.0f));

        if(h < 60){
            f_PixelTable.emplace_back((unsigned char)std::round(colormax));
            f_PixelTable.emplace_back((unsigned char)std::round(z + colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(colormin));
        }
        else if (h < 120){
            f_PixelTable.emplace_back((unsigned char)std::round(z + colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(colormax));
            f_PixelTable.emplace_back((unsigned char)std::round(colormin));
        }
        else if (h < 180){
            f_PixelTable.emplace_back((unsigned char)std::round(colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(colormax));
            f_PixelTable.emplace_back((unsigned char)std::round(z + colormin));
        }
        else if (h < 240){
            f_PixelTable.emplace_back((unsigned char)std::round(colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(z + colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(colormax));
        }
        else if (h < 300){
            f_PixelTable.emplace_back((unsigned char)(z + colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(colormax));
        }
        else{
            f_PixelTable.emplace_back((unsigned char)std::round(colormax));
            f_PixelTable.emplace_back((unsigned char)std::round(colormin));
            f_PixelTable.emplace_back((unsigned char)std::round(z + colormin));
        }
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}

//version avec des tableaux c pour une meilleur comparaison avec GPU
float hsv2rgb(const float f_HueTable[],const float f_SaturationTable[],const float f_ValueTable[], unsigned int f_sizeTable, unsigned char f_PixelTable[]){

    ChronoCPU timeresult;
    
    timeresult.start();
    for (int i = 0; i < f_sizeTable; i++)
    {
        float colormax = 255.0f * f_ValueTable[i];
        float colormin = colormax*(1.0f-f_SaturationTable[i]);

        float h = f_HueTable[i];

        float z = (colormax - colormin)* (1.0f - abs(fmod(h/60.0f,2.0f) -1.0f));

        if(h < 60){
            f_PixelTable[i*3] = (unsigned char)std::round(colormax);
            f_PixelTable[i*3+1] = ((unsigned char)std::round(z + colormin));
            f_PixelTable[i*3+2] = ((unsigned char)std::round(colormin));
        }
        else if (h < 120){
            f_PixelTable[i*3] = ((unsigned char)std::round(z + colormin));
            f_PixelTable[i*3+1] = ((unsigned char)std::round(colormax));
            f_PixelTable[i*3+2] = ((unsigned char)std::round(colormin));
        }
        else if (h < 180){
            f_PixelTable[i*3] = ((unsigned char)std::round(colormin));
            f_PixelTable[i*3+1] = ((unsigned char)std::round(colormax));
            f_PixelTable[i*3+2] = ((unsigned char)std::round(z + colormin));
        }
        else if (h < 240){
            f_PixelTable[i*3] = ((unsigned char)std::round(colormin));
            f_PixelTable[i*3+1] = ((unsigned char)std::round(z + colormin));
            f_PixelTable[i*3+2] = ((unsigned char)std::round(colormax));
        }
        else if (h < 300){
            f_PixelTable[i*3] = ((unsigned char)std::round(z + colormin));
            f_PixelTable[i*3+1] = ((unsigned char)std::round(colormin));
            f_PixelTable[i*3+2] = ((unsigned char)std::round(colormax));
        }
        else{
            f_PixelTable[i*3] = ((unsigned char)std::round(colormax));
            f_PixelTable[i*3+1] = ((unsigned char)std::round(colormin));
            f_PixelTable[i*3+2] = ((unsigned char)std::round(z + colormin));
        }
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}


// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
float histogram(const std::vector<float> & f_ValueTable, const unsigned int f_NbEchantillon, std::vector<unsigned int> & f_HistoTable){
    
    ChronoCPU timeresult;
    f_HistoTable.clear();
    f_HistoTable.resize(f_NbEchantillon);
    std::fill(f_HistoTable.begin(), f_HistoTable.end(), 0); //rempli l'entièreté du vecteur Histogramme avec des 0

    timeresult.start();

    for (unsigned int i = 0; i < f_ValueTable.size(); i++)
    {
        unsigned int indiceHisto = (unsigned int)std::round(f_ValueTable.at(i)*(f_NbEchantillon-1));
        f_HistoTable.at(indiceHisto) ++;
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}

//version avec des tableaux c pour une meilleur comparaison avec GPU
float histogram(const float f_ValueTable[], unsigned int f_sizeTable, const unsigned int f_NbEchantillon, unsigned int f_HistoTable[]){
    
    ChronoCPU timeresult;

    timeresult.start();
    for (unsigned int i = 0; i < f_NbEchantillon; i++){
        f_HistoTable[i] = 0;
    }

    for (unsigned int i = 0; i < f_sizeTable; i++)
    {
        unsigned int indiceHisto = std::min((unsigned int)std::max((int)std::round(f_ValueTable[i]*(f_NbEchantillon-1)),0),f_NbEchantillon-1);
        f_HistoTable[indiceHisto] ++;
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}

// À partir de l’histogramme, applique la fonction de répartition r(l)
float repart(const std::vector<unsigned int> & f_HistoTable, std::vector<unsigned int> & f_RepartionTable){
    
    ChronoCPU timeresult;
    f_RepartionTable.clear();
    f_RepartionTable.resize(f_HistoTable.size());

    timeresult.start();

    f_RepartionTable.at(0) = f_HistoTable.at(0);
    for (size_t i = 1; i < f_HistoTable.size(); i++)
    {
        f_RepartionTable.at(i) = f_RepartionTable.at(i-1) + f_HistoTable.at(i);
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}

//version avec des tableaux c pour une meilleur comparaison avec GPU
float repart(const unsigned int f_HistoTable[], unsigned int f_sizeTable, unsigned int f_RepartionTable[]){
    
    ChronoCPU timeresult;
    timeresult.start();

    f_RepartionTable[0] = f_HistoTable[0];
    for (size_t i = 1; i < f_sizeTable; i++)
    {
        f_RepartionTable[i] = f_RepartionTable[i-1] + f_HistoTable[i];
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}

// À partir de la répartition précédente, “étaler” l’histogramme.
float equalization(const std::vector<unsigned int> & f_RepartionTable,  std::vector<float> & f_ValueTable){
    
    ChronoCPU timeresult;
    unsigned int imageSize = f_ValueTable.size();
    unsigned int NbEchantillon = f_RepartionTable.size();

    timeresult.start();
    float coefficient = ((float)NbEchantillon - 1.f)/(((float)NbEchantillon)*imageSize);

    for (size_t i = 0; i < imageSize; i++)
    {
        unsigned int indiceRepartitionTable = std::round(f_ValueTable[i]*(NbEchantillon-1));
        f_ValueTable.at(i) = coefficient * (float)f_RepartionTable.at(indiceRepartitionTable); 
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}

//version avec des tableaux c pour une meilleur comparaison avec GPU
float equalization(const unsigned int f_RepartionTable[], unsigned int f_sizeTableRepartition, float f_ValueTable[], unsigned int f_sizeValueTable){
    
    ChronoCPU timeresult;
    timeresult.start();

    float coefficient = ((float)f_sizeTableRepartition - 1.f)/(((float)f_sizeTableRepartition)*f_sizeValueTable);

    f_sizeTableRepartition -= 1 ; //évitera la redondance dans la boucle
    
    for (size_t i = 0; i < f_sizeValueTable; i++)
    {
        unsigned int indiceRepartitionTable = std::round(f_ValueTable[i]*(f_sizeTableRepartition));
        f_ValueTable[i] = coefficient * (float)f_RepartionTable[indiceRepartitionTable]; 
    }
    timeresult.stop();
    return timeresult.elapsedTime();
}