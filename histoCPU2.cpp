#include "histoCPU2.hpp"

#include <iostream>

#define PI 3.14159265f

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents

//version found at https://www.rapidtables.com/convert/color/hsv-to-rgb.html
/*
void rgb2hsv(const Image & f_image, std::vector<float> & f_Htable,  std::vector<float> &  f_Stable,  std::vector<float> &  f_Vtable){

    int imagesize = f_image._height*f_image._width*f_image._nbChannels;
    //on s'assure que les vecteur sont vides
    f_Htable.clear();
    f_Stable.clear();
    f_Vtable.clear();

    for (int i = 0; i < imagesize; i+=3)
    {
        int tableindice = i /3;

        float red = (float)f_image._pixels[i]/255.0f;
        float green = (float)f_image._pixels[i+1]/255.0f;
        float blue = (float)f_image._pixels[i+2]/255.0f;

        float colormax = std::max(red,std::max(green,blue));
        float colormin = std::min(red,std::min(green,blue));
        float colordelta = colormax - colormin;

        if(colordelta > 0){
            if(colormax == red){
                f_Htable.push_back(60 * ((green - blue)/colordelta));
            }
            else if (colormax == green)
            {
                f_Htable.push_back(60 * (((blue - red)/colordelta) + 2.0f));
            }
            else{
                f_Htable.push_back(60 * (((red - green)/colordelta) + 4.0f));
            }
            if(f_Htable.at(tableindice) < 0){
                f_Htable.at(tableindice) += 360.0;
            }
        }
        else{
            f_Htable.push_back(0);
        }


        if(colormax > 0){
            f_Stable.push_back(colordelta / colormax);
        }
        else{
            f_Stable.push_back(0);
        }

        f_Vtable.push_back(colormax);

    }
}
*/

//version found at https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html#:~:text=Converting%20RGB%20to%20HSV&text=H%20%3D%20360%20%2D%20cos%2D1,cosine%20is%20calculated%20in%20degrees.
void rgb2hsv(const Image & f_image, std::vector<float> & f_Htable,  std::vector<float> &  f_Stable,  std::vector<float> &  f_Vtable){

    int imagesize = f_image._height*f_image._width*f_image._nbChannels;
    //on s'assure que les vecteur sont vides
    f_Htable.clear();
    f_Stable.clear();
    f_Vtable.clear();

    for (int i = 0; i < imagesize; i+=3)
    {
        float red = (float)f_image._pixels[i];
        float green = (float)f_image._pixels[i+1];
        float blue = (float)f_image._pixels[i+2];

        float colormax = std::max(red, std::max(green, blue));
        float colormin = std::min(red, std::min(green, blue));

        f_Vtable.emplace_back(colormax/255.0f);
        
        if(colormax > 0){
            f_Stable.emplace_back(1.0f-(colormin/colormax));
        }
        else
        {
            f_Stable.emplace_back(0.0f);
        }

        if(colormax - colormin > 0){

            float hue = (std::acos((red - (green/2.0f + blue/2.0f))/std::sqrt(red*red + green*green + blue*blue - (red*green + red*blue + green*blue))))*180/PI;
            if( blue > green){
                f_Htable.emplace_back(360.0f - hue);
            }
            else{
                f_Htable.emplace_back(hue);
            }
        }
        else{
                f_Htable.emplace_back(0.0f);
        }
    }
}




// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
//version found at https://www.rapidtables.com/convert/color/hsv-to-rgb.html
/*
void hsv2rgb(const std::vector<float> & f_Htable, const std::vector<float> & f_Stable, const std::vector<float> & f_Vtable, std::vector<unsigned char> & f_pixeltable){
    int tablesize = f_Htable.size(); 
    f_pixeltable.clear();

    for (int i = 0; i < tablesize; i++)
    {
        float h = f_Htable[i];

        float c = f_Vtable[i] * f_Stable[i];
        float x = c *(1.0f - std::abs((std::fmod((h/60.0f), 2.0f) -1.0f)));
        float m = f_Vtable[i] - c;

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
    
        f_pixeltable.push_back((unsigned char)((red + m) * 255));
        f_pixeltable.push_back((unsigned char)((green + m) * 255));
        f_pixeltable.push_back((unsigned char)((blue + m) * 255));

    }
}
*/

//version found at https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html#:~:text=Converting%20RGB%20to%20HSV&text=H%20%3D%20360%20%2D%20cos%2D1,cosine%20is%20calculated%20in%20degrees.
void hsv2rgb(const std::vector<float> & f_Htable, const std::vector<float> & f_Stable, const std::vector<float> & f_Vtable, std::vector<unsigned char> & f_pixeltable){
    int tablesize = f_Htable.size(); 
    f_pixeltable.clear();

    for (int i = 0; i < tablesize; i++)
    {
        float colormax = 255.0f * f_Vtable.at(i);
        float colormin = colormax*(1.0f-f_Stable.at(i));

        float h = f_Htable.at(i);

        float z = (colormax - colormin)* (1.0f - abs(fmod(h/60.0f,2.0f) -1.0f));

        if(h < 60){
            f_pixeltable.emplace_back((unsigned char)std::round(colormax));
            f_pixeltable.emplace_back((unsigned char)std::round(z + colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(colormin));
        }
        else if (h < 120){
            f_pixeltable.emplace_back((unsigned char)std::round(z + colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(colormax));
            f_pixeltable.emplace_back((unsigned char)std::round(colormin));
        }
        else if (h < 180){
            f_pixeltable.emplace_back((unsigned char)std::round(colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(colormax));
            f_pixeltable.emplace_back((unsigned char)std::round(z + colormin));
        }
        else if (h < 240){
            f_pixeltable.emplace_back((unsigned char)std::round(colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(z + colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(colormax));
        }
        else if (h < 300){
            f_pixeltable.emplace_back((unsigned char)(z + colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(colormax));
        }
        else{
            f_pixeltable.emplace_back((unsigned char)std::round(colormax));
            f_pixeltable.emplace_back((unsigned char)std::round(colormin));
            f_pixeltable.emplace_back((unsigned char)std::round(z + colormin));
        }
        

    }
}


// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
void histogram(){}

// À partir de l’histogramme, applique la fonction de répartition r(l)
void repart(){}

// À partir de la répartition précédente, “étaler” l’histogramme.
void equalization(){}

