#include "histoGPU.hpp"
#include "../../utils/commonCUDA.hpp"

#define PI 3.14159265f

// fonction d'appel au fonction gpu
void gpuCall(const Image & f_ImageIn, int nbEchantillon, Image & f_ImageOut){
    dim3 bloc(32,32,1);
    dim3 grille((f_ImageIn._width + bloc.x-1)/bloc.x,(f_ImageIn._height + bloc.y-1)/bloc.y,1);


    //création des pointeurs pour gpu
    unsigned long sizeImage = f_ImageIn._width * f_ImageIn._height;
    float *hueTable, *saturationTable, *valueTable;
    unsigned char * pixelTableIn, *pixelTableOut;
    unsigned int * histoTable, *repartTable;

    HANDLE_ERROR(cudaMalloc((void**)&pixelTableIn, sizeImage*f_ImageIn._nbChannels*sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc((void**)&pixelTableOut, sizeImage*f_ImageIn._nbChannels*sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc((void**)&hueTable, sizeImage*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&saturationTable, sizeImage*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&valueTable, sizeImage*sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(pixelTableOut, f_ImageIn._pixels, sizeImage*f_ImageIn._nbChannels*sizeof(unsigned char),cudaMemcpyHostToDevice));

    rgb2hsv<<<grille,bloc>>>(pixelTableIn,sizeImage,hueTable,saturationTable,valueTable);
    hsv2rgb<<<grille,bloc>>>(hueTable,saturationTable,valueTable, sizeImage, pixelTableOut);

    //définition de l'image de sortie
    f_ImageOut._width = f_ImageIn._width;
    f_ImageOut._height = f_ImageIn._height;
    f_ImageOut._nbChannels = f_ImageIn._nbChannels;
    f_ImageOut._pixels = (unsigned char*)malloc(sizeImage*f_ImageIn._nbChannels*sizeof(unsigned char)) ;
    HANDLE_ERROR(cudaMemcpy(f_ImageOut._pixels, pixelTableOut, sizeImage*f_ImageIn._nbChannels*sizeof(unsigned char),cudaMemcpyDeviceToHost));


    HANDLE_ERROR(cudaFree(pixelTableIn));
    HANDLE_ERROR(cudaFree(pixelTableOut));
    HANDLE_ERROR(cudaFree(hueTable));
    HANDLE_ERROR(cudaFree(saturationTable));
    HANDLE_ERROR(cudaFree(valueTable));

}

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
__global__ void rgb2hsv(const unsigned char f_PixelTable[], unsigned long f_sizeTable, float f_HueTable[],float f_SaturationTable[],float f_ValueTable[]){
    int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    int tidglobal = tidx + tidy *blockDim.x*gridDim.x;
    int nbThreadTotal = blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    while (tidglobal < f_sizeTable)
    {
        float red = (float)f_PixelTable[tidglobal*3];
        float green = (float)f_PixelTable[tidglobal*3+1];
        float blue = (float)f_PixelTable[tidglobal*3+2];

        float colormax = fmaxf(red, fmaxf(green, blue));
        float colormin = fminf(red, fminf(green, blue));

        f_ValueTable[tidglobal] = colormax/255.0f;
        
        if(colormax > 0){
            f_SaturationTable[tidglobal] = 1.0f-(colormin/colormax);
        }
        else
        {
            f_SaturationTable[tidglobal] = 0.0f;
        }

        if(colormax - colormin > 0){

            float hue = (acosf((red - (green/2.0f + blue/2.0f))/sqrtf(red*red + green*green + blue*blue - (red*green + red*blue + green*blue))))*180/PI;
            if( blue > green){
                f_HueTable[tidglobal] = 360.0f - hue;
            }
            else{
                f_HueTable[tidglobal] = hue;
            }
        }
        else{
                f_HueTable[tidglobal] = 0.0f;
        }
        tidglobal += nbThreadTotal;
    }
    
}

// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
__global__ void hsv2rgb(const float f_HueTable[],const float f_SaturationTable[],const float f_ValueTable[], unsigned long f_sizeTable, unsigned char f_PixelTable[]){

    int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    int tidy = threadIdx.y + blockIdx.y*blockDim.y;
    int tidglobal = tidx + tidy *blockDim.x*gridDim.x;
    int nbThreadTotal = blockDim.x*gridDim.x*blockDim.y*gridDim.y;


    while (tidglobal < f_sizeTable)
    {
        float colormax = 255.0f * f_ValueTable[tidglobal];
        float colormin = colormax*(1.0f-f_SaturationTable[tidglobal]);

        float h = f_HueTable[tidglobal];

        float z = (colormax - colormin)* (1.0f - fabsf(fmodf(h/60.0f,2.0f) -1.0f));

        if(h < 60){
            f_PixelTable[tidglobal*3] = (unsigned char)roundf(colormax);
            f_PixelTable[tidglobal*3+1] = ((unsigned char)roundf(z + colormin));
            f_PixelTable[tidglobal*3+2] = ((unsigned char)roundf(colormin));
        }
        else if (h < 120){
            f_PixelTable[tidglobal*3] = ((unsigned char)roundf(z + colormin));
            f_PixelTable[tidglobal*3+1] = ((unsigned char)roundf(colormax));
            f_PixelTable[tidglobal*3+2] = ((unsigned char)roundf(colormin));
        }
        else if (h < 180){
            f_PixelTable[tidglobal*3] = ((unsigned char)roundf(colormin));
            f_PixelTable[tidglobal*3+1] = ((unsigned char)roundf(colormax));
            f_PixelTable[tidglobal*3+2] = ((unsigned char)roundf(z + colormin));
        }
        else if (h < 240){
            f_PixelTable[tidglobal*3] = ((unsigned char)roundf(colormin));
            f_PixelTable[tidglobal*3+1] = ((unsigned char)roundf(z + colormin));
            f_PixelTable[tidglobal*3+2] = ((unsigned char)roundf(colormax));
        }
        else if (h < 300){
            f_PixelTable[tidglobal*3] = ((unsigned char)roundf(z + colormin));
            f_PixelTable[tidglobal*3+1] = ((unsigned char)roundf(colormin));
            f_PixelTable[tidglobal*3+2] = ((unsigned char)roundf(colormax));
        }
        else{
            f_PixelTable[tidglobal*3] = ((unsigned char)roundf(colormax));
            f_PixelTable[tidglobal*3+1] = ((unsigned char)roundf(colormin));
            f_PixelTable[tidglobal*3+2] = ((unsigned char)roundf(z + colormin));
        }
        tidglobal+=nbThreadTotal;
    }

}

// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
__global__ void histogram(const float f_ValueTable[], unsigned long sizeTable, const unsigned int f_NbEchantillon, unsigned int f_HistoTable[]){

}

// À partir de l’histogramme, applique la fonction de répartition r(l)
__global__ void repart(const unsigned int f_HistoTable[], unsigned long sizeTable, unsigned int f_RepartionTable[]){

}

// À partir de la répartition précédente, “étaler” l’histogramme.
__global__ void equalization(const unsigned int f_RepartionTable[], unsigned long sizeTableRepartition, float f_ValueTable[], unsigned long sizeValueTable){

}
