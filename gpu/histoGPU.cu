#include "histoGPU.hpp"
#include "../utils/commonCUDA.hpp"
#include <iostream>


#define PI 3.14159265f

// fonction d'appel au fonction gpu
void gpuCall(Image & f_Image, const int f_nbEchantillon){

    // Tailles 
    unsigned int sizeImage = f_Image._width * f_Image._height;
    unsigned int sizeTableInBytes = sizeImage * sizeof(float);
    unsigned int sizeImageInBytes = sizeImage * sizeof(unsigned char) * f_Image._nbChannels; 
    // Création des pointeurs pour gpu
    float *hueTable, *saturationTable, *valueTable;
    unsigned char * pixelTableIn, *pixelTableOut;
    unsigned int * histoTable, *repartTable;

    HANDLE_ERROR(cudaMalloc((void**)&pixelTableIn, sizeImageInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&pixelTableOut, sizeImageInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&hueTable, sizeTableInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&saturationTable, sizeTableInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&valueTable, sizeTableInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&histoTable, f_nbEchantillon*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void**)&repartTable, f_nbEchantillon*sizeof(unsigned int)));

    HANDLE_ERROR(cudaMemcpy(pixelTableIn, f_Image._pixels, sizeImageInBytes, cudaMemcpyHostToDevice));

    //définition des bloc et grille selon les différents Kernel
    dim3 blocRGB2HSV(32,32,1);
    dim3 grilleRGB2HSV((f_Image._width + blocRGB2HSV.x-1)/blocRGB2HSV.x,(f_Image._height + blocRGB2HSV.y-1)/blocRGB2HSV.y,1);
    
    dim3 blocHistogramme(32,1,1);
    dim3 grilleHistogramme(1,1,1);
    
    dim3 blocRepart(32,1,1);
    dim3 grilleRepart(1,1,1);
    
    dim3 blocEqualization(32,1,1);
    dim3 grilleEqualization(1,1,1);
    
    dim3 blocHSV2RGB(32,32,1);
    dim3 grilleHSV2RGB((f_Image._width + blocHSV2RGB.x-1)/blocHSV2RGB.x,(f_Image._height + blocHSV2RGB.y-1)/blocHSV2RGB.y,1);


    rgb2hsv<<<blocRGB2HSV, grilleRGB2HSV>>>(pixelTableIn, sizeImage, hueTable, saturationTable, valueTable);
    histogram<<<blocHistogramme, grilleHistogramme>>>(valueTable, sizeImage, f_nbEchantillon, histoTable);
    repart<<<blocRepart, grilleRepart>>>(histoTable, f_nbEchantillon, repartTable);
    equalization<<<blocEqualization,grilleEqualization>>>(repartTable, f_nbEchantillon, valueTable, sizeImage);

    hsv2rgb<<<blocHSV2RGB,grilleHSV2RGB>>>(hueTable,saturationTable,valueTable, sizeImage, pixelTableOut);

    HANDLE_ERROR(cudaMemcpy(f_Image._pixels, pixelTableOut, sizeImageInBytes,cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(pixelTableIn));
    HANDLE_ERROR(cudaFree(pixelTableOut));
    HANDLE_ERROR(cudaFree(hueTable));
    HANDLE_ERROR(cudaFree(saturationTable));
    HANDLE_ERROR(cudaFree(valueTable));
    HANDLE_ERROR(cudaFree(histoTable));
    HANDLE_ERROR(cudaFree(repartTable));
}


// fonction d'appel au fonction gpu pour tests
void gpuCallTest(Image & f_Image, const int f_nbEchantillon, dim3 f_bloc, dim3 f_grille, kernelToTest f_kernel){

    // Tailles 
    unsigned int sizeImage = f_Image._width * f_Image._height;
    unsigned int sizeTableInBytes = sizeImage * sizeof(float);
    unsigned int sizeImageInBytes = sizeImage * sizeof(unsigned char) * f_Image._nbChannels; 
    // Création des pointeurs pour gpu
    float *hueTable, *saturationTable, *valueTable;
    unsigned char * pixelTableIn, *pixelTableOut;
    unsigned int * histoTable, *repartTable;
    

    HANDLE_ERROR(cudaMalloc((void**)&pixelTableIn, sizeImageInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&pixelTableOut, sizeImageInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&hueTable, sizeTableInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&saturationTable, sizeTableInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&valueTable, sizeTableInBytes));
    HANDLE_ERROR(cudaMalloc((void**)&histoTable, f_nbEchantillon*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void**)&repartTable, f_nbEchantillon*sizeof(unsigned int)));

    HANDLE_ERROR(cudaMemcpy(pixelTableIn, f_Image._pixels, sizeImageInBytes, cudaMemcpyHostToDevice));
    
    dim3 blocRGB2HSV(32,32,1);
    dim3 grilleRGB2HSV(1,1,1);
    dim3 blocHistogramme(32,1,1);
    dim3 grilleHistogramme(1,1,1);
    dim3 blocRepart(32,1,1);
    dim3 grilleRepart(1,1,1);
    dim3 blocEqualization(32,1,1);
    dim3 grilleEqualization(1,1,1);

    switch (f_kernel) {
        case kernelToTest::RGB2HSV : 
            rgb2hsv<<<f_bloc, f_grille>>>(pixelTableIn, sizeImage, hueTable, saturationTable, valueTable);
            break;
            
        case kernelToTest::HISTOGRAM :
            rgb2hsv<<<blocRGB2HSV, grilleRGB2HSV>>>(pixelTableIn, sizeImage, hueTable, saturationTable, valueTable);
            histogram<<<f_bloc, f_grille>>>(valueTable, sizeImage, f_nbEchantillon, histoTable);
            break;
            
        case kernelToTest::REPART :
            rgb2hsv<<<blocRGB2HSV, grilleRGB2HSV>>>(pixelTableIn, sizeImage, hueTable, saturationTable, valueTable);
            histogram<<<blocHistogramme, grilleHistogramme>>>(valueTable, sizeImage, f_nbEchantillon, histoTable);
            repart<<<f_bloc, f_grille>>>(histoTable, f_nbEchantillon, repartTable);
            break;

        case kernelToTest::EQUALIZATION :
            rgb2hsv<<<blocRGB2HSV, grilleRGB2HSV>>>(pixelTableIn, sizeImage, hueTable, saturationTable, valueTable);
            histogram<<<blocHistogramme, grilleHistogramme>>>(valueTable, sizeImage, f_nbEchantillon, histoTable);
            repart<<<blocRepart, grilleRepart>>>(histoTable, f_nbEchantillon, repartTable);
            equalization<<<f_bloc, f_grille>>>(repartTable, f_nbEchantillon, valueTable, sizeImage);
            break;        

        case kernelToTest::HSV2RGB : 
            rgb2hsv<<<blocRGB2HSV, grilleRGB2HSV>>>(pixelTableIn, sizeImage, hueTable, saturationTable, valueTable);
            histogram<<<blocHistogramme, grilleHistogramme>>>(valueTable, sizeImage, f_nbEchantillon, histoTable);
            repart<<<blocRepart, grilleRepart>>>(histoTable, f_nbEchantillon, repartTable);
            equalization<<<blocEqualization, grilleEqualization>>>(repartTable, f_nbEchantillon, valueTable, sizeImage);
            hsv2rgb<<<f_bloc,f_grille>>>(hueTable,saturationTable,valueTable, sizeImage, pixelTableOut);
            break;
    }
    HANDLE_ERROR(cudaFree(pixelTableIn));
    HANDLE_ERROR(cudaFree(pixelTableOut));
    HANDLE_ERROR(cudaFree(hueTable));
    HANDLE_ERROR(cudaFree(saturationTable));
    HANDLE_ERROR(cudaFree(valueTable));
    HANDLE_ERROR(cudaFree(histoTable));
    HANDLE_ERROR(cudaFree(repartTable));
}

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
__global__ void rgb2hsv(const unsigned char f_PixelTable[], const unsigned int f_sizeTable, float f_HueTable[], float f_SaturationTable[], float f_ValueTable[]){
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
    const int nbThreadTotal = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tidGlobal = tidx + tidy * blockDim.x * gridDim.x;

    while (tidGlobal < f_sizeTable)
    {
        const float red   = (float)f_PixelTable[tidGlobal*3];
        const float green = (float)f_PixelTable[tidGlobal*3 + 1];
        const float blue  = (float)f_PixelTable[tidGlobal*3 + 2];

        const float colorMax = fmaxf(red, fmaxf(green, blue));
        const float colorMin = fminf(red, fminf(green, blue));

        f_ValueTable[tidGlobal]      = colorMax / 255.f;
        f_SaturationTable[tidGlobal] = colorMax > 0 ? 1.f - colorMin/colorMax : 0.f;
        
        if (colorMax - colorMin > 0) {
            float hue = acosf( 
                (red - (green / 2.f + blue/2.f)) / sqrtf(red*red + green*green + blue*blue - (red*green + red*blue + green*blue))
            ) * 180.f / PI;

            f_HueTable[tidGlobal] = blue > green ? 360.f - hue : hue;
        } else {
            f_HueTable[tidGlobal] = 0.f;
        }

        tidGlobal += nbThreadTotal;
    }
    
}

// Transformation de HSV vers RGB (donc de trois tableaux vers un seul).
__global__ void hsv2rgb(const float f_HueTable[], const float f_SaturationTable[], const float f_ValueTable[], const unsigned int f_sizeTable, unsigned char f_PixelTable[]) {

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
    const int nbThreadTotal = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tidGlobal = tidx + tidy * blockDim.x * gridDim.x;

    while (tidGlobal < f_sizeTable) {
        const float cMax = 255.f * f_ValueTable[tidGlobal];
        const float cMin = cMax  * (1.f - f_SaturationTable[tidGlobal]);
        const float hue  = f_HueTable[tidGlobal];
        const float cAdd = (cMax - cMin) * (1.f - fabsf(fmodf(hue / 60.f, 2.f) - 1.f));
        const unsigned char colorMax   = roundf(cMax);
        const unsigned char colorMin   = roundf(cMin);
        const unsigned char colorInter = roundf(cAdd + cMin);

        const int pixelIndex = tidGlobal * 3; 
        if (hue < 60) {
            f_PixelTable[pixelIndex]     = colorMax;
            f_PixelTable[pixelIndex + 1] = colorInter;
            f_PixelTable[pixelIndex + 2] = colorMin;
        } else if (hue < 120) {
            f_PixelTable[pixelIndex]     = colorInter;
            f_PixelTable[pixelIndex + 1] = colorMax;
            f_PixelTable[pixelIndex + 2] = colorMin;
        } else if (hue < 180) {
            f_PixelTable[pixelIndex]     = colorMin;
            f_PixelTable[pixelIndex + 1] = colorMax;
            f_PixelTable[pixelIndex + 2] = colorInter;
        } else if (hue < 240) {
            f_PixelTable[pixelIndex]     = colorMin;
            f_PixelTable[pixelIndex + 1] = colorInter;
            f_PixelTable[pixelIndex + 2] = colorMax;
        } else if (hue < 300) {
            f_PixelTable[pixelIndex]     = colorInter;
            f_PixelTable[pixelIndex + 1] = colorMin;
            f_PixelTable[pixelIndex + 2] = colorMax;
        } else { // si (hue < 360)
            f_PixelTable[pixelIndex]     = colorMax;
            f_PixelTable[pixelIndex + 1] = colorMin;
            f_PixelTable[pixelIndex + 2] = colorInter;
        }
        
        tidGlobal += nbThreadTotal;
    }

}

// Fonction qui à partir de la composante V de chaque pixel, calcule l’histogramme de l’image.
__global__ void histogram(const float f_ValueTable[], unsigned int f_sizeTable, const unsigned int f_f_nbEchantillon, unsigned int f_HistoTable[]) {
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
    const int nbThreadTotal = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tidGlobal = tidx + tidy * blockDim.x * gridDim.x;

	for (; tidGlobal < f_sizeTable; tidGlobal += nbThreadTotal) { 
        int indexHist = roundf(f_ValueTable[tidGlobal] * f_f_nbEchantillon);
        // On doit attendre que les threads ont terminer d'écrire sur la valeur pour incrémenter.
        atomicAdd(&f_HistoTable[indexHist], 1.f);
    }
}

// À partir de l’histogramme, applique la fonction de répartition r(l)
__global__ void repart(const unsigned int f_HistoTable[], const unsigned int f_sizeTable, unsigned int f_RepartionTable[]) {
    //__shared__ repartitionTable [f_sizeTable]; 
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
    const int nbThreadTotal = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tidGlobal = tidx + tidy * blockDim.x * gridDim.x;

	for (; tidGlobal < f_sizeTable; tidGlobal += nbThreadTotal) { 
        // Deux façons de procéder:

        // On attend que la valeur précedente soit calculée avec de la synchronisation de thread
        //__syncthreads() ou atomicAdd(f_RepartionTable[x - 1])
        //f_RepartionTable[x] = f_RepartionTable[x - 1] + f_HistoTable[x];

        // Soit on fait des calculs redondants de somme
        int res = 0;
        for (int k = 0; k <= tidGlobal; k++) {  
            res += f_HistoTable[k]; 
        }
        f_RepartionTable[tidGlobal] = res;
    } 
}

// À partir de la répartition précédente, “étaler” l’histogramme.
__global__ void equalization(const unsigned int f_RepartionTable[], const unsigned int f_sizeTableRepartition, float f_ValueTable[], const unsigned int sizeValueTable) {
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
    const int nbThreadTotal = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tidGlobal = tidx + tidy * blockDim.x * gridDim.x;
    // f_sizeTableRepartition = L
    // sizeValueTable = n
    float coef = ((float)f_sizeTableRepartition - 1.f) / (float)(sizeValueTable * f_sizeTableRepartition) ; // (L - 1) / (L * n)
    for (; tidGlobal < sizeValueTable; tidGlobal += nbThreadTotal) {
        unsigned int indiceRepart = roundf(f_ValueTable[tidGlobal] * (f_sizeTableRepartition-1));
        f_ValueTable[tidGlobal] = coef * f_RepartionTable[indiceRepart];
    }
}