#include "histoGPU.hpp"
#include "../utils/commonCUDA.hpp"
#include <iostream>


#define PI 3.14159265f

// fonction d'appel au fonction gpu
void gpuCall(Image & f_Image, int nbEchantillon){

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
    HANDLE_ERROR(cudaMalloc((void**)&histoTable, nbEchantillon*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void**)&repartTable, nbEchantillon*sizeof(unsigned int)));

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
    histogram<<<blocHistogramme, grilleHistogramme>>>(valueTable, sizeImage, nbEchantillon, histoTable);
    repart<<<blocRepart, grilleRepart>>>(histoTable, nbEchantillon, repartTable);
    equalization<<<blocEqualization,grilleEqualization>>>(repartTable, nbEchantillon, valueTable, sizeImage);

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
void gpuCallTest(Image & f_Image, int nbEchantillon){

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
    HANDLE_ERROR(cudaMalloc((void**)&histoTable, nbEchantillon*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void**)&repartTable, nbEchantillon*sizeof(unsigned int)));

    HANDLE_ERROR(cudaMemcpy(pixelTableIn, f_Image._pixels, sizeImageInBytes, cudaMemcpyHostToDevice));


    //définition des bloc et grille selon les différents Kernel
    dim3 blocRGB2HSV(32,32,1);
    dim3 grilleRGB2HSV((f_Image._width + blocRGB2HSV.x-1)/blocRGB2HSV.x,(f_Image._height + blocRGB2HSV.y-1)/blocRGB2HSV.y,1);
    rgb2hsv<<<blocRGB2HSV, grilleRGB2HSV>>>(pixelTableIn, sizeImage, hueTable, saturationTable, valueTable);

        dim3 blocHistogramme(32,1,1);
        dim3 grilleHistogramme(1,1,1);
        histogram<<<blocHistogramme, grilleHistogramme>>>(valueTable, sizeImage, nbEchantillon, histoTable);


        dim3 blocRepart(32,1,1);
        dim3 grilleRepart(1,1,1);
        repart<<<blocRepart, grilleRepart>>>(histoTable, nbEchantillon, repartTable);

    for (int i = 1; i < 1025; i++)
    {
        
    dim3 blocEqualization(i,1,1);
    dim3 grilleEqualization(1,1,1);
        
    equalization<<<blocEqualization,grilleEqualization>>>(repartTable, nbEchantillon, valueTable, sizeImage);
    }
    
    
    //HANDLE_ERROR(cudaMemcpy(f_Image._pixels, pixelTableOut, sizeImageInBytes,cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(pixelTableIn));
    HANDLE_ERROR(cudaFree(pixelTableOut));
    HANDLE_ERROR(cudaFree(hueTable));
    HANDLE_ERROR(cudaFree(saturationTable));
    HANDLE_ERROR(cudaFree(valueTable));
    HANDLE_ERROR(cudaFree(histoTable));
    HANDLE_ERROR(cudaFree(repartTable));
}

// Fonction qui pour chaque pixel de l’image, calcule sa valeur dans l’espace HSV, et répartit le résultat dans trois tableaux différents
__global__ void rgb2hsv(const unsigned char f_PixelTable[], unsigned int f_sizeTable, float f_HueTable[],float f_SaturationTable[],float f_ValueTable[]){
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
__global__ void hsv2rgb(const float f_HueTable[],const float f_SaturationTable[], const float f_ValueTable[], unsigned int f_sizeTable, unsigned char f_PixelTable[]){

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
__global__ void histogram(const float f_ValueTable[], unsigned int sizeTable, const unsigned int f_NbEchantillon, unsigned int f_HistoTable[]) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x ; // tidx
	for (; tidx < sizeTable; tidx += gridDim.x * blockDim.x) {
        int indexHist = roundf(f_ValueTable[tidx] * f_NbEchantillon);
        // On doit attendre que les threads ont terminer d'écrire sur la valeur pour incrémenter.
        atomicAdd(&f_HistoTable[indexHist], 1.f);
    }
}

// À partir de l’histogramme, applique la fonction de répartition r(l)
__global__ void repart(const unsigned int f_HistoTable[], const unsigned int sizeTable, unsigned int f_RepartionTable[]) {
    //__shared__ repartitionTable [sizeTable]; 

    int tidx = blockIdx.x * blockDim.x + threadIdx.x ; // tidx
	for (; tidx < sizeTable; tidx += gridDim.x * blockDim.x) { 
        // Deux façons de procéder:

        // On attend que la valeur précedente soit calculée avec de la synchronisation de thread
        //__syncthreads() ou atomicAdd
        //f_RepartionTable[x] = f_RepartionTable[x - 1] + f_HistoTable[x];

        // Soit on fait des calculs redondants de somme
        int res = 0;
        for (size_t k = 0; k <= tidx; k++) {  
            res += f_HistoTable[k]; 
        }
        f_RepartionTable[tidx] = res;
    } 
}

// À partir de la répartition précédente, “étaler” l’histogramme.
__global__ void equalization(const unsigned int f_RepartionTable[], unsigned int sizeTableRepartition, float f_ValueTable[], const unsigned int sizeValueTable) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x ; // tidx
    // sizeTableRepartition = L
    // sizeValueTable = n
    float coef = ((float)sizeTableRepartition - 1.f) / (float)(sizeValueTable * sizeTableRepartition) ; // (L - 1) / (L * n)
    sizeTableRepartition --; // avoir L-1 avant la boucle
    for (; tidx < sizeValueTable; tidx += gridDim.x * blockDim.x) {
        unsigned int indiceRepar = roundf(f_ValueTable[tidx] * sizeTableRepartition);
        f_ValueTable[tidx] = coef * f_RepartionTable[indiceRepar];
    }
}
