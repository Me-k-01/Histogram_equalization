#include <iostream> 
#include <string>
#include <cstring>
 
#include "histoCPU2.hpp" 
#include "image.hpp"
 
std::string outPutImgDir = ".\\imgoutput\\";

void printUsage() 
{
	std::cerr	<< "Usage: " << std::endl
			<< " \t -f <F>: <F> image file name" 			
		    << std::endl << std::endl;
	exit( EXIT_FAILURE );
}
 
int main( int argc, char **argv ) 
{	 
	char fileName[2048];

	// Parse program arguments
	if ( argc == 1 ) 
	{
		std::cerr << "Please give a file..." << std::endl;
		printUsage();
	}

	for ( int i = 1; i < argc; ++i ) 
	{
		if ( !strcmp( argv[i], "-f" ) ) 
		{
			if ( sscanf( argv[++i], "%s", &fileName ) != 1 )
				printUsage();
		}
		else
			printUsage();
	}

	// Version CPU
	// Get input image
	std::cout << "Loading image: " << fileName << std::endl;
	Image inputImage;
    inputImage.load(fileName);
    std::cout << "Image has " << inputImage._width << " x " << inputImage._height << " pixels and has " << inputImage._nbChannels << " channel of color" << std::endl;


	// tests des fonctions CPU
	/*
	Image outputImage;
	std::vector<float> htable, stable, vtable;
	std::vector<unsigned char> pixels;
	std::vector<unsigned int> histoTable, repartTable;
	rgb2hsv(inputImage, htable,stable,vtable);

	histogram(vtable, 256, histoTable);
	repart(histoTable, repartTable);
	equalization(repartTable, vtable);
	
	hsv2rgb(htable,stable,vtable,pixels);
	outputImage._height = inputImage._height;
	outputImage._width = inputImage._width;
	outputImage._nbChannels = inputImage._nbChannels;
	outputImage._pixels = pixels.data();
*/

	const int imagesize = inputImage._height* inputImage._width;

	float * hueTable = new float[imagesize],* saturationTable = new float[imagesize],* valueTable = new float[imagesize];
	int nbEchantillon = 256;
	unsigned int * histoTable = new unsigned int[nbEchantillon], * repartTable = new unsigned int[nbEchantillon];

	rgb2hsv(inputImage, hueTable, saturationTable, valueTable);
	histogram(valueTable, imagesize, nbEchantillon, histoTable);
	repart(histoTable, nbEchantillon, repartTable);
	equalization(repartTable, nbEchantillon, valueTable, imagesize);
	hsv2rgb(hueTable,saturationTable,valueTable, imagesize,inputImage._pixels);

	inputImage.save(outPutImgDir + "testallfonc.png");

	return 0;
}