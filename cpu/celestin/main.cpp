#include <iostream> 
#include <string>
#include <cstring>
#include <iomanip>

#include "histoCPU.hpp" 
#include "../../utils/image.hpp"
 
std::string outPutImgDir = ".\\img_out\\";

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
	const int imagesize = inputImage._height* inputImage._width;

	float * hueTable = new float[imagesize],* saturationTable = new float[imagesize],* valueTable = new float[imagesize];
	int nbEchantillon = 256;
	unsigned int * histoTable = new unsigned int[nbEchantillon], * repartTable = new unsigned int[nbEchantillon];

	float rgb2hsvTime = rgb2hsv(inputImage, hueTable, saturationTable, valueTable);
	float histogramTime = histogram(valueTable, imagesize, nbEchantillon, histoTable);
	float repartTime = repart(histoTable, nbEchantillon, repartTable);
	float equalizationTime = equalization(repartTable, nbEchantillon, valueTable, imagesize);
	float hsv2rgbTime = hsv2rgb(hueTable,saturationTable,valueTable, imagesize,inputImage._pixels);

	inputImage.save(outPutImgDir + "output.png");

	std::cout <<"Complete !" << std::endl << std::endl;

	std::cout <<"=================================================================" << std::endl;
	std::cout <<"   Recapitulatif des temps d'execution pour chaque fonction :    " << std::endl;
	std::cout << "\trgb2hsv      -> " << std::setprecision(10) << rgb2hsvTime << " milisecondes" << std::endl;
	std::cout << "\thistogram    -> " << std::setprecision(10) << histogramTime << " milisecondes" << std::endl;
	std::cout << "\trepart       -> " << std::setprecision(10) << repartTime << " milisecondes" << std::endl;
	std::cout << "\tequalization -> " << std::setprecision(10) << equalizationTime << " milisecondes" << std::endl;
	std::cout << "\thsv2rgb      -> " << std::setprecision(10) << hsv2rgbTime << " milisecondes" << std::endl;
	std::cout <<"=================================================================" << std::endl;

	return 0;
}