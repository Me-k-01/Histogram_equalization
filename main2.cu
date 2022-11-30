#include <iostream> 
#include <string>
 
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

	outputImage.save(outPutImgDir + "testallfonc.png");

	return 0;
}