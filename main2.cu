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
    
    std::cout << "Image has " << inputImage._width << " x " << inputImage._height << " pixels and has " << inputImage._nbChannels << "channel of color" << std::endl;

    inputImage.save(outPutImgDir +"test.png");

	return 0;
}