#include <iostream> 
#include <string>
 
#include "histoCPU.hpp" 
#include "image.hpp" 
 
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
	Image inputImg;
	inputImg.load(fileName);
	std::cout << "Image has " << inputImg._width << " x " << inputImg._height << " pixels" << std::endl;
	std::cout << "and has " << inputImg._nbChannels << " channels of color" << std::endl;
	size_t size = inputImg._width * inputImg._height * 3 ;
	unsigned char hsv[3][size] = {{},{},{}};   
	rgb2hsv(&inputImg, hsv[0], hsv[1], hsv[2]);
	//unsigned char hue[size]; 
	//unsigned char sat[size];
	//unsigned char val[size]; 
	//rgb2hsv(&inputImg, hue, sat, val);

	return 0;
}