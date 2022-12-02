#include <iostream> 
#include <cstring>
#include "histoCPU.hpp" 
#include "image.hpp" 
 
 void printUsage() 
{
	std::cerr << "Usage: " << std::endl
			<< " \t -f <F>: <F> image file name" 			
		    << std::endl << std::endl;
	exit( EXIT_FAILURE );
}
 
int main( int argc, char **argv ) 
{	 
	char fileName[2048];

	// Parse program arguments
	if ( argc == 1 )  {
		std::cerr << "Please give a file..." << std::endl;
		printUsage();
	} 

	for ( int i = 1; i < argc; ++i ) {
		if ( !strcmp( argv[i], "-f" ) )  {
			if ( sscanf( argv[++i], "%s", fileName ) != 1 )
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
	int width = inputImg._width;
	int height = inputImg._height;
	int nbChannel = inputImg._nbChannels;

	std::cout << "Image has " << width << " x " << height << " pixels" << std::endl;
	std::cout << "and has " << nbChannel << " channels of color" << std::endl; 
	
	unsigned char hsv[3][width * height];   
	rgb2hsv(&inputImg, hsv[0], hsv[1], hsv[2]);
	hsv2rgb(hsv[0], hsv[1], hsv[2], &inputImg);
	std::cout << "Test rgb to hsv" << fileName << std::endl;
  
	inputImg.save("./imgoutput/test.png"); 

	return 0;
}