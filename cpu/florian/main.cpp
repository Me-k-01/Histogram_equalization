#include <iostream> 
#include <cstring>
#include "histoCPU.hpp" 
#include "../../utils/image.hpp" 
 
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
	size_t width = inputImg._width;
	size_t height = inputImg._height;
	int nbChannel = inputImg._nbChannels;
	size_t imgSize = width * height;

	std::cout << "Image has " << width << " x " << height << " pixels" << std::endl;
	std::cout << "and has " << nbChannel << " channels of color" << std::endl; 
	
	std::cout << "Test rgb to hsv et hsv to rgb" << fileName << std::endl;
	unsigned char hsv[3][imgSize]; 
	rgb2hsv(&inputImg, hsv[0], hsv[1], hsv[2]);
  
	//hsv2rgb(hsv[0], hsv[1], hsv[2], &inputImg);
	//inputImg.save("./img_out/test_convertion.png"); 
	
	std::cout << "Test création histogramme" << std::endl;
	unsigned int histArray[256];   
	histogram(hsv[2], imgSize, histArray);

	std::cout << "Test égalisation d'histogramme" << std::endl;
	equalization(hsv[2], imgSize, histArray);
	hsv2rgb(hsv[0], hsv[1], hsv[2], &inputImg);
	inputImg.save("./img_out/test_egalisation.png"); 


	return 0;
}