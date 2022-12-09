#include <iostream> 
#include <string>
#include <cstring> 

#include "histoGPU.hpp" 
#include "../image.hpp"
 
std::string outPutImgDir = "./img_out/"; 

void printUsage() 
{
	std::cerr	<< "Usage: " << std::endl 
			<< "Required argument"
			<< " \t -f, --file <F>: The image file name" 		
			<< "Optional arguments:"
			<< " \t -o, --output-file <F>: The destination of the generated image. By default, it is saved under ./img_out/output.png" 		
			<< " \t -d, --block-dimension <X,Y>: dimension of each block" 
			<< " \t -D, --grid-dimension  <X,Y>: dimension of the grid" 	 		
			<< " \t -b, --benchmark  <N>: The kernel number to be tested for the benchmark, if this option is not used, the provided image is processed." 	
			<< " \t				      0 : rgb2hsv - kernel to convert rgb to hsv"		
			<< " \t					  1 : histogram - kernel to generate an histogram of value"		
			<< " \t					  2 : repart - kernel to repart the histogram"		
			<< " \t				      3 : equalization - kernel to equalize the histogram"		
			<< " \t					  4 : hsv2rgb - kernel to convert back hsv to rgb"	
		    << std::endl << std::endl;
	exit( EXIT_FAILURE );
}
 
int main( int argc, char **argv ) 
{	 
	char fileName[2048];
	dim3 blocsize = {32,1,1};
	dim3 gridsize = {1,1,1};
	int numKernelToUse = -1;
	char outFileName [2048] = "./img_out/output.png";


	// Parse program arguments
	if ( argc == 1 ) {
		std::cerr << "Please give at least a file..." << std::endl;
		printUsage();
	}
	for ( int i = 1; i < argc; ++i ) {
		if ( !strcmp( argv[i], "-f") || !strcmp( argv[i], "--file") ) {
			if ( sscanf( argv[++i], "%s", fileName ) != 1 )
				printUsage();
		} else if ( !strcmp( argv[i], "-o" ) || !strcmp( argv[i], "--output-file")  ) {
			if	(sscanf( argv[++i], "%s", outFileName ) != 1)
				printUsage();
		} else if ( !strcmp( argv[i], "-d" ) || !strcmp( argv[i], "--block-dimension")  ) {
			if	(sscanf(argv[++i], "%i,%i", &blocsize.x, &blocsize.y) != 1)
				printUsage();
		} else if ( !strcmp( argv[i], "-D" ) || !strcmp( argv[i], "--grid-dimension")  ) {
			if	(sscanf(argv[++i], "%i,%i", &gridsize.x, &gridsize.y) != 1)
				printUsage();
		} else if ( !strcmp( argv[i], "-b" ) || !strcmp( argv[i], "--benchmark")  ) {
			if	(sscanf(argv[++i], "%i", &numKernelToUse) != 1)
				printUsage();
		} else {
			printUsage();
		}
	}

	Image inputImage;
    inputImage.load(fileName);

	// On regarde si le programme est a lancer en mode benchmark
	if(numKernelToUse == -1){
		// Si non, on lance le traitement de l'image
		std::cout << "Loading image: " << fileName << std::endl;
    	std::cout << "Image has " << inputImage._width << " x " << inputImage._height << " pixels and has " << inputImage._nbChannels << " channel of color" << std::endl;
		gpuCall(inputImage, 256);

		std::cout << "Saving image: " << outFileName << std::endl;
		inputImage.save(outFileName);

		std::cout << "Complete!" << std::endl;
	} else { 
		//Si oui, on effectue le benchmark
		gpuCallTest(inputImage, 256, blocsize, gridsize, static_cast<kernelToTest>(numKernelToUse));
	}

	return 0;
}