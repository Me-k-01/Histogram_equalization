#include <iostream> 
#include <string>
#include <cstring> 

#include "histoGPU.hpp" 
#include "../utils/image.hpp"
 
std::string outPutImgDir = "./img_out/"; 
const unsigned int maxKernelNumber = 12;

void printUsage() 
{
	std::cerr	<< "Usage: " << std::endl 
			<< " \t -h, --help  : Display help" << std::endl 
			<< "Required argument" << std::endl 
			<< " \t -f, --file <F>: The image file name" << std::endl 
			<< "Optional arguments:" << std::endl 
			<< " \t -o, --output-file <F>: The destination of the generated image. By default, it is saved under ./img_out/output.png" << std::endl 
			<< " \t -d, --block-dimension <X Y>: dimension of each block" << std::endl 
			<< " \t -D, --grid-dimension  <X Y>: dimension of the grid" << std::endl 
			<< " \t -b, --benchmark  <N>: The kernel number to be tested for the benchmark, if this option is not used, the provided image is processed." << std::endl 
			<< " \t\t\t    0 : rgb2hsv - kernel to convert rgb to hsv" << std::endl 
			<< " \t\t\t    1 : rgb2hsv_MinimuDivergence - kernel to convert rgb to hsv with minimum divergence" << std::endl 
			<< " \t\t\t    2 : rgb2hsv_CoordinatedOutputs - kernel to convert rgb to hsv with coordinated entries" << std::endl 
			<< " \t\t\t    3 : histogram - kernel to generate an histogram of value" << std::endl 
			<< " \t\t\t    4 : histogram_WithSharedMemory - kernel to generate an histogram of value with use of shared memory" << std::endl 
			<< " \t\t\t    5 : histogram_WithSharedMemoryAndHardcodedSize - kernel to generate an histogram of value with use of shared memory with hardcoded size (" << HISTO_SIZE << ")" << std::endl 
			<< " \t\t\t    6 : histogram_WithMinilmumCalculationDepencies - kernel to generate an histogram of value with use of shared memory and minimum calculation dependencies" << std::endl 	
			<< " \t\t\t    7 : repart - kernel to repart the histogram" << std::endl 	
			<< " \t\t\t    8 : repart_WithSharedMemory - kernel to repart the histogram with use of shared memory" << std::endl 	
			<< " \t\t\t    9 : repart_WithSharedMemoryAndHardcodedSize - kernel to repart the histogram with use of shared memory with hardcoded size (" << HISTO_SIZE << ")"  << std::endl 
			<< " \t\t\t   10 : equalization - kernel to equalize the histogram" << std::endl 
			<< " \t\t\t   11 : equalization_ConstantCoefficient - kernel to equalize the histogram with the use of constant coefficient" << std::endl 
			<< " \t\t\t   12 : hsv2rgb - kernel to convert back hsv to rgb" << std::endl 
		    << std::endl; 
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
		if ( !strcmp( argv[i], "-h") || !strcmp( argv[i], "--help") ) {
			printUsage();
		} else if ( !strcmp( argv[i], "-f") || !strcmp( argv[i], "--file") ) {
			i++;
			if ( sscanf( argv[i], "%s", fileName ) != 1 )
				printUsage();
		} else if ( !strcmp( argv[i], "-o" ) || !strcmp( argv[i], "--output-file")  ) {
			i++;
			if	(sscanf( argv[i], "%s", outFileName ) != 1)
				printUsage();
		} else if ( !strcmp( argv[i], "-d" ) || !strcmp( argv[i], "--block-dimension")  ) {
			i++;
			if	(sscanf(argv[i], "%i", &blocsize.x) != 1)
				printUsage();
			i++;
			if	(sscanf(argv[i], "%i", &blocsize.y) != 1)
				printUsage();
		} else if ( !strcmp( argv[i], "-D" ) || !strcmp( argv[i], "--grid-dimension")  ) {
			i++;
			if	(sscanf(argv[i], "%i", &gridsize.x) != 1)
				printUsage();
			i++;
			if	(sscanf(argv[i], "%i", &gridsize.y) != 1)
				printUsage();
		} else if ( !strcmp( argv[i], "-b" ) || !strcmp( argv[i], "--benchmark")  ) {
			if	(sscanf(argv[++i], "%i", &numKernelToUse) != 1)
				printUsage();
				//on s'assure aussi que le numéro sélectionné n'est pas supérieur au maximum utilisable et est bien au moins égale à zéro
			if (numKernelToUse < 0 || numKernelToUse > maxKernelNumber)
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
		gpuCallBenchmark(inputImage, 256, blocsize, gridsize, static_cast<kernelToTest>(numKernelToUse));
	}

	return 0;
}