#include <cuda_runtime.h>
#include <iostream>

using namespace std;

int main()
{
	int device,deviceCount,version;
	cudaError_t error;
	cudaDeviceProp deviceProp;

	error = cudaDriverGetVersion(&version);
	if (error != cudaSuccess)
	{
		cout << cudaGetErrorString(error) << endl;
		return -1;
	}
	cout << "The driver supports CUDA version " << version / 100.0 << endl;

	error = cudaGetDeviceCount(&deviceCount);
	if(error!=cudaSuccess)
	{
		if(error==cudaErrorNoDevice)
			cout << "No CUDA capable device found" << endl;
		else
			cout << cudaGetErrorString(error) << endl;
		return -1;
	}

	for (device = 0; device<deviceCount; ++device)
	{
		error=cudaGetDeviceProperties(&deviceProp, device);
		if(error!=cudaSuccess)
		{
			cout << "Can't get properties of device " << device << ": " << cudaGetErrorString(error) << endl;
		}
		else
		{
			cout << endl;
			cout << "Device " << device << ": " << deviceProp.name << endl;
			cout << "   Compute Capability " << deviceProp.major << "." << deviceProp.minor << endl;
			cout << "   " << deviceProp.clockRate / 1000 << " MHz clock rate" << endl;
			cout << "   " << deviceProp.memoryClockRate/1000 << " MHz memory clock rate" << endl;
			cout << "   " << deviceProp.totalGlobalMem << " bytes total global memory" << endl;
			cout << "   " << deviceProp.sharedMemPerMultiprocessor << " bytes shared memory per multiprocessor" << endl;
			cout << "   " << deviceProp.totalConstMem << " bytes total constant memory" << endl;
			cout << "   " << deviceProp.multiProcessorCount << " multiprocessors" << endl;
			cout << "   max " << deviceProp.maxThreadsPerBlock << " threads per Block" << endl;
			cout << "   max " << deviceProp.maxThreadsPerMultiProcessor << " threads per multiprocessor" << endl;
			cout << "   max " << deviceProp.regsPerBlock << " registers per Block" << endl;
			cout << "   max " << deviceProp.regsPerMultiprocessor << " registers per multiprocessor" << endl;
			cout << "   max " << deviceProp.maxGridSize[0] << "," << deviceProp.maxGridSize[1] << "," << deviceProp.maxGridSize[2] << " grid size (x,y,z)" << endl;
		}
	}
	return 0;
}