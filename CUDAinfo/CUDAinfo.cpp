#include <cuda_runtime.h>
#include <iostream>

using namespace std;

int main()
{
	int device,deviceCount,version;
	cudaError_t error;
	cudaDeviceProp deviceProp;

	cout << "This program is using the Nvidia CUDA Toolkit " << CUDART_VERSION / 1000 << "." << (CUDART_VERSION % 1000) / 10 << endl << endl;
	error = cudaDriverGetVersion(&version);
	if (error != cudaSuccess)
	{
		cout << cudaGetErrorString(error) << endl;
		return -1;
	}

	int major = version / 1000;
	int minor = (version - (major * 1000)) / 10;
	cout << "The driver supports CUDA up to version " << major << "." << minor << endl;

	error = cudaGetDeviceCount(&deviceCount);
	if(error!=cudaSuccess)
	{
		cout << "cudaGetDeviceCount() error: " << cudaGetErrorString(error) << endl;
		return -1;
	}
	else
		if (deviceCount == 0)
		{
			cout << "No CUDA device found" << endl;
			return -1;
		}

	for (device = 0; device<deviceCount; ++device)
	{
		error=cudaGetDeviceProperties(&deviceProp, device);
		if(error!=cudaSuccess)
		{
			cout << "Can't get properties of device " << device << ": " << cudaGetErrorString(error) << endl;
			cout << "cudaGetDeviceProperties() error:" << cudaGetErrorString(error) << endl;
			return -1;
		}
		else
		{
			cout << endl;
			cout << "Device " << device << ": " << deviceProp.name << endl;
			cout << "   " << deviceProp.clockRate/1000 << " MHz clock rate" << endl;
			cout << "   " << deviceProp.memoryClockRate/1000 << " MHz memory clock rate" << endl;
			cout << "   Compute Capability " << deviceProp.major << "." << deviceProp.minor << endl;
			cout << "   " << deviceProp.multiProcessorCount << " multiprocessors" << endl;
/*			if(deviceProp.major<3)
				cout << "   max 8 resident blocks per multiprocessor" << endl;
			else
				cout << "   max 16 resident blocks per multiprocessor" << endl;
			cout << "   max " << deviceProp.regsPerBlock << " registers per block" << endl;
			if(deviceProp.major==1)
				cout << "   max 128 registers per thread" << endl;
			if(deviceProp.major==2)
				cout << "   max 63 registers per thread" << endl;
			if(deviceProp.major==3)
			{
				if(deviceProp.minor==5)
					cout << "   max 255 registers per thread" << endl;
				else
					cout << "   max 63 registers per thread" << endl;
			}
*/
			cout << "   max " << deviceProp.maxThreadsPerBlock << " threads per Block" << endl;
			cout << "   max " << deviceProp.maxThreadsPerMultiProcessor << " threads per multiprocessor" << endl;
			cout << "   max " << deviceProp.regsPerBlock << " registers per Block" << endl;
			cout << "   max " << deviceProp.regsPerMultiprocessor << " registers per multiprocessor" << endl;

		}
	}
	return 0;
}
