
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void kernel(double *vals, double *x) {
	int id = threadIdx.x;
	double a = vals[0];
	double b = vals[1];
	double c = vals[2];
	if (id < 2)
		x[id] = ((-1.0)*(b / (2.0 * a))) + (pow(-1.0, (double)(id + 1)) *
			(sqrt(pow(b, (double) 2.0) - (4.0 * a * c)) / (2.0 * a)));
}

int main(){
    cudaError_t cudaStatus;
	//Host variables
	double hstVals[] = { 1.0, 2.0, 0.0 };
	double *hstX;
	//device variables
	double *devVals;
	double *devX;

	hstX = (double *)malloc(2 * sizeof(double));
	cudaStatus = cudaMalloc((void **)&devX, 2 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc has failed... " << endl;
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&devVals, 3 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc has failed... " << endl;
		goto Error;
	}

	cudaStatus = cudaMemcpy(devVals, hstVals, 3 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy1 has failed... " << endl;
		goto Error;
	}

	kernel <<<1, 2 >>> (devVals,devX);
	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(hstX, devX, 2 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy4 has failed... " << endl;
		goto Error;
	}

	cout << "Results:\n x1 = " << hstX[0] << "\n x2 = " << hstX[1] << endl;
	cout << "Everythyng was ok... :)" << endl;

Error:
	cudaFree(devVals);
	cudaFree(devX);
    return 0;
}
