
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define N 8

using namespace std;


int main(){
    
    cudaError_t cudaStatus;
	float *hstMat;
	float *hstMat2;
	float *devMat;
	float *devMat2;
	unsigned char flg = 0;

	size_t freeMem;
	size_t totMem;

	hstMat = (float *)malloc(N * N * sizeof(float));
	hstMat2 = (float *)malloc(N * N * sizeof(float));
	cudaStatus = cudaMalloc((void **)&devMat, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc has failed... \n" << endl;
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&devMat2, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc has failed... \n" << endl;
		goto Error;
	}

	srand((int)time(NULL));

	for (unsigned char i = 0; i < N * N; i++)
		hstMat[i] = (float)(rand() % 10);

	cudaStatus = cudaMemcpy(devMat,hstMat,N*N*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpyHostToDevice has failed... \n";
		goto Error;
	}

	cudaStatus = cudaMemcpy(devMat2, devMat, N*N * sizeof(float), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpyDeviceToDevice has failed... \n";
		goto Error;
	}

	cudaStatus = cudaMemcpy(hstMat2, devMat2, N*N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpyDeviceToHost has failed... \n";
		goto Error;
	}

	for (unsigned char i = 0; i < N * N; i++)
		if (hstMat[i] != hstMat2[i])
			flg++;

	cudaStatus = cudaMemGetInfo(&freeMem,&totMem);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemGetInfo has failed... \n";
		goto Error;
	}

	cout << "Free memory: " << freeMem << endl;
	cout << "Total memory: " << totMem << endl;
	cout << "Everything was ok... :)" << endl;
	cout << "total errors: " << (int) flg << endl;

Error:
	free(hstMat);
	free(hstMat2);
	cudaFree(devMat);
	cudaFree(devMat2);

    return 0;
}

