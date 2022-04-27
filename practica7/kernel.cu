
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define N 16

using namespace std;

__global__ void pi(float *vector, float *res) {
	__shared__ float tmp[N];
	int id = threadIdx.x;
	tmp[id] = 1 / pow(vector[id],2);
	__syncthreads();

	int step = N / 2;

	while (step){
		if (id < step)
			tmp[id] += tmp[id + step];
		__syncthreads();
		step /= 2;
	}

	if (id == 0)
		*res = tmp[id];//sqrt(6 * tmp[id]);
}

int main(){
	cudaError_t cudaStatus;
	//host variables
	float *hVec = (float *)malloc(N * sizeof(float));
	float *hSum = (float *)malloc(sizeof(float));
	//device variables
	float *dVec;
	float *dSum;

	cudaStatus = cudaMalloc((void **)&dVec, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc 1 has failed... " << endl;
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dSum, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc 2 has failed... " << endl;
		goto Error;
	}

	for (char i = 0; i < N; i++)
		hVec[i] = (float)i;

	cudaStatus = cudaMemcpy(dVec, hVec, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy1 has failed... " << endl;
		goto Error;
	}

	pi <<<1, N >>> (dVec,dSum);
	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(hSum, dSum, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy2 has failed... " << endl;
		goto Error;
	}

	cout << "Result = " << *hSum << endl;

Error:
	cudaFree(dVec);
	cudaFree(dSum);
	free(hVec);
	free(hSum);
	return 0;
}
