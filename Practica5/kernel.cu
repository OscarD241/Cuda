
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define N 16 //vector size
#define BLOCK 5 //block size

using namespace std;

__global__ void sum(float *a, float *b, float *c) {
	int dataId = threadIdx.x + blockDim.x * blockIdx.x;
	if (N > dataId)
		c[dataId] = a[dataId] + b[dataId];
}

int main(){
	cudaError_t cudaStatus;
	float *h_vector1 = (float *) malloc(N * sizeof(float));
	float *h_vector2 = (float *)malloc(N * sizeof(float));
	float *h_res = (float *)malloc(N * sizeof(float));
	float *d_vector1;
	float *d_vector2;
	float *d_res;
	int nBloques = 0;
	int i;

	cudaStatus = cudaMalloc((void **)&d_vector1, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc on d_vector1 has failed " << endl;
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void **)&d_vector2, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc on d_vector2 has failed " << endl;
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void **)&d_res, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc on d_res has failed " << endl;
		goto Error;
	}
	
	for ( i = 0; i < N; i++) {
		h_vector1[i] = (float)i;// rand() / RAND_MAX;
		h_vector2[i] = (float) i;// rand() / RAND_MAX;
	}
	
	cudaStatus = cudaMemcpy(d_vector1, h_vector1, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy h_vector1 -> d_vector1 has failed" << endl;
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(d_vector2, h_vector2, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy h_vector2 -> d_vector2 has failed" << endl;
		goto Error;
	}
	
	nBloques = (int) N / BLOCK;
	if (N % BLOCK == 0)
		nBloques++;
	
	sum <<<nBloques, BLOCK >>> (d_vector1, d_vector2, d_res);
	cudaDeviceSynchronize();
	
	cudaStatus = cudaMemcpy(h_res, d_res, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy d_res -> h_res has failed" << endl;
		goto Error;
	}
	
	cout << "Vector 1: " << endl;
	for (i = 0; i < N; i++)
		printf("%.2f ", h_vector1[i]);
	
	cout << "\nVector 2: " << endl;
	for (i = 0; i < N; i++)
		printf("%.2f ", h_vector2[i]);

	cout << "\nResult: " << endl;
	for (i = 0; i < N; i++)
		printf("%.2f ",h_res[i]);

		
Error:
	cudaFree(d_vector1);
	cudaFree(d_vector2);
	cudaFree(d_res);
	free(h_vector1);
	free(h_vector2);
	free(h_res);
    return 0;
}

