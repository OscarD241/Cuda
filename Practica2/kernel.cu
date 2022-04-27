
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
 * Práctica 2
 * 
 * Objetivo: Conocer el funcionamiento de las sumas atómicas y el llamado a los kernels de la GPU
 */

__device__ int b = 0;

__global__ void myKernel() {
	int a = 1;
	atomicAdd(&b,a);
}

int main(){
	myKernel <<<1, 12 >>> ();
	cudaDeviceSynchronize();
	printf("%d",b);

    return 0;
}



