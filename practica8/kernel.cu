
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


#define N 3

__constant__ float dev_A[N][N];

__global__ void cuadrada(float *dev_B){
	// kernel lanzado con un solo bloque y NxN hilos
	int columna = threadIdx.x;
	int fila = threadIdx.y;
	int pos = fila + N * columna;
	dev_B[pos] = 0;
	// cada hilo coloca un elemento de la matriz final
	for(int k = 0 ; k < N ; k++)
		dev_B[pos] += dev_A[columna][k] * dev_A[k][fila];
}


int main(int argc, char** argv){
	float *hst_A, *hst_B;;
	float *dev_B;

	// reserva en el host
	hst_A = (float*)malloc(N*N * sizeof(float));
	hst_B = (float*)malloc(N*N * sizeof(float));
	// reserva en el device
	cudaMalloc((void**)&dev_B, N*N * sizeof(float));
	// inicializacion
	for (int i = 0; i < N*N; i++){
		hst_A[i] = (float)i;
	}
	// copia de datos
	cudaMemcpyToSymbol(dev_A, hst_A, N*N * sizeof(float));
	// dimensiones del kernel
	dim3 Nbloques(1);
	dim3 hilosB(N, N);

	// llamada al kernel bidimensional de NxN hilos
	cuadrada <<<Nbloques, hilosB >>> (dev_B);

	// recogida de datos
	cudaMemcpy(hst_B, dev_B, N*N * sizeof(float), cudaMemcpyDeviceToHost);
	// impresion de resultados
	printf("Resultado:\n");
	printf("ORIGINAL:\n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%2.0f ", hst_A[j + i * N]);
		}
		printf("\n");
	}
	printf("CUADRADA:\n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%2.0f ", hst_B[j + i * N]);
		}
		printf("\n");
	}
	// salida
	return 0;
}