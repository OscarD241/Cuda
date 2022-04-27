
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
* Práctica 1
* Objetivo: Obtener una breve descripción de los recursos con los que cuenta el sistema
*/

int main(){
	int counter;											//Contador de dispositivos
	cudaDeviceProp prop;									//Estructura en la que se almacenarán las características de la GPU

	cudaError_t error = cudaGetDeviceCount(&counter);		//obtenemos la cantidad de dispositivos que se tiene
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));
	else {
		printf("Total de dispositivos: %d\n", counter);
		for (int i = 0; i < counter; i++) {
			cudaGetDeviceProperties(&prop, i);
			printf("Nombre: %s\n", prop.name);
			printf("Capacidad de cómputo mayor: %d\n",prop.major);
			printf("Capacidad de cómputo menor: %d\n", prop.minor);
			printf("Número de Streaming Multiprocessors: %d\n",prop.multiProcessorCount);
			printf("Máximo número de hilos por bloque: %d\n",prop.maxThreadsPerBlock);
			printf("Máximo número de bloques por grid: %d\n",prop.maxGridSize[0]);
		}
	}

	printf("\n Presiona una tecla para continuar...");
	fflush(stdin);
	char t = getchar();
    return 0;
}

