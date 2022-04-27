#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cpu_bitmap.h"

#include <stdio.h>
#include <stdlib.h>

#define DIM 1024

__global__ void kernel(unsigned char *img) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int pixel = x + y * blockDim.x * gridDim.x;

	img[pixel * 4 + 0] = 255 * x / (blockDim.x * gridDim.x);
	img[pixel * 4 + 1] = 255 * y / (blockDim.y * gridDim.y);
	img[pixel * 4 + 2] = 2 * blockIdx.x + 2 * blockIdx.y;
	img[pixel * 4 + 3] = 255;
}

int main(){/*
	// declaracion del bitmap
	CPUBitmap bitmap(DIM, DIM);
	// tamaño en bytes
	size_t size = bitmap.image_size();
	// reserva en el host
	unsigned char *host_bitmap = bitmap.get_ptr();
	// reserva en el device
	unsigned char *dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, size);
	
	// generamos el bitmap
	dim3 Nbloques(DIM / 16, DIM / 16);
	dim3 hilosB(16, 16);
	
	kernel <<<Nbloques, hilosB >>> (dev_bitmap);
	// recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy(host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost);
	// liberacion de recursos
	cudaFree(dev_bitmap);
	// visualizacion y salida
	printf("\n...pulsa ESC para finalizar...");
	bitmap.display_and_exit();
	*/
	return 0;

}
