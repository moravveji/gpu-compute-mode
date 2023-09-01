
#include <stdio.h>
#include <stdlib.h>

void print_gpu_info(void) {
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);

    fprintf(stdout, "maxBlocksPerMultiProcessor=%d, \
            multiProcessorCount=%d, \
            maxGridSize[0]=%d, \
            maxThreadsDim[0]=%d, \
            maxThreadsPerBlock=%d,\n", \
            dev.maxBlocksPerMultiProcessor, \
            dev.multiProcessorCount, \
            dev.maxGridSize[0], \
            dev.maxThreadsDim[0], \
            dev.maxThreadsPerBlock);
}

int get_maxThreadsPerBlock(void) {
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    return dev.maxThreadsPerBlock;
}

int get_num_gpus(void) {
    int ndev;
    cudaGetDeviceCount(&ndev);
    return ndev;
}

int get_num_blocks(int nranks, int nomp) {
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    int nblocks = (dev.maxBlocksPerMultiProcessor * dev.multiProcessorCount) / (nranks * nomp);

    return nblocks;
}

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

float call_saxpy(int narr, int nblocks, int nthreads) {
    float *x, *y, *dx, *dy;  // host and device x and y arrays
    x = (float*) malloc(narr * sizeof(float));
    y = (float*) malloc(narr * sizeof(float));
    if (x == NULL || y == NULL) {
        fprintf(stderr, "ERROR: malloc failed");
        exit(21);
    }

    if (cudaMalloc((void**) &dx, narr * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc failed for dx");
        exit(21);
    }
    if (cudaMalloc((void**) &dy, narr * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc failed for dy");
        exit(21);
    }

    for (int i=0; i<narr; i++) {
        x[i] = 1.0f; y[i] = 2.0f;
    }

    cudaMemcpy(dx, x, narr*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, narr*sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<nblocks, nthreads>>>(narr, 2.0f, dx, dy);

    cudaMemcpy(y, dy, narr*sizeof(float), cudaMemcpyDeviceToHost);

    float maxerr = 0.0f;
    for (int i=0; i<narr; i++) {
        float locerr = fabs(y[i] - 4.0f);
        maxerr = (locerr > maxerr) ? locerr : maxerr;
    }

    cudaFree(dx); cudaFree(dy);
    free(x); free(y);

    return maxerr;
}
