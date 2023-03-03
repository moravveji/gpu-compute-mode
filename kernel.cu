
#include <stdlib.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

float call_saxpy(int narr, int nthreads) {
    float *x, *y, *dx, *dy;  // host and device x and y arrays
    x = (float*) malloc(narr * sizeof(float));
    y = (float*) malloc(narr * sizeof(float));

    cudaMalloc(&dx, narr * sizeof(float));
    cudaMalloc(&dy, narr * sizeof(float));

    for (int i=0; i<narr; i++) {
        x[i] = 1.0f; y[i] = 2.0f;
    }

    cudaMemcpy(dx, x, narr*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, narr*sizeof(float), cudaMemcpyHostToDevice);

    int nblocks = (narr + nthreads - 1) / nthreads;
    saxpy<<<nblocks, nthreads>>>(narr, 2.0f, dx, dy);

    cudaMemcpy(y, dy, narr*sizeof(float), cudaMemcpyDeviceToHost);

    float maxerr = 0.0f;
    for (int i=0; i<narr; i++) {
        maxerr = max(maxerr, abs(y[i] - 4.0f));
    }

    cudaFree(dx); cudaFree(dy);
    free(x); free(y);

    return maxerr;
}
