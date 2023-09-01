#ifndef KERNEL_H
#define KERNEL_H
    void print_gpu_info(void);
    int get_maxThreadsPerBlock(void);
    int get_num_gpus(void);
    int get_num_blocks(int, int);
    void saxpy(int, float, float*, float*);
    float call_saxpy(int, int, int);
#endif