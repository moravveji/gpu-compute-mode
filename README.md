# Testing GPU Compute Mode

## Purpose
In Slurm, it is possible to set the GPU compute mode when requesting resources.
The two options of interest are `default` (multi-process access) and 'exclusive_process` (single-process access).

Setting GPU compute mode is made possible by using an additional [GPU Compute Mode SPANK plugin](https://github.com/stanford-rc/slurm-spank-gpu_cmode), which basically tailors a call to the `nvidia-smi` and sets the GPU access mode.

In order to test this capability on the Nvidia A100 GPUs on the VSC wICE cluster, this simplistic program can be used.

## In simple words
... each MPI process spawns multiple threads, and each thread (of each process) launches the `saxpy` kernel on a device.
This is repeated `N` times to give all processes and threads sufficient time to sit on the device and execute concurrently.

## Dependencies
- `OpenMPI` (tested with version 4.1)
- `CUDA` (version 11.7 or later)

## Installation
In order to compile this snippet, execute the `compile.sh` shell script.
Perhaps you want to adapt the dependency versions to what is available in your site before the compilation.

## Changing compiler flags
To modify the choice of compiler (`CC` or `NVCC`) and/or compiler flags (e.g. `CFLAGS` or `NVCCFLAGS`), you may export your environment varialbe before triggering the `./compile.sh`.

## Call signature
How to launch the so-called `hybrid` executable?

Assume you plan to launch `<N_MPI_PROCS>` processes, where each process would spawn maximum `<N_OMP_THREADS>` threads.
You additionally need to specify the number of iterations `<N_ITER>` (something between 10 and 100 would be reasonable).
Lastly, the array size must be specified.
This is in power of 2 (actually `1<<M` in the code).

E.g. for 2 processes and 4 threads per process:

``` bash
# step 1
srun -A <account> --cluster wice --partition gpu --nodes 1 --ntasks 2 --cpus-per-task 4 --gpus-per-node 1 gpu_cmode default --pty /bin/bash -l

# step 2
cd <somewhere>
git clone git@github.com:moravveji/gpu-compute-mode.git
cd gpu-compute-mode
./compile

# step 3
export OMP_NUM_THREADS=<N_OMP_THREADS>
mpirun -n <N_MPI_PROCS> ./hybrid -n <N_ITER> -m <M> 
```