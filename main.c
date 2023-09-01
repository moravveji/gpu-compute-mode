#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <omp.h>
#include <mpi.h>

#include "kernel.h"

#define SUCCESS 0
#define FAILURE 11
#define USAGE "\nUsage: \n" \
              "mpirun -n <np> launcher " \
              "[-n NITER] " \
              "[-m POW2]\n\n"


int main(int argc, char ** argv) {
    // declarations
    int rank, nranks;
    int const master = 0;
    int provided_thread;
    int niter, narr;
    int ncudablocks, ncudathreads;

    // MPI prep
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided_thread);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // OpenMP prep
    int nomp;
    #pragma omp parallel shared(nomp)
    {
        nomp = omp_get_num_threads();
    }

    // argument parsing
    if (rank == master) {

        if (argc <= 1) {
            fprintf(stderr, USAGE);
            exit(EXIT_FAILURE);
        }

        // parse cmd args
        int opt, power2;
        while ((opt = getopt(argc, argv, "hn:m:")) != -1) {
            switch (opt) {
                case 'h':
                    fprintf(stdout, USAGE);
                    MPI_Abort(MPI_COMM_WORLD, SUCCESS);
                    break;
                // fall through
                case 'n':
                    niter = atoi(optarg);
                    if (niter < 1) {
                        fprintf(stderr, "Error: choose n >= 1");
                        MPI_Abort(MPI_COMM_WORLD, FAILURE);
                    }
                    fprintf(stdout, "niter = %d\n", niter);
                    break;
                case 'm':
                    power2 = atoi(optarg);
                    if (power2 > 30) {
                        fprintf(stderr, "Error: choose m <= 30");
                        MPI_Abort(MPI_COMM_WORLD, FAILURE);
                    }
                    narr = 1 << power2;
                    fprintf(stdout, "array_size = %d\n", narr);
                    break;
                default:
                    fprintf(stderr, USAGE);
                    MPI_Abort(MPI_COMM_WORLD, FAILURE);
            }
        }

        // divide GPU SMs among all procs and threads
        // print_gpu_info();
        ncudathreads = get_maxThreadsPerBlock();
        ncudablocks = get_num_blocks(nranks, nomp);

        fprintf(stdout, "Launch Config: ");
        fprintf(stdout, "MPI ranks = %d; ", nranks);
        fprintf(stdout, "OMP_NUM_THREADS = %d; ", nomp);
        fprintf(stdout, "Num blocks = %d; ", ncudablocks);
        fprintf(stdout, "Threads per block = %d\n", ncudathreads);
    
        MPI_Bcast(&niter, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&narr, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&ncudablocks, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&ncudathreads, 1, MPI_INT, master, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // multi-threaded & multi-process loop
    #pragma omp parallel default(none) shared(rank, nranks, nomp, niter, narr, ncudablocks, ncudathreads)
    { 
        int iomp = omp_get_thread_num();
        float maxerr = 0.0f;

        // every thread of every process uses the GPU for `niter` times
        for (int i=0; i<niter; i++) {
            maxerr += call_saxpy(narr, ncudablocks, ncudathreads);
        }

        maxerr /= niter;
        printf ("rank %d out of %d procs; thread %d out of %d OMP threads: maxerr=%.6f\n", \
                rank, nranks, iomp, nomp, maxerr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
