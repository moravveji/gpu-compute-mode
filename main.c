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
              "mpirun -n <np> hybrid " \
              "[-n NITER] " \
              "[-m POW2]\n\n"


int main(int argc, char ** argv) {
    // declarations
    int rank, nranks;
    int const master = 0;
    int provided_thread;
    int niter, narr;
    int ncudathreads;

    // MPI prep
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided_thread);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == master) {

        if (argc <= 1) {
            fprintf(stderr, USAGE);
            exit(EXIT_FAILURE);
        }

        int opt, power2;
        while ((opt = getopt(argc, argv, "hn:m:t:")) != -1) {
            fprintf(stdout, "%s\n", "in while loop");
            switch (opt) {
                case 'h':
                    fprintf(stdout, USAGE);
                    MPI_Abort(MPI_COMM_WORLD, SUCCESS);
                // fall through
                case 'n':
                    niter = atoi(optarg);
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
                case 't':
                    ncudathreads = atoi(optarg);
                    if (ncudathreads > 128) {
                        fprintf(stderr, "Error: choose t <= 128");
                        MPI_Abort(MPI_COMM_WORLD, FAILURE);
                    }
                default:
                    fprintf(stderr, USAGE);
                    MPI_Abort(MPI_COMM_WORLD, FAILURE);
            }
        }
        MPI_Bcast(&niter, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&narr, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&ncudathreads, 1, MPI_INT, master, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // OpenMP threading prep
    int nomp = omp_get_num_threads();
    float maxerr[nomp];
    #pragma omp parallel
    { 
        int iomp = omp_get_thread_num();
         printf ("rank %d out of %d procs; thread %d out of %d OMP threads\n", \
                 rank, nranks, iomp, nomp);

        // every thread of every process uses the GPU for `niter` times
        for (int i=0; i<niter; i++) {
            maxerr[iomp] = call_saxpy(narr, ncudathreads);
        }

    }
    #pragma omp barrier

    MPI_Finalize();

    return 0;
}
