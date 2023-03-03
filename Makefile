NVCC ?= nvcc
NVCCFLAGS ?= --compile -arch=sm_80

CC ?= mpicc
CFLAGS ?= -fPIC -g -Wall -Wextra -fopenmp -Wimplicit-fallthrough=0
INC ?= -I. -I$(EBROOTOPENMPI)/include
LIBS ?= -L$(EBROOTOPENMPI)/lib64 -lmpi

.PHONY: all clean

exec = hybrid
all: $(exec) $(execute)

# execute: mpi_only multiprocs_multithread

# multiprocs_multithread: hybrid
# 	export OMP_NUM_THREADS=2
# 	@echo "Using 2 processes and $OMP_NUM_THREADS threads"
# 	mpirun -n 2 $<

# mpi_only: hybrid
# 	unset OMP_NUM_THREADS
# 	@echo "Using 2 processes"
# 	mpirun -n 2 $<

hybrid : main.o kernel.o
	$(CC) $(CFLAGS) $(LIBS) $< -o $@

main.o : main.c kernel.o
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

kernel.o : kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(exec)
	rm -f core.*
