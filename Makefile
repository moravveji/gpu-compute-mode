NVCC = nvcc
NVCCFLAGS = --compile -arch=native
NVCCLIBS = -L$(EBROOTCUDA)/lib64 -lcudart

CXX = mpic++
CXXCFLAGS = -fPIC -g -Wall -Wextra -fopenmp -Wimplicit-fallthrough=0
INC = -I. -I$(EBROOTOPENMPI)/include
LIBS = -L$(EBROOTOPENMPI)/lib64 -lmpi

exec_name = launcher-${VSC_INSTITUTE_CLUSTER}-${VSC_ARCH_LOCAL}-${VSC_OS_LOCAL}

.PHONY: all clean

######################################################

exec = $(exec_name)
all: $(exec)

$(exec_name) : main.o kernel.o
	$(CXX) $(CXXCFLAGS) $^ -o $@ $(LIBS) $(NVCCLIBS)

main.o : main.c
	$(CXX) $(CXXCFLAGS) $(INC) -c $< -o $@

kernel.o : kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(NVCCLIBS)

######################################################

launch: np_4_omp_2

np_4_omp_2: hybrid
	OMP_NUM_THREADS=2 mpirun --map-by numa:PE=2 --report-bindings ./$< -n 10 -m 20 -t 32

######################################################

clean:
	rm -f *.o
	rm -f $(exec)
	rm -f core.*
