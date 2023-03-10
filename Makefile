NVCC ?= nvcc
NVCCFLAGS ?= --compile -arch=sm_80 -dc --x cu
NVCCLIBS ?= -L$(EBROOTCUDA)/lib64 -lcudart 

CXX ?= mpic++
CXXCFLAGS ?= -fPIC -g -Wall -Wextra -fopenmp -Wimplicit-fallthrough=0
INC ?= -I. -I$(EBROOTOPENMPI)/include
LIBS ?= -L$(EBROOTOPENMPI)/lib64 -lmpi

.PHONY: all clean

######################################################

exec = hybrid
all: $(exec)

hybrid : main.o kernel.o
	$(CXX) $(CXXCFLAGS) $^ -o $@ $(LIBS) $(NVCCLIBS)

main.o : main.c
	$(CXX) $(CXXCFLAGS) $(INC) -c $< -o $@

kernel.o : kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(NVCCLIBS)

######################################################

clean:
	rm -f *.o
	rm -f $(exec)
	rm -f core.*
