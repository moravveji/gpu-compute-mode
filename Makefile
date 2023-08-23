NVCC ?= nvcc
NVCCFLAGS ?= --compile -dc --x cu
NVCCLIBS ?= -L$(EBROOTCUDA)/lib64 -lcudart 
SM_ARCH ?= -arch=sm_80

CXX ?= mpic++
CXXCFLAGS ?= -fPIC -g -Wall -Wextra -fopenmp -Wimplicit-fallthrough=0
INC ?= -I. -I$(EBROOTOPENMPI)/include
LIBS ?= -L$(EBROOTOPENMPI)/lib64 -lmpi

.PHONY: all clean

######################################################

exec = hybrid
all: $(exec)

hybrid : main.o kernel.o link.o
	$(CXX) $(CXXCFLAGS) $^ -o $@ $(LIBS) $(NVCCLIBS)

main.o : main.c
	$(CXX) $(CXXCFLAGS) $(INC) -c $< -o $@

kernel.o : kernel.cu
	$(NVCC) $(SM_ARCH) $(NVCCFLAGS) -c $< -o $@ $(NVCCLIBS)

link.o : kernel.o
	$(NVCC) $(SM_ARCH) -dlink $< -o $@ $(NVCCLIBS)
######################################################

clean:
	rm -f *.o
	rm -f $(exec)
	rm -f core.*
