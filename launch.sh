#!/usr/bin/env bash

set -e

launcher="launcher-${VSC_INSTITUTE_CLUSTER}-${VSC_ARCH_LOCAL}-${VSC_OS_LOCAL}"

source dependencies.sh

# echo -e "\nInfo: Launching a single-process with a single-thread"
# export OMP_NUM_THREADS=1
# mpirun -n 1 ./${launcher} -n 10 -m 5

# echo -e "\nInfo: Launching a single-process with two-threads"
# export OMP_NUM_THREADS=2
# mpirun -n 1 ./${launcher} -n 10 -m 5

# echo -e "\nInfo: Launching a single-process with eight-threads"
# export OMP_NUM_THREADS=8
# mpirun -n 1 ./${launcher} -n 10 -m 5

# echo -e "\nInfo: Launching two-processes with single-thread each"
# export OMP_NUM_THREADS=1
# mpirun -n 2 ./${launcher} -n 10 -m 5

echo -e "\nInfo: Launching two-processes with two-threads each"
export OMP_NUM_THREADS=2
mpirun -n 2 ./${launcher} -n 10 -m 5

# echo -e "\nInfo: Launching four-processes with two-threads each"
# export OMP_NUM_THREADS=2
# mpirun -n 4 ./${launcher} -n 10 -m 5

echo -e "\nDone\n"