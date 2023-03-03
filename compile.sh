#!/bin/bash

set -e

[[ $VSC_INSTITUTE_CLUSTER == 'wice' ]] || { echo 'Error: At the moment, only wICE is supported'; exit 11; }

module --force purge
module load OpenMPI/4.1.1-GCC-10.3.0
module load CUDA/11.7.1

make clean
make 
