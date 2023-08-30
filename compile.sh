#!/bin/bash

set -e

#[[ $VSC_INSTITUTE_CLUSTER == 'wice' ]] || { echo 'Error: At the moment, only wICE is supported'; exit 11; }

module --force purge
if [[ "${VSC_INSTITUTE_CLUSTER}" == 'genius' ]]; then
    module use /apps/leuven/skylake/2021a/modules/all
    module load OpenMPI/4.1.1-GCC-10.3.0
    module load CUDA/11.7.0
elif [[ "${VSC_INSTITUTE_CLUSTER}" == 'wice' ]]; then
    module load OpenMPI/4.1.1-GCC-10.3.0
    module load CUDA/11.7.1
else
    echo "ERROR: Unrecognized machine; exiting"
    exit 21
fi

module list

make clean
make 
