#!/usr/bin/env bash

module --force purge
case "${VSC_INSTITUTE_CLUSTER}" in 
    'genius' )
        module use /apps/leuven/rocky8/skylake/2021a/modules/all
        module load OpenMPI/4.1.1-GCC-10.3.0
        module load CUDA/11.7.0
        ;;
    'wice' )
        module load OpenMPI/4.1.1-GCC-10.3.0
        module load CUDA/11.7.1
        ;;
    'sandbox' )
        module use /apps/leuven/rocky8/skylake-6132/2021a/modules/all
        module load OpenMPI/4.1.1-GCC-10.3.0
        module load CUDA/11.7.0
        ;;
    * )
        echo "Unrecognized cluster: ${VSC_INSTITUTE_CLUSTER}. Exiting"
        exit 21
        ;;
esac
