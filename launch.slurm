#!/usr/bin/env bash

#SBATCH --account lpt2_sysadmin
#SBATCH --cluster genius
#SBATCH --partition gpu_p100
##SBATCH --reservation Rocky8
#SBATCH --time 10:00
#SBATCH --nodes 1
#SBATCH --ntasks 9
#SBATCH --gpus-per-node 1
##SBATCH --gpu_cmode shared

bash compile.sh

# collect device info and start a monitoring daemon
#nvidia-smi --querry --id=0 --display=MEMORY,UTILIZATION,ECC,COMPUTE
#nvidia-smi compute-policy --id=0 --list --set-timeslice=1
#nvidia-smi daemon --id=0 --delay=1 --select=puce --path=$(pwd)/daemon

bash launch.sh

#nvidia-smi daemon --term   # stop the daemon
