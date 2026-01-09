#!/bin/bash
#SBATCH --account=odokumaci
#SBATCH --partition=orfoz
#SBATCH --constraint=weka
#SBATCH --job-name=isingc256
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH --output=ising_mpi_C.out
#SBATCH --error=ising_mpi_C.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=odokumaci24@ku.edu.tr

set -euo pipefail

echo "SLURM_JOB_ID    $SLURM_JOB_ID"
echo "SLURM_NODELIST  $SLURM_NODELIST"
echo "SLURM_NTASKS    $SLURM_NTASKS"

module purge

# Load OpenMPI (TRUBA examples commonly reference lib/openmpi/<version>)
module load lib/openmpi/4.1.6 2>/dev/null || true
module load openmpi/4.1.6     2>/dev/null || true
module load openmpi           2>/dev/null || true

echo "mpicc: $(command -v mpicc || echo MISSING)"
echo "mpirun: $(command -v mpirun || echo MISSING)"

if ! command -v mpicc >/dev/null; then
  echo "ERROR: mpicc not found. On TRUBA you must load an OpenMPI module."
  echo "Try these interactively on the login node:"
  echo "  module avail openmpi"
  echo "  module avail lib/openmpi"
  module list || true
  exit 2
fi


mpicc -O3 -march=native -ffast-math ising_mpi_Etol.c -o ising_mpi_Etol -lm


#mkdir -p data_c

N=512
Tmin=2.4
Tmax=3.0
steps=4
reps=20

OUT="data_c/ising_Etol_N${N}_T${Tmin}-${Tmax}_steps${steps}_reps${reps}.tsv"

mpirun -np ${SLURM_NTASKS} ./ising_mpi_Etol \
  --N ${N} --Tmin ${Tmin} --Tmax ${Tmax} --steps ${steps} \
  --reps ${reps} \
  --meas_sweeps 20000 --max_sweeps 100000 \
  --window 7000 --tol 0.0001 \
  --seed 12345 \
  --out "${OUT}"
