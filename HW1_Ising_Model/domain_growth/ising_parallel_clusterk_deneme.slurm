#!/bin/bash
#SBATCH --account=odokumaci
#SBATCH --partition=hamsi
#SBATCH --constraint=weka
#SBATCH --job-name=sf_nonzero
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -c 1
#SBATCH -C weka
#SBATCH -t 72:00:00
#SBATCH --output=ising_growth.out
#SBATCH --error=ising_growth.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=odokumaci24@ku.edu.tr

echo "SLURM_JOB_ID $SLURM_JOB_ID"
echo "SLURM_NODELIST $SLURM_NODELIST"

# 1) Clean env + load toolchain
module purge
module load apps/truba-ai/cpu-2024.0

# 2) Activate conda env (mpi4py lives here)
eval "$(/arf/sw/apps/truba-ai/cpu/miniforge3-2024/bin/conda shell.bash hook)"
conda activate cpu-2024.0

# Safety: avoid Intel MPI env leakage
unset I_MPI_HYDRA_BOOTSTRAP I_MPI_PMI_LIBRARY I_MPI_FABRICS I_MPI_SHM

# 3) Keep BLAS single-threaded per rank
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------
# Run parameters
# -----------------------
PY=ising_sim_parallel_v4_clusterk.py   # <-- rename to your actual script filename
OUTDIR=data_sf_nonzero

N=64
Tmin=1.9
Tmax=2.0 
steps=1
reps=1

tmax=1000
nsamples=10
tmin_sample=1
seed=12345

# Choose growth metric:
#   floodfill -> cluster recognition
#   sf        -> structure factor first-moment length
GROWTH_METRIC="sf"

# Structure-factor options:
# - I recommend keeping --sf_use_2pi (since your k is in radians)
# - I recommend NOT using --sf_subtract_mean (you already set S[0,0]=0)
# - qmax_frac is important: don't leave it at 1.0 (Nyquist junk)
SF_QMAX_FRAC=0.5
SF_FLAGS="--sf_use_2pi --sf_subtract_mean --sf_qmax_frac ${SF_QMAX_FRAC}"
# If you really want mean-subtraction too:
# SF_FLAGS="--sf_use_2pi --sf_subtract_mean --sf_qmax_frac ${SF_QMAX_FRAC}"

srun --mpi=pmi2 -n 56 python -u "${PY}" \
  --mode growth \
  --growth_metric "${GROWTH_METRIC}" \
  --N "${N}" \
  --Tmin "${Tmin}" --Tmax "${Tmax}" --steps "${steps}" \
  --reps "${reps}" \
  --tmax "${tmax}" \
  --nsamples "${nsamples}" \
  --tmin_sample "${tmin_sample}" \
  --seed "${seed}" \
  --outdir "${OUTDIR}" \
  ${SF_FLAGS}







