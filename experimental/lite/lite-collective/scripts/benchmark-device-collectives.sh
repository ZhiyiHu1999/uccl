#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MPI_HOME="${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.7rc1}"
NP="${NP:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

make -C "${PROJECT_DIR}/nccl" device-collectives-bench \
  MPI_HOME="${MPI_HOME}"

export CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="${PROJECT_DIR}/nccl/build:${PROJECT_DIR}/build:${CUDA_HOME:-/usr/local/cuda}/lib64:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"

exec "${MPI_HOME}/bin/mpirun" -np "${NP}" \
  -x CUDA_VISIBLE_DEVICES \
  -x LD_LIBRARY_PATH \
  "${PROJECT_DIR}/nccl/build/device_collectives_bench" "$@"
