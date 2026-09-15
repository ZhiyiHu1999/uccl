#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MPI_HOME="${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.7rc1}"
EXTERNAL_NCCL_DIR="${EXTERNAL_NCCL_DIR:-/home/yangz/nfs/zhongjie/nccl}"
NP="${NP:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
UCCL_GPU_DRIVEN_BACKEND="${UCCL_GPU_DRIVEN_BACKEND:-host}"
HOSTS="${HOSTS:-}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
ITERS="${ITERS:-100}"

if [[ -z "${NCCL_BASELINE_LIB:-}" ]]; then
  if [[ ! -d "${EXTERNAL_NCCL_DIR}" ]]; then
    echo "error: external NCCL directory not found: ${EXTERNAL_NCCL_DIR}" >&2
    exit 1
  fi
  NCCL_BASELINE_LIB="$(find "${EXTERNAL_NCCL_DIR}" -maxdepth 4 -type f \
    \( -name 'libnccl.so' -o -name 'libnccl.so.*' \) | sort -V | tail -n 1)"
fi
if [[ -z "${NCCL_BASELINE_LIB}" || ! -f "${NCCL_BASELINE_LIB}" ]]; then
  echo "error: set NCCL_BASELINE_LIB to the real NCCL shared library" >&2
  exit 1
fi

make -C "${PROJECT_DIR}/nccl" device-collectives-bench \
  MPI_HOME="${MPI_HOME}"

export CUDA_VISIBLE_DEVICES
export UCCL_GPU_DRIVEN_BACKEND
export NCCL_BASELINE_LIB
export WARMUP_ITERS ITERS
export LD_LIBRARY_PATH="${PROJECT_DIR}/nccl/build:${PROJECT_DIR}/build:${CUDA_HOME:-/usr/local/cuda}/lib64:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"

MPI_ARGS=(-np "${NP}")
if [[ -n "${HOSTS}" ]]; then
  IFS=',' read -r -a HOST_ARRAY <<<"${HOSTS}"
  IFS=',' read -r -a GPU_ARRAY <<<"${CUDA_VISIBLE_DEVICES}"
  EXPECTED_NP=$((${#HOST_ARRAY[@]} * ${#GPU_ARRAY[@]}))
  if [[ "${NP}" -ne "${EXPECTED_NP}" ]]; then
    echo "error: NP=${NP}, expected ${EXPECTED_NP} for HOSTS and CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
  HOST_SPEC=""
  for host in "${HOST_ARRAY[@]}"; do
    [[ -n "${HOST_SPEC}" ]] && HOST_SPEC+=","
    HOST_SPEC+="${host}:${#GPU_ARRAY[@]}"
  done
  MPI_ARGS+=(-H "${HOST_SPEC}")
fi

exec "${MPI_HOME}/bin/mpirun" "${MPI_ARGS[@]}" \
  -x CUDA_VISIBLE_DEVICES \
  -x UCCL_GPU_DRIVEN_BACKEND \
  -x NCCL_BASELINE_LIB \
  -x WARMUP_ITERS \
  -x ITERS \
  -x LD_LIBRARY_PATH \
  "${PROJECT_DIR}/nccl/build/device_collectives_bench" "$@"
