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
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/.tmp/gpu-driven-benchmarks}"
RESULT_FILE="${RESULT_FILE:-${RESULT_DIR}/gpu-driven-${UCCL_GPU_DRIVEN_BACKEND}-${NP}ranks.md}"

find_nccl_baseline_lib() {
  if [[ -n "${NCCL_BASELINE_LIB:-}" ]]; then
    [[ -f "${NCCL_BASELINE_LIB}" ]] || {
      echo "error: NCCL_BASELINE_LIB does not exist: ${NCCL_BASELINE_LIB}" >&2
      return 1
    }
    printf '%s\n' "${NCCL_BASELINE_LIB}"
    return
  fi

  if [[ -n "${NCCL_LIB_PATH:-}" ]]; then
    [[ -f "${NCCL_LIB_PATH}" ]] || {
      echo "error: NCCL_LIB_PATH does not exist: ${NCCL_LIB_PATH}" >&2
      return 1
    }
    printf '%s\n' "${NCCL_LIB_PATH}"
    return
  fi

  local standard_path="${EXTERNAL_NCCL_DIR}/build/lib/libnccl.so"
  if [[ -f "${standard_path}" ]]; then
    printf '%s\n' "${standard_path}"
    return
  fi

  if [[ -d "${EXTERNAL_NCCL_DIR}" ]]; then
    local discovered
    discovered="$(find "${EXTERNAL_NCCL_DIR}" -maxdepth 4 -type f \
      \( -name 'libnccl.so' -o -name 'libnccl.so.*' \) | sort -V | tail -n 1)"
    if [[ -n "${discovered}" ]]; then
      printf '%s\n' "${discovered}"
      return
    fi
  fi

  echo "error: unable to find the real NCCL shared library" >&2
  echo "set NCCL_BASELINE_LIB, NCCL_LIB_PATH, or EXTERNAL_NCCL_DIR" >&2
  return 1
}

NCCL_BASELINE_LIB="$(find_nccl_baseline_lib)"
echo "[gpu-driven-benchmark] NCCL baseline: ${NCCL_BASELINE_LIB}" >&2

make -C "${PROJECT_DIR}/nccl" device-collectives-bench \
  MPI_HOME="${MPI_HOME}"

export CUDA_VISIBLE_DEVICES
export UCCL_GPU_DRIVEN_BACKEND
export NCCL_BASELINE_LIB
export WARMUP_ITERS ITERS
export LD_LIBRARY_PATH="${PROJECT_DIR}/nccl/build:${PROJECT_DIR}/build:${CUDA_HOME:-/usr/local/cuda}/lib64:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"

MPI_ARGS=(-np "${NP}" --bind-to none)
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

for variable in \
  NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_NET_GDR_LEVEL NCCL_BUFFSIZE \
  MSCCLPP_SOCKET_IFNAME MSCCLPP_HCA_DEVICES; do
  if [[ -n "${!variable:-}" ]]; then
    MPI_ARGS+=(-x "${variable}")
  fi
done

mkdir -p "$(dirname "${RESULT_FILE}")"
RAW_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/gpu-driven-benchmark.XXXXXX")"
trap 'rm -f "${RAW_OUTPUT}"' EXIT

"${MPI_HOME}/bin/mpirun" "${MPI_ARGS[@]}" \
  -x CUDA_VISIBLE_DEVICES \
  -x UCCL_GPU_DRIVEN_BACKEND \
  -x NCCL_BASELINE_LIB \
  -x WARMUP_ITERS \
  -x ITERS \
  -x LD_LIBRARY_PATH \
  "${PROJECT_DIR}/nccl/build/device_collectives_bench" "$@" \
  2>&1 | tee "${RAW_OUTPUT}"

{
  printf '# GPU-driven lite collectives vs NCCL\n\n'
  printf -- '- Backend: `%s`\n' "${UCCL_GPU_DRIVEN_BACKEND}"
  printf -- '- Ranks: `%s`\n' "${NP}"
  printf -- '- CUDA devices per node: `%s`\n' "${CUDA_VISIBLE_DEVICES}"
  printf -- '- Warmup iterations: `%s`\n' "${WARMUP_ITERS}"
  printf -- '- Measured iterations: `%s`\n' "${ITERS}"
  printf -- '- NCCL baseline: `%s`\n' "${NCCL_BASELINE_LIB}"
} >"${RESULT_FILE}"

awk '
  /^(allgather|allreduce|reducescatter)[[:space:]]/ {
    collective = $1
    row = ++count[collective]
    for (i = 2; i <= NF; ++i) {
      split($i, field, "=")
      key = field[1]
      parsed = field[2]
      # Accept both key=value and the older padded key=  value format.
      if (parsed == "" && i < NF) parsed = $(++i)
      value[collective, row, key] = parsed
    }
  }
  END {
    order[1] = "allgather"
    order[2] = "allreduce"
    order[3] = "reducescatter"
    title["allgather"] = "AllGather"
    title["allreduce"] = "AllReduce"
    title["reducescatter"] = "ReduceScatter"
    for (section = 1; section <= 3; ++section) {
      collective = order[section]
      printf "\n## %s\n\n", title[collective]
      print "| Bytes per rank | GPU avg device (us) | GPU avg E2E (us) | NCCL avg device (us) | NCCL avg E2E (us) | Avg E2E speedup |"
      print "|---:|---:|---:|---:|---:|---:|"
      for (row = 1; row <= count[collective]; ++row) {
        speedup = value[collective, row, "avg_speedup_e2e"]
        printf "| %s | %s | %s | %s | %s | %s |\n", \
          value[collective, row, "bytes_per_rank"], \
          value[collective, row, "gpu_avg_device_us"], \
          value[collective, row, "gpu_avg_e2e_us"], \
          value[collective, row, "nccl_avg_device_us"], \
          value[collective, row, "nccl_avg_e2e_us"], speedup
      }
    }
  }
' "${RAW_OUTPUT}" >>"${RESULT_FILE}"

echo "[gpu-driven-benchmark] Markdown result: ${RESULT_FILE}" >&2
