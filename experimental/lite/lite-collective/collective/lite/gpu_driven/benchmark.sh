#!/usr/bin/env bash
set -euo pipefail

for argument in "$@"; do
  if [[ "$argument" == -h || "$argument" == --help ]]; then
    cat <<'USAGE'
Usage: benchmark.sh [sizes...]
       benchmark.sh -b BEGIN -e END [-f FACTOR] [-g 1] [-w WARMUPS] [-n ITERS]
-c, --collective: allgather/allreduce/reducescatter/all (default all).
Sizes: bytes or B/K/M/G suffixes (binary units).
Both bounds are required. Multiply by integer FACTOR >= 2 (default 2) while <= END.
Do not mix ranges with positional sizes. Only -g 1 (one GPU per MPI process).
-w >= 0 and -n >= 1 override WARMUP_ITERS/ITERS (defaults 20/100).
NP selects MPI rank count. Bytes mean AG input, AR full tensor, RS output shard.
No fixed wall-clock timeout is imposed on the MPI run.
USAGE
    exit 0
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MPI_HOME="${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.7rc1}"
EXTERNAL_NCCL_DIR="${EXTERNAL_NCCL_DIR:-/home/yangz/nfs/zhongjie/nccl}"
NP="${NP:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
UCCL_GPU_DRIVEN_BACKEND="${UCCL_GPU_DRIVEN_BACKEND:-cuda_ipc}"
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
if [[ "${UCCL_GPU_DRIVEN_BACKEND}" == host ]]; then
  export MSCCLPP_NCCL_HOST_ALLGATHER="${MSCCLPP_NCCL_HOST_ALLGATHER:-1}"
fi
export NCCL_BASELINE_LIB
export WARMUP_ITERS ITERS
export LD_LIBRARY_PATH="${PROJECT_DIR}/nccl/build:${PROJECT_DIR}/build:${CUDA_HOME:-/usr/local/cuda}/lib64:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Use the executable's parser before MPI/CUDA initialization and record CLI overrides.
BENCH_CONFIG="$("${PROJECT_DIR}/nccl/build/device_collectives_bench" --print-config "$@")"
read -r WARMUP_ITERS ITERS SELECTED_COLLECTIVE <<<"${BENCH_CONFIG}"

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
  MSCCLPP_SOCKET_IFNAME MSCCLPP_HCA_DEVICES MSCCLPP_NCCL_HOST_ALLGATHER \
  MSCCLPP_NCCL_HOST_ALLGATHER_MAP_SLAB MSCCLPP_NCCL_HOST_ALLGATHER_CHUNK_BYTES \
  MSCCLPP_NCCL_HOST_ALLGATHER_MIN_BYTES MSCCLPP_NCCL_HOST_ALLGATHER_KERNEL_MAX_BYTES \
  MSCCLPP_NCCL_HOST_ALLGATHER_COOP_MAX_BYTES MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC; do
  if [[ -n "${!variable:-}" ]]; then
    MPI_ARGS+=(-x "${variable}")
  fi
done

# RS and AllReduce policy must match on remote MPI ranks as well as local ranks;
# the benchmark's own knobs are forwarded so every rank launches the same way.
while IFS= read -r variable; do
  case "${variable}" in
    MSCCLPP_NCCL_RS_* | MSCCLPP_NCCL_AR_* | MSCCLPP_NCCL_2RANK_RING_* | \
    UCCL_GPU_DRIVEN_BENCH_*)
      MPI_ARGS+=(-x "${variable}")
      ;;
  esac
done < <(compgen -e)

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
  printf -- '- Collective: `%s`\n' "${SELECTED_COLLECTIVE}"
  printf -- '- Backend: `%s`\n' "${UCCL_GPU_DRIVEN_BACKEND}"
  printf -- '- Ranks: `%s`\n' "${NP}"
  printf -- '- CUDA devices per node: `%s`\n' "${CUDA_VISIBLE_DEVICES}"
  printf -- '- Warmup iterations: `%s`\n' "${WARMUP_ITERS}"
  printf -- '- Measured iterations: `%s`\n' "${ITERS}"
  printf -- '- Arguments:'
  printf ' %q' "$@"
  printf '\n'
  printf -- '- NCCL baseline: `%s`\n' "${NCCL_BASELINE_LIB}"
} >"${RESULT_FILE}"

awk -v selected="${SELECTED_COLLECTIVE}" '
  # A result group starts with "<collective> bytes_per_rank=N" and spans the
  # following key=value lines up to the one carrying avg_speedup_e2e. Other lines
  # that merely start with a collective name (selected_backend, path notices) never
  # reach the speedup line, so they are dropped when the next group header resets.
  /^(allgather|allreduce|reducescatter)[[:space:]]+bytes_per_rank=/ {
    current = $1
    delete pending
    collecting = 1
  }
  collecting {
    for (i = 1; i <= NF; ++i) {
      position = index($i, "=")
      if (!position) continue
      pending[substr($i, 1, position - 1)] = substr($i, position + 1)
    }
    if ($0 ~ /avg_speedup_e2e=/) {
      row = ++count[current]
      for (key in pending) value[current, row, key] = pending[key]
      collecting = 0
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
      if (selected != "all" && selected != collective) continue
      printf "\n## %s\n\n", title[collective]
      # Rows that report the selected path (ReduceScatter) get Path and Options
      # columns; rows that report the selected backend (AllGather) a Backend column.
      hasPath = 0
      hasBackend = 0
      for (row = 1; row <= count[collective]; ++row) {
        if (value[collective, row, "path"] != "") hasPath = 1
        if (value[collective, row, "selected_backend"] != "") hasBackend = 1
      }
      header = "| Bytes per rank |"
      divider = "|---:|"
      if (hasPath) {
        header = header " Path | Options |"
        divider = divider "---|---|"
      }
      if (hasBackend) {
        header = header " Backend |"
        divider = divider "---|"
      }
      print header " GPU avg device (us) | GPU avg E2E (us) | NCCL avg device (us) | NCCL avg E2E (us) | Avg E2E speedup |"
      print divider "---:|---:|---:|---:|---:|"
      for (row = 1; row <= count[collective]; ++row) {
        line = "| " value[collective, row, "bytes_per_rank"] " |"
        if (hasPath)
          line = line " " value[collective, row, "path"] " | " value[collective, row, "opts"] " |"
        if (hasBackend)
          line = line " " value[collective, row, "selected_backend"] " |"
        printf "%s %s | %s | %s | %s | %s |\n", line, \
          value[collective, row, "gpu_avg_device_us"], \
          value[collective, row, "gpu_avg_e2e_us"], \
          value[collective, row, "nccl_avg_device_us"], \
          value[collective, row, "nccl_avg_e2e_us"], \
          value[collective, row, "avg_speedup_e2e"]
      }
    }
  }
' "${RAW_OUTPUT}" >>"${RESULT_FILE}"

echo "[gpu-driven-benchmark] Markdown result: ${RESULT_FILE}" >&2

if [[ "${SELECTED_COLLECTIVE}" == allreduce || "${SELECTED_COLLECTIVE}" == all ]]; then
  {
    printf '\n## AllReduce paths\n\n```text\n'
    awk '/^allreduce / && /path=/' "${RAW_OUTPUT}"
    printf '```\n'
  } >>"${RESULT_FILE}"
fi

# The selected path is part of every ReduceScatter result row; keep only the
# preflight coverage notes (for example generic int/min/max being unavailable).
if [[ "${SELECTED_COLLECTIVE}" == reducescatter || "${SELECTED_COLLECTIVE}" == all ]] \
    && grep -q '^reducescatter preflight coverage=' "${RAW_OUTPUT}"; then
  {
    printf '\n## ReduceScatter notes\n\n```text\n'
    grep '^reducescatter preflight coverage=' "${RAW_OUTPUT}"
    printf '```\n'
  } >>"${RESULT_FILE}"
fi
