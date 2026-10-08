#!/usr/bin/env bash
# Runs the GPU-driven ReduceScatter benchmark (collective/lite/gpu_driven/
# benchmark.sh) for the supported topologies and compares it with NCCL.
#
# USAGE
#   ./run_reducescatter.sh [TARGET...]
#
#   No TARGET runs everything, in this order: 1n2g, 2n1g, 2n2g, 2n4g.
#   TARGET is GROUP or GROUP:RANGE.
#
#   GROUP (nodes x GPUs per node):
#     1n2g            1 node, 2 GPUs: host and cuda_ipc backends, one after the other
#     1n2g-host       1 node, 2 GPUs: host backend only
#     1n2g-cuda_ipc   1 node, 2 GPUs: cuda_ipc backend only
#                     (sets MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC=1)
#     2n1g            2 nodes, 1 GPU each  (NP=2, CUDA_VISIBLE_DEVICES=0)
#     2n2g            2 nodes, 2 GPUs each (NP=4, CUDA_VISIBLE_DEVICES=0,1)
#     2n4g            2 nodes, 4 GPUs each (NP=8, CUDA_VISIBLE_DEVICES=0,1,2,3)
#     The multi-node groups use the host backend and HOSTS (see below).
#
#   RANGE (bytes per rank = the output shard, -f 2 doubling):
#     small   128B..1M, 100 warmups, 1000 iterations
#     large   1M..1G,   20 warmups,  50 iterations
#     omitted: both ranges, small first.
#
# EXAMPLES
#   ./run_reducescatter.sh                         # everything
#   ./run_reducescatter.sh 2n2g                    # 2n2g, small and large
#   ./run_reducescatter.sh 2n2g:large              # 2n2g, large range only
#   ./run_reducescatter.sh 2n2g:large 2n4g:small   # several targets
#   ./run_reducescatter.sh 1n2g-cuda_ipc           # one backend of 1n2g
#
# RESULTS
#   .tmp/gpu-driven-benchmarks/reducescatter/<name>.md under this directory,
#   e.g. gpu-driven-2n2g-large.md, gpu-driven-host-1n2g-small.md,
#   gpu-driven-cuda_ipc-1n2g-large.md. An existing file of the same name is
#   overwritten. The first failing run stops the script.
#
# ENVIRONMENT (defaults for this machine; anything already set wins)
#   MPI_HOME, CUDA_HOME, MSCCLPP_SOCKET_IFNAME, MSCCLPP_HCA_DEVICES,
#   NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_BASELINE_LIB, UCCL_GPU_DRIVEN_BACKEND
#   HOSTS    the two node addresses for the multi-node groups
#            (default 10.10.55.1,10.10.55.2), e.g.
#              HOSTS=10.0.0.1,10.0.0.2 ./run_reducescatter.sh 2n2g
#   Other variables (UCCL_GPU_DRIVEN_RS_TRACE, MSCCLPP_NCCL_RS_*, ...) are
#   passed on to benchmark.sh when set in the calling shell.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export MPI_HOME="${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.7rc1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MSCCLPP_SOCKET_IFNAME="${MSCCLPP_SOCKET_IFNAME:-ibp55s0f0}"
export MSCCLPP_HCA_DEVICES="${MSCCLPP_HCA_DEVICES:-mlx5_0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ibp55s0f0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_BASELINE_LIB="${NCCL_BASELINE_LIB:-/home/yangz/nfs/zhongjie/nccl/build/lib/libnccl.so}"
export UCCL_GPU_DRIVEN_BACKEND="${UCCL_GPU_DRIVEN_BACKEND:-host}"
HOSTS="${HOSTS:-10.10.55.1,10.10.55.2}"

RESULT_SUBDIR="$PWD/.tmp/gpu-driven-benchmarks/reducescatter"
mkdir -p "${RESULT_SUBDIR}"

# run NAME NP DEVICES HOSTS BACKEND BEGIN END WARMUPS ITERS
# HOSTS "-" means a single node.
run() {
  local name="$1" np="$2" devices="$3" hosts="$4" backend="$5"
  local begin="$6" end="$7" warmups="$8" iters="$9"
  echo "=== ${name} (NP=${np}, backend=${backend}, ${begin}..${end}) ==="
  local -a env_args=(
    NP="${np}"
    CUDA_VISIBLE_DEVICES="${devices}"
    UCCL_GPU_DRIVEN_BACKEND="${backend}"
    RESULT_FILE="${RESULT_SUBDIR}/${name}.md"
  )
  [[ "${hosts}" != "-" ]] && env_args+=(HOSTS="${hosts}")
  [[ "${backend}" == cuda_ipc ]] && env_args+=(MSCCLPP_NCCL_CUDAIPC_EVENT_SYNC=1)
  env "${env_args[@]}" \
    bash collective/lite/gpu_driven/benchmark.sh \
      -c reducescatter -g 1 -b "${begin}" -e "${end}" -f 2 \
      -w "${warmups}" -n "${iters}"
  echo
}

# sizes RANGE NAME NP DEVICES HOSTS BACKEND: RANGE is small, large or both.
sizes() {
  local range="$1"
  shift
  if [[ "${range}" == both || "${range}" == small ]]; then
    run "$1-small" "$2" "$3" "$4" "$5" 128B 1M 100 1000
  fi
  if [[ "${range}" == both || "${range}" == large ]]; then
    run "$1-large" "$2" "$3" "$4" "$5" 1M 1G 20 50
  fi
}

group() {
  local name="$1" range="$2"
  case "${name}" in
    1n2g)
      group 1n2g-host "${range}"
      group 1n2g-cuda_ipc "${range}"
      ;;
    1n2g-host)      sizes "${range}" gpu-driven-host-1n2g     2 0,1 - host ;;
    1n2g-cuda_ipc)  sizes "${range}" gpu-driven-cuda_ipc-1n2g 2 0,1 - cuda_ipc ;;
    2n1g) sizes "${range}" gpu-driven-2n1g 2 0       "${HOSTS}" host ;;
    2n2g) sizes "${range}" gpu-driven-2n2g 4 0,1     "${HOSTS}" host ;;
    2n4g) sizes "${range}" gpu-driven-2n4g 8 0,1,2,3 "${HOSTS}" host ;;
    *)
      echo "error: unknown group '${name}'" >&2
      echo "use 1n2g, 1n2g-host, 1n2g-cuda_ipc, 2n1g, 2n2g or 2n4g" >&2
      exit 1
      ;;
  esac
}

if [[ $# -eq 0 ]]; then
  set -- 1n2g 2n1g 2n2g 2n4g
fi

# Validate every target before starting a long run.
for target in "$@"; do
  range="${target#*:}"
  [[ "${target}" == *:* ]] || range=both
  case "${range}" in
    both | small | large) ;;
    *)
      echo "error: unknown range '${range}' in '${target}' (use small or large)" >&2
      exit 1
      ;;
  esac
done

for target in "$@"; do
  name="${target%%:*}"
  range="${target#*:}"
  [[ "${target}" == *:* ]] || range=both
  group "${name}" "${range}"
done
