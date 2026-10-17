#!/usr/bin/env bash
# Local launcher for this example. These settings affect only this process and
# do not change the system compiler, CUDA symlinks, or the Conda environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ubuntu 24.04's headers and NVCC can disagree when NVCC auto-detects a host
# compiler. Pin the already-installed compiler for this extension build.
export CC="${CC:-/usr/bin/gcc-9}"
export CXX="${CXX:-/usr/bin/g++-9}"

# NVIDIA A40 has compute capability 8.6. Avoid compiling unused GPU targets.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

exec python elementwise.py "$@"
