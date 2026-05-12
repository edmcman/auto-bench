#!/usr/bin/env bash

export PATH=/usr/local/cuda/bin:$PATH

set -euo pipefail
cd /opt/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --config Release -- -j"$(nproc)"
cmake --install build --prefix /usr
