#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "This feature must be installed as root."
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential cmake libopenblas-dev

install_dir=/opt/llama.cpp
rm -rf "$install_dir"
git clone --branch b9009 https://github.com/ggml-org/llama.cpp.git "$install_dir"

feature_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install -m 0755 "$feature_dir/llama-cpp-build.sh" /usr/local/bin/llama-cpp-build.sh

echo "llama.cpp source cloned to /opt/llama.cpp; build will run after the container starts."
