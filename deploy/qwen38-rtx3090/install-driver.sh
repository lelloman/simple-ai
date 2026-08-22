#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "run this script with sudo" >&2
  exit 1
fi

. /etc/os-release
if [ "$ID" != ubuntu ] || [ "$VERSION_ID" != 24.04 ]; then
  echo "expected Ubuntu 24.04; found $ID $VERSION_ID" >&2
  exit 1
fi

apt-get update
candidate=$(apt-cache policy nvidia-driver-580 | awk '/Candidate:/ {print $2}')
if [ -z "$candidate" ] || [ "$candidate" = "(none)" ]; then
  echo "nvidia-driver-580 is unavailable from the configured Ubuntu repositories" >&2
  exit 1
fi

apt-get install -y nvidia-driver-580

echo
echo "Installed nvidia-driver-580 $candidate. Reboot the host, then run:"
echo "  nvidia-smi --query-gpu=driver_version,power.limit --format=csv,noheader"
echo "  docker run --rm --gpus all qwen38-27b-3090:latest nvidia-smi"
echo "The existing Snap Docker NVIDIA runtime is intentionally left unchanged."
