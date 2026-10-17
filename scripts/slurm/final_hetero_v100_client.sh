#!/usr/bin/env bash
set -euo pipefail
export CLIENT_GPU_TYPE=v100
exec bash "$(dirname "$0")/final_hetero_a40_client.sh"
