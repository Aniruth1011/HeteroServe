#!/usr/bin/env bash
#SBATCH --job-name=heteroserve-capability
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gpus-per-task=a40:1
#SBATCH --signal=B:TERM@60

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${repo_root}/results/raw/capabilities"
output_path="${output_dir}/gpu_debug_${SLURM_JOB_ID}.json"

mkdir -p "${output_dir}"
cd "${repo_root}"

module purge
module load python/3.11.9

python phase0_collect_capabilities.py --output "${output_path}"
python -m json.tool "${output_path}" >/dev/null

echo "Capability evidence: ${output_path}"
