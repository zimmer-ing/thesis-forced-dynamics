#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage: bash Scripts/Train.sh TASK [OPTIONS]

Tasks:
  main        Training/Train_Models.py
  data-fraction Training/Train_DataFraction.py
  fractional  Training/Train_NFDE.py
  runtime     Evaluation/Supplementary/Evaluate_Resources.py

Examples:
  bash Scripts/Train.sh main --dry-run
  bash Scripts/Train.sh main
  bash Scripts/Train.sh main --dataset SMDSystem --model LaplaceNet --seeds 42 \
      --hparams-from results/thesis_results/full_data_hpo \
      --results-path results/new_runs/laplace_smd --dry-run
  bash Scripts/Train.sh data-fraction --dry-run
  bash Scripts/Train.sh fractional --dry-run
  bash Scripts/Train.sh runtime --dry-run

Edit datasets, models, seeds and paths at the top of the Python script.
The main task also accepts the selections above; --hparams-from skips HPO.
Each script accepts --help and --dry-run. PYTHON_BIN selects the environment.
EOF
}

case "${1:-help}" in
    help|-h|--help) usage; exit 0 ;;
    main) module="Scripts.Training.Train_Models" ;;
    data-fraction) module="Scripts.Training.Train_DataFraction" ;;
    fractional) module="Scripts.Training.Train_NFDE" ;;
    runtime) module="Scripts.Evaluation.Supplementary.Evaluate_Resources" ;;
    *) usage >&2; exit 2 ;;
esac
shift
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
exec "$PYTHON_BIN" -m "$module" "$@"
