#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

usage() {
    cat <<'EOF'
Usage: bash Scripts/Evaluate.sh TASK [OPTIONS]

Tasks:
  all         Empirical research questions and appendix
  rq1_1       Accuracy and data budget (RQ1.1 / RQ2.2)
  rq1_2       RLC physics prior and component ablation
  rq1_3       Fractional systems: NFDE and NODE
  rq2_3       Saved runtime measurements
  appendix    Dataset windows, hyperparameters, epochs and full tables
  demos       Illustrative ILT and DLT/FFLT examples (RQ2.1 and Part II)

Examples:
  bash Scripts/Evaluate.sh all --dry-run
  bash Scripts/Evaluate.sh rq1_1
  bash Scripts/Evaluate.sh appendix
  bash Scripts/Evaluate.sh demos --dry-run

Edit paths, models and seeds at the top of the Python scripts.
PYTHON_BIN selects the environment. This launcher does not start training.
EOF
}

case "${1:-help}" in
    help|-h|--help) usage; exit 0 ;;
    all) modules=(Evaluation.RQs.Evaluate_RQ1_1_RQ2_2 Evaluation.RQs.Evaluate_RQ1_2 Evaluation.RQs.Evaluate_RQ1_3 Evaluation.RQs.Evaluate_RQ2_3 Evaluation.Supplementary.Evaluate_Appendix) ;;
    rq1_1|rq2_2) modules=(Evaluation.RQs.Evaluate_RQ1_1_RQ2_2) ;;
    rq1_2) modules=(Evaluation.RQs.Evaluate_RQ1_2) ;;
    rq1_3) modules=(Evaluation.RQs.Evaluate_RQ1_3) ;;
    rq2_3) modules=(Evaluation.RQs.Evaluate_RQ2_3) ;;
    appendix) modules=(Evaluation.Supplementary.Evaluate_Appendix) ;;
    demos|demonstrations|rq2_1) modules=(Demonstrations.Evaluate_RQ2_1) ;;
    *) usage >&2; exit 2 ;;
esac
shift

for module in "${modules[@]}"; do
    "$PYTHON_BIN" -m "Scripts.${module}" "$@"
done
