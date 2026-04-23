#!/usr/bin/env bash
set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kuaisim310

REPO_ROOT=~/KuaiSim
CODE_DIR="$REPO_ROOT/code"
ML1M_UIRM_KEY="user_KRMBUserResponse_lr0.0001_reg0.01_nlayer2.model"

prepare_dataset_output() {
  local dataset="$1"
  local canonical="$REPO_ROOT/output/$dataset"
  local legacy="$CODE_DIR/output/$dataset"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"

  mkdir -p "$REPO_ROOT/output" "$CODE_DIR/output"

  if [ -L "$legacy" ]; then
    rm -f "$legacy"
  elif [ -e "$legacy" ]; then
    if [ ! -e "$canonical" ]; then
      mv "$legacy" "$canonical"
    else
      mkdir -p "$canonical"
      cp -an "$legacy"/. "$canonical"/ 2>/dev/null || true
      mv "$legacy" "${legacy}.bak.${stamp}"
    fi
  fi

  mkdir -p "$canonical"
  ln -s "$canonical" "$legacy"
}

prepare_dataset_output "Kuairand_Pure"
prepare_dataset_output "ml_1m"

ml1m_uirm_ready() {
  local env_root
  for env_root in "$REPO_ROOT/output/ml_1m/env" "$CODE_DIR/output/ml_1m/env"; do
    if [ -f "$env_root/log/${ML1M_UIRM_KEY}.log" ] && [ -f "$env_root/${ML1M_UIRM_KEY}.checkpoint" ]; then
      return 0
    fi
  done
  return 1
}

cd "$REPO_ROOT"

mkdir -p run_logs
LOG="run_logs/run_12_and_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "Start: $(date)"
python -V
python -c "import torch, numpy; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(numpy.__version__)"

echo "Preprocess ML1M if needed"
if [ ! -f dataset/ml-1m/log_session.csv ]; then
  python code/preprocess/preprocess_ml1m.py --input_dir dataset/ml-1m --output_dir dataset/ml-1m
fi

echo "Train ML1M user response model if needed"
if ! ml1m_uirm_ready; then
  bash code/run_multibehavior_ml.sh
fi

echo "Run KUAI 6 models"
cd "$CODE_DIR"
bash train_ddpg_krpure_wholesession.sh
bash train_TD3_krpure_wholesession.sh
bash train_A2C_krpure_wholesession.sh
bash train_hac_krpure_wholesession.sh
bash train_mera_krpure_wholesession.sh
bash train_meracomplete_krpure_wholesession.sh

echo "Run ML1M 6 models"
cd "$REPO_ROOT"
bash code/train_ddpg_krpure_wholesession_ml.sh
bash code/train_TD3_krpure_wholesession_ml.sh
bash code/train_A2C_krpure_wholesession_ml.sh
bash code/train_hac_krpure_wholesession_ml.sh
bash code/train_mera_krpure_wholesession_ml.sh
bash code/train_meracomplete_krpure_wholesession_ml.sh

echo "Evaluate KUAI reports"
cd "$REPO_ROOT"
python code/evaluate_benchmarks.py --task whole \
  --baseline DDPG="output/Kuairand_Pure/agents/DDPG_*/model.report" \
  --baseline TD3="output/Kuairand_Pure/agents/TD3_*/model.report" \
  --baseline A2C="output/Kuairand_Pure/agents/A2C_*/model.report" \
  --baseline HAC="output/Kuairand_Pure/agents/HAC_*/model.report" \
  --baseline MERA="output/Kuairand_Pure/agents/MERA_*/model.report" \
  --baseline MERAComplete="output/Kuairand_Pure/agents/MERAComplete_*/model.report" \
  --select auto \
  --save_csv output/Kuairand_Pure/eval/whole_report_metrics.csv

echo "Validate KUAI checkpoints"
python code/validate_benchmarks.py --task whole \
  --baseline DDPG="output/Kuairand_Pure/agents/DDPG_*" \
  --baseline TD3="output/Kuairand_Pure/agents/TD3_*" \
  --baseline A2C="output/Kuairand_Pure/agents/A2C_*" \
  --baseline HAC="output/Kuairand_Pure/agents/HAC_*" \
  --baseline MERA="output/Kuairand_Pure/agents/MERA_*" \
  --baseline MERAComplete="output/Kuairand_Pure/agents/MERAComplete_*" \
  --eval_steps 1000 \
  --device cuda:0

echo "Evaluate ML1M reports"
cd "$REPO_ROOT"
python code/evaluate_benchmarks.py --task whole \
  --baseline DDPG="output/ml_1m/agents/ddpg_*/model.report" \
  --baseline TD3="output/ml_1m/agents/TD3_*/model.report" \
  --baseline A2C="output/ml_1m/agents/A2C_*/model.report" \
  --baseline HAC="output/ml_1m/agents/HAC_*/model.report" \
  --baseline MERA="output/ml_1m/agents/MERA_*/model.report" \
  --baseline MERAComplete="output/ml_1m/agents/MERAComplete_*/model.report" \
  --select auto \
  --save_csv output/ml_1m/eval/whole_report_metrics.csv

echo "Validate ML1M checkpoints"
python code/validate_benchmarks.py --task whole \
  --baseline DDPG="output/ml_1m/agents/ddpg_*" \
  --baseline TD3="output/ml_1m/agents/TD3_*" \
  --baseline A2C="output/ml_1m/agents/A2C_*" \
  --baseline HAC="output/ml_1m/agents/HAC_*" \
  --baseline MERA="output/ml_1m/agents/MERA_*" \
  --baseline MERAComplete="output/ml_1m/agents/MERAComplete_*" \
  --eval_steps 1000 \
  --device cuda:0

echo "Done: $(date)"
echo "Log saved to: $LOG"
echo "KUAI CSV: ~/KuaiSim/output/Kuairand_Pure/eval/whole_report_metrics.csv"
echo "ML1M CSV: ~/KuaiSim/output/ml_1m/eval/whole_report_metrics.csv"
