SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p output

# RL4RS environment

mkdir -p output/ml_1m/
mkdir -p output/ml_1m/env
mkdir -p output/ml_1m/env/sess_data

# output_path='output/kuairand/'
output_path='output/ml_1m'
# KRMB_MODEL_KEY='user_KRMBUserResponse_MaxOut_lr0.0001_reg0'
KRMB_MODEL_KEY='user_KRMBUserResponse_lr0.0001_reg0.01_nlayer2'

python "${SCRIPT_DIR}/generate_session_data_ml.py"\
    --behavior_model_log_file ${output_path}/env/log/${KRMB_MODEL_KEY}.model.log\
    --data_output_path ${output_path}/env/sess_data/${KRMB_MODEL_KEY}.csv
