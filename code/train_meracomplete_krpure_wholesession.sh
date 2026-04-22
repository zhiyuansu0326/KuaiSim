mkdir -p output

mkdir -p output/Kuairand_Pure/
mkdir -p output/Kuairand_Pure/agents/

output_path="output/Kuairand_Pure/"
log_name="user_KRMBUserResponse_lr0.0001_reg0_nlayer2"

# environment args
ENV_CLASS='KREnvironment_WholeSession_GPU'
MAX_STEP=20
SLATE_SIZE=20
EP_BS=32
RHO=0.2
INTRA_SLATE_METRIC=EILD
TEMPER_DISCOUNT=2.0

# MERAComplete classes
POLICY_CLASS='MERACompletePolicy'
CRITIC_CLASS='MERACompleteCritic'
BUFFER_CLASS='MERACompleteBuffer'
AGENT_CLASS='MERAComplete'

# MERAComplete policy args
MERAC_K=100
MERAC_TAU=0.7
MERAC_SINKHORN_ITER=20
MERAC_GUMBEL=1.0

# MERAComplete critic args
MERAC_CRITIC_HIDDEN="256 64"
MERAC_CRITIC_DROP=0.1
MERAC_CRITIC_HEAD=4
MERAC_CRITIC_D_FWD=64
MERAC_CRITIC_N_LAYER=2

# buffer args
BUFFER_SIZE=100000

# agent args
GAMMA=0.9
REWARD_FUNC='get_immediate_reward'
N_ITER=20000
START_STEP=100
INITEP=0.01
ELBOW=0.1
EXPLORE_RATE=1.0
BS=128
MERAC_ACTOR_REG=0.0
MERAC_USE_LEGACY_DONE_MASK=1

for REG in 0.00001
do
    for INITEP in 0.01
    do
        for CRITIC_LR in 0.001
        do
            for ACTOR_LR in 0.0001
            do
                for SEED in 11
                do
                    file_key=${AGENT_CLASS}_${POLICY_CLASS}_K${MERAC_K}_tau${MERAC_TAU}_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}

                    mkdir -p ${output_path}agents/${file_key}/

                    python train_actor_critic.py\
                        --env_class ${ENV_CLASS}\
                        --policy_class ${POLICY_CLASS}\
                        --critic_class ${CRITIC_CLASS}\
                        --buffer_class ${BUFFER_CLASS}\
                        --agent_class ${AGENT_CLASS}\
                        --seed ${SEED}\
                        --cuda 0\
                        --max_step_per_episode ${MAX_STEP}\
                        --initial_temper ${MAX_STEP}\
                        --uirm_log_path ${output_path}env/log/${log_name}.model.log\
                        --slate_size ${SLATE_SIZE}\
                        --episode_batch_size ${EP_BS}\
                        --item_correlation ${RHO}\
                            --intra_slate_metric ${INTRA_SLATE_METRIC}\
                        --temper_discount ${TEMPER_DISCOUNT}\
                        --merac_shortlist_size ${MERAC_K}\
                        --merac_policy_hidden 256 64\
                        --merac_tau ${MERAC_TAU}\
                        --merac_sinkhorn_iter ${MERAC_SINKHORN_ITER}\
                        --merac_gumbel_noise ${MERAC_GUMBEL}\
                        --state_user_latent_dim 16\
                        --state_item_latent_dim 16\
                        --state_transformer_enc_dim 32\
                        --state_transformer_n_head 4\
                        --state_transformer_d_forward 64\
                        --state_transformer_n_layer 3\
                        --state_dropout_rate 0.1\
                        --merac_critic_hidden_dims ${MERAC_CRITIC_HIDDEN}\
                        --merac_critic_dropout_rate ${MERAC_CRITIC_DROP}\
                        --merac_critic_transformer_n_head ${MERAC_CRITIC_HEAD}\
                        --merac_critic_transformer_d_forward ${MERAC_CRITIC_D_FWD}\
                        --merac_critic_transformer_n_layer ${MERAC_CRITIC_N_LAYER}\
                        --buffer_size ${BUFFER_SIZE}\
                        --gamma ${GAMMA}\
                        --reward_func ${REWARD_FUNC}\
                        --n_iter ${N_ITER}\
                        --train_every_n_step 1\
                        --start_policy_train_at_step ${START_STEP}\
                        --initial_epsilon ${INITEP}\
                        --final_epsilon ${INITEP}\
                        --elbow_epsilon ${ELBOW}\
                        --explore_rate ${EXPLORE_RATE}\
                        --check_episode 10\
                        --save_episode 200\
                        --save_path ${output_path}agents/${file_key}/model\
                        --actor_lr ${ACTOR_LR}\
                        --actor_decay ${REG}\
                        --batch_size ${BS}\
                        --critic_lr ${CRITIC_LR}\
                        --critic_decay ${REG}\
                        --target_mitigate_coef 0.01\
                        --merac_actor_reg ${MERAC_ACTOR_REG}\
                        --merac_use_legacy_done_mask ${MERAC_USE_LEGACY_DONE_MASK}\
                        > ${output_path}agents/${file_key}/log
                done
            done
        done
    done
done
