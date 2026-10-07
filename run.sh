#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2 
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200GB
#SBATCH --job-name=mint
#SBATCH --output=
#SBATCH --error=
##SBATCH --account=
#SBATCH --account=

module purge
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
cd /scratch/hvp2011/implement/rgpo/
conda activate /scratch/hvp2011/envs/rgpo/
conda env list

nvidia-smi
# unset PYTHONPATH

export HYDRA_FULL_ERROR=1 

export HF_TOKEN=
export WANDB_API_KEY=
export HYDRA_FULL_ERROR=1 
unset ROCR_VISIBLE_DEVICES

ENGINE=${1:-vllm}
export DEBUG_MODE=true

export WANDB_DIR="outputs/wandb/$USER/"
system_prompt="You are a helpful assistant. When responding to any user query, first provide a clear, step-by-step thinking trace explaining your reasoning process. Then, output only the final answer enclosed <answer> </answer> tags. Please strictly follow the format."

# prompt=optimal
# system_prompt="You first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \boxed{}."
# custom_reward_function=./verl/verl/utils/reward_score/math_ruler.py

prompt=mint-long
# prompt=mint-short
system_prompt="./prompts/$prompt.txt"
custom_reward_function=./verl/verl/utils/reward_score/math_mint.py


adv_estimator=grpo
# adv_estimator=reinforce_plus_plus

model=Qwen2-VL-2B-Instruct

refine_coef=${2:-1.}
use_refine=${3:-False}
full_hint=${4:-False}
use_sft=${5:-False}
experiment_name="$USER-mint-mathvista-$model-prompt-$prompt-adv_estimator-$adv_estimator-refine_coef-$refine_coef-refine-$use_refine-hint-$full_hint-sft-$use_sft"



export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
export N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Using GPU" "$CUDA_VISIBLE_DEVICES"

batch_size_per_gpu=32
log_prob_batch_size_per_gpu=64

gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
if echo "$gpu_type" | grep -qi "h200"; then
    batch_size_per_gpu=$((batch_size_per_gpu * 2))
    log_prob_batch_size_per_gpu=$((log_prob_batch_size_per_gpu * 2))

fi
echo "Detected GPU type: $gpu_type"
echo "Using ppo micro batch size per GPU: $batch_size_per_gpu"
echo "Using log prob micro batch size per GPU: $log_prob_batch_size_per_gpu"

set -x
# ray stop --force || true
export LOG_PATH=outputs/$experiment_name.log
python -m verl.trainer.main_rgpo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=data/mint_cot_r1_1024.parquet \
    data.val_files=data/mathvista_mini.parquet \
    data.return_raw_chat=True \
    data.train_batch_size=128 \
    data.max_prompt_length=768 \
    data.max_hint_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.use_full_solution_hint=$full_hint \
    actor_rollout_ref.actor.max_reprompt_len=2560 \
    actor_rollout_ref.actor.refine_coef=$refine_coef \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.use_dpo_loss=False \
    actor_rollout_ref.actor.use_refine=$use_refine \
    actor_rollout_ref.actor.use_sft=$use_sft \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.prompt_key="problem" \
    data.response_key=answer \
    data.system_prompt="$system_prompt" \
    data.return_multi_modal_inputs=True \
    reward_model.enable=False \
    custom_reward_function.path=$custom_reward_function \
    actor_rollout_ref.model.path=Qwen/$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=3584 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rgpo-mint-6' \
    trainer.experiment_name=$experiment_name \
    trainer.validation_data_dir=outputs/$experiment_name \
    trainer.rollout_data_dir=outputs/$experiment_name \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=10 \
    trainer.total_epochs=1

