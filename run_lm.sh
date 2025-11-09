#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 -C 'a100|h100'
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --job-name=sp
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# set -x
module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /vast/hvp2011/h_envs/verl
conda env list
nvidia-smi
# unset PYTHONPATH

cd /scratch/hvp2011/implement/self_play
export HYDRA_FULL_ERROR=1 
# pip install torch transformers datasets 
# pip install wandb --force-reinstall
# pip install packaging 
# pip install -e /scratch/hvp2011/implement/self_play/verl
# pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0", "vllm>=0.7.3,<=0.9.1"
# pip install qwen_vl_utils
# pip install math-verify
# pip install -e /scratch/hvp2011/implement/self_play/MathRuler
# pip install flash-attn --no-build-isolation
# pip install uvloop==0.21.0
export WANDB_API_KEY="8e206762de4253cfbf4fd8db344147e420be8b78"
export HYDRA_FULL_ERROR=1 
unset ROCR_VISIBLE_DEVICES
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
# python llava-cot.py 
# python3 data/mathvista.py
# python3 data/process.py
export DEBUG_MODE=true


# ⚠️ This makes all newly created files 666 -> 666, dirs 777 -> 777

chmod 777 -R outputs/
chmod 777 -R wandb/
chmod 777 -R checkpoints/


prompt=open_thoughts

prompt=deepseek

system_prompt="./prompts/$prompt.txt"
custom_reward_function=verl/verl/utils/reward_score/math_verify.py


adv_estimator=grpo
adv_estimator=reinforce_plus_plus

model=Qwen2.5-3B-Instruct

use_dpo_loss=True
# use_dpo_loss=False


experiment_name="open_thoughts-hendrycks_math-$model-$prompt-$adv_estimator-dpo-$use_dpo_loss-$USER-0"
# experiment_name=test
export LOG_PATH=outputs/$experiment_name.log
CUDA_VISIBLE_DEVICES=0 python -m verl.trainer.main_rgpo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=data/open_thoughts_114k_math_correct.parquet \
    data.val_files=data/hendrycks_math_test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.use_dpo_loss=$use_dpo_loss \
    actor_rollout_ref.actor.dpo_coef=.1 \
    actor_rollout_ref.actor.dpo_beta=0.1 \
    actor_rollout_ref.actor.dpo_loss_type=simpo \
    actor_rollout_ref.actor.simpo_gamma=0.5 \
    actor_rollout_ref.actor.dpo_label_smoothing=0.0 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key="problem" \
    data.response_key=answer \
    "data.system_prompt='${system_prompt}'" \
    reward_model.enable=False \
    custom_reward_function.path=$custom_reward_function \
    actor_rollout_ref.model.path=Qwen/$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=Yes \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rgpo-lm-0' \
    trainer.experiment_name=$experiment_name \
    trainer.validation_data_dir=outputs/$experiment_name \
    trainer.rollout_data_dir=outputs/$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@

chmod 777 -R outputs/
chmod 777 -R wandb/
chmod 777 -R checkpoints/
