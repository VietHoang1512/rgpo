#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2 -C 'a100|h100'
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --job-name=sp
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# set -x
module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /vast/hvp2011/h_envs/self_play
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

# CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=data/mathvista_mini.parquet \
#     data.val_files=data/mathvista_mini.parquet \
#     data.train_batch_size=512 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=2048 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     data.image_key=image \
#     data.prompt_key=hint \
#     'data.system_prompt="You are a helpful assistant. When responding to any user query, first provide a clear, step-by-step thinking trace explaining your reasoning process. Then, output only the final answer enclosed <answer> </answer> tags. Please strictly follow the format."' \
#     reward_model.enable=False \
#     custom_reward_function.path=verl/verl/utils/reward_score/math_verify_custom.py \
#     actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
#     actor_rollout_ref.actor.optim.lr=3e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.use_fused_kernels=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.model.lora_rank=8 \
#     actor_rollout_ref.model.lora_alpha=32 \
#     actor_rollout_ref.model.target_modules=all-linear \
#     actor_rollout_ref.model.exclude_modules='.*visual.*' \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.kl_loss_coef=0.0 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=$ENGINE \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.99 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=True \
#     actor_rollout_ref.rollout.n=4 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","wandb"]' \
#     trainer.project_name='verl' \
#     trainer.experiment_name=$experiment_name \
#     trainer.validation_data_dir=ouputs/$experiment_name \
#     trainer.rollout_data_dir=ouputs/$experiment_name \
#     trainer.n_gpus_per_node=2 \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=5 $@

# cd /scratch/hvp2011/implement/self_play/vlaa_data/images
# python ../process.py
system_prompt="You are a helpful assistant. When responding to any user query, first provide a clear, step-by-step thinking trace explaining your reasoning process. Then, output only the final answer enclosed <answer> </answer> tags. Please strictly follow the format."

prompt=optimal
system_prompt="You first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \boxed{}."
custom_reward_function=/scratch/hvp2011/implement/self_play/verl/verl/utils/reward_score/math_ruler.py

prompt=suboptimal
system_prompt="/scratch/hvp2011/implement/self_play/prompts/mint.txt"
custom_reward_function=/scratch/hvp2011/implement/self_play/verl/verl/utils/reward_score/math_mint.py



adv_estimator=grpo
# adv_estimator=reinforce_plus_plus

experiment_name="qwen25-3b-mint-mathvista-$prompt-$adv_estimator-$USER-0"
# experiment_name=test
export LOG_PATH=outputs/$experiment_name.log
CUDA_VISIBLE_DEVICES=0 python -m verl.trainer.main_rgpo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=data/mint_cot_r1.parquet \
    data.val_files=data/mathvista_mini.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=image \
    data.prompt_key="problem" \
    data.response_key=answer \
    "data.system_prompt='${system_prompt}'" \
    reward_model.enable=False \
    custom_reward_function.path=$custom_reward_function \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
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
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    trainer.project_name='rgpo' \
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

# experiment_name="qwen25_vl_32b_mint_mathvista_optimal_grpo_6"
# # experiment_name=test
# export LOG_PATH=outputs/$experiment_name.log
# CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=data/mint_cot_r1.parquet\
#     data.val_files=data/mathvista_mini.parquet \
#     data.train_batch_size=128 \
#     data.max_prompt_length=2048 \
#     data.max_response_length=4096 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=128 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     data.image_key=image \
#     data.prompt_key="problem" \
#     "data.system_prompt='${system_prompt}'" \
#     reward_model.enable=False \
#     custom_reward_function.path=/scratch/hvp2011/implement/self_play/verl/verl/utils/reward_score/math_ruler.py \
#     actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-32B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.use_fused_kernels=True \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.kl_loss_coef=0.0 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.name=$ENGINE \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=True \
#     actor_rollout_ref.rollout.n=4 \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
#     actor_rollout_ref.model.enable_activation_offload=True \
#     actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
#     actor_rollout_ref.actor.entropy_checkpointing=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","wandb"]' \
#     trainer.project_name='sp' \
#     trainer.experiment_name=$experiment_name \
#     trainer.validation_data_dir=outputs/$experiment_name \
#     trainer.rollout_data_dir=outputs/$experiment_name \
#     trainer.n_gpus_per_node=2 \
#     trainer.nnodes=1 \
#     trainer.save_freq=500 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=1 $@