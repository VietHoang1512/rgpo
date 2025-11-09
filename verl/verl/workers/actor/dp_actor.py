# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---- Add near the top (helpers) ----
import torch.nn.functional as F

def get_batch_logps(
    token_logps: torch.Tensor,           # (B, T_resp) per-token log p(y_t | prefix, y_<t)
    labels: torch.LongTensor,            # (B, T_resp) with -100 for padding
    average_log_prob: bool = False,
) -> torch.Tensor:                       # -> (B,)
    mask = (labels != -100).to(token_logps.dtype)  # (B, T_resp)
    seq_sum = (token_logps * mask).sum(dim=-1)     # (B,)
    if average_log_prob:
        lengths = mask.sum(dim=-1).clamp(min=1)    # (B,)
        return seq_sum / lengths
    return seq_sum


def get_batch_logps_from_logits(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (e.g., huggingface CausalLMOutputs `logits`).
                Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for computing the sequence log probabilities. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per sequence. Otherwise, return the sum.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given sequences.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same shape[:-1]")

    # Ensure labels are contiguous and on the same device as logits
    labels = labels.contiguous().to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Calculate per token log probability
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token_logps = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_logps = per_token_logps.view(
        shift_logits.size(0), shift_logits.size(1)
    )  # Reshape back to (batch_size, seq_len-1)

    # Create a mask for the labels that are not -100
    loss_mask = shift_labels != -100

    # Apply the mask to the per token log probabilities
    masked_logps = per_token_logps * loss_mask

    # Calculate the sum or average log probability per sequence
    sequence_logps = masked_logps.sum(dim=-1)

    if average_log_prob:
        # Avoid division by zero for sequences with no valid tokens
        num_valid_tokens = loss_mask.sum(dim=-1)
        return sequence_logps / torch.clamp(num_valid_tokens, min=1)
    else:
        return sequence_logps


def _dpo_loss(chosen_logps: torch.Tensor,
              rejected_logps: torch.Tensor,
              beta: float,
              loss_type: str = "sigmoid",
              label_smoothing: float = 0.0,
              simpo_gamma: float = 0.5
              ) -> torch.Tensor:
    """Standard DPO: mean over pairs."""

    logits = chosen_logps - rejected_logps
    if loss_type == "sigmoid":
        # label smoothing on pairwise preference
        loss = (-F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing).mean()
    elif loss_type == "ipo":
        loss =  ((logits - 1.0 / (2.0 * beta)) ** 2).mean()
    elif loss_type == "simpo":
        gamma_logratios = simpo_gamma / beta
        logits = logits - gamma_logratios
        # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
        loss = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        ).mean()
    elif loss_type == "hinge":
        loss =  torch.relu(1 - beta * logits).mean()
    else:
        raise ValueError(f"Unknown dpo loss_type: {loss_type}")
    chosen_rewards = beta * (chosen_logps).detach()
    rejected_rewards = beta * (rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards

class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()


        # --- config knobs for DPO ---
        self.use_dpo = self.config["use_dpo_loss"] 
        self.dpo_coef = self.config["dpo_coef"] 
        self.dpo_beta =  self.config["dpo_beta"] 
        self.dpo_loss_type =  self.config["dpo_loss_type"] 
        self.dpo_label_smoothing =  self.config["dpo_label_smoothing"] 
        self.simpo_gamma =  self.config["simpo_gamma"] 

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        # print("response_length", response_length, response_length)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                # print("multi_modal_inputs 1", multi_modal_inputs)
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                # print("multi_modal_inputs 1", multi_modal_inputs)

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):


        
        response_length = data.batch["responses"].size(-1)
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")



        # ---- Extra keys for DPO (we build chosen=GT and rejected=generated) ----
        if self.use_dpo:
            # either you provide explicit chosen_* / rejected_*,
            # or we derive them from full_input_ids/full_attention_mask vs input_ids/attention_mask
            explicit_keys = [
                "chosen_input_ids", "chosen_attention_mask", "chosen_position_ids",
                "chosen_labels", "rejected_input_ids", "rejected_attention_mask",
                "rejected_position_ids", "rejected_labels",
            ]
            select_keys.extend([k for k in explicit_keys if k in data.batch])





        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss
                    

                    if self.use_dpo:
                        # try explicit chosen/rejected; otherwise derive from full_* vs generated

                        ch_ids = model_inputs["chosen_input_ids"]
                        ch_mask = model_inputs["chosen_attention_mask"]
                        ch_pos  = model_inputs["chosen_position_ids"]
                        ch_lbls = model_inputs["chosen_labels"]
                        
                        rj_ids = model_inputs["rejected_input_ids"]
                        rj_mask = model_inputs["rejected_attention_mask"]
                        rj_pos  = model_inputs["rejected_position_ids"]
                        rj_lbls = model_inputs["rejected_labels"]
                        # print("ch_ids", ch_ids.shape, ch_ids)
                        # print("ch_mask", ch_mask.shape, ch_mask)
                        # print("ch_lbls", ch_lbls.shape, ch_lbls)
                        # print("rj_ids", rj_ids.shape, rj_ids)
                        # print("rj_mask", rj_mask.shape, rj_mask)
                        # print("rj_lbls", rj_lbls.shape, rj_lbls)
                        # Handle Qwen2VL-style MRoPE position_ids if present
                        if ch_pos is not None:
                            if ch_pos.dim() == 3:
                                # (bsz, 4, seqlen) -> (4, bsz, seqlen) AND make contiguous
                                ch_pos = ch_pos.transpose(0, 1).contiguous()
                            else:
                                ch_pos = ch_pos.contiguous()
                            # qwen models expect long
                            ch_pos = ch_pos.to(torch.long)

                        if rj_pos is not None:
                            if rj_pos.dim() == 3:
                                rj_pos = rj_pos.transpose(0, 1).contiguous()
                            else:
                                rj_pos = rj_pos.contiguous()
                            rj_pos = rj_pos.to(torch.long)

                        # multi-modal inputs (optional)
                        mm_inputs = {}
                        if "multi_modal_inputs" in model_inputs:
                            from verl.utils.model import extract_multi_modal_inputs
                            mm_inputs = extract_multi_modal_inputs(model_inputs["multi_modal_inputs"])
                        # print("mm_inputs", mm_inputs)
                        extra_args = {}
                        if self.use_fused_kernels:
                            extra_args["temperature"] = temperature
                            extra_args["return_dict"] = True
                        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                            ch_out = self.actor_module(
                                input_ids=ch_ids, attention_mask=ch_mask, position_ids=ch_pos,
                                **mm_inputs, use_cache=False, **extra_args, #return_dict=True,
                            )
                            rj_out = self.actor_module(
                                input_ids=rj_ids, attention_mask=rj_mask, position_ids=rj_pos,
                                **mm_inputs, use_cache=False, **extra_args, #return_dict=True,
                            )
                            # print("ch_out", ch_out)
                            # print("rj_out", rj_out)
                        # print("ch_out.log_probs", ch_out.log_probs)
                        # print("rj_out.log_probs", rj_out.log_probs)
                        # print("ch_out.logits", ch_out.logits)
                        # print("rj_out.logits", rj_out.logits)                        
                        # if self.use_fused_kernels:
                        #     ch_log_probs = ch_out.log_probs
                        #     rj_log_probs = rj_out.log_probs
                        #     # print("ch_log_probs", ch_log_probs.shape, ch_log_probs)
                        #     # print("rj_log_probs", rj_log_probs.shape, rj_log_probs)
                            
                        #     # ch_logits = ch_logits[:, -response_length - 1 : -1]
                        #     # rj_logits = rj_logits[:, -response_length - 1 : -1]

                        #     # print("ch_logits", ch_logits.shape)
                        #     # print("rj_logits", rj_logits.shape)
                        # else:
                        #     raise NotImplementedError

                        average_log_prob=self.dpo_loss_type in ["ipo", "simpo"]
                        if self.use_fused_kernels:
                            ch_log_probs = ch_out.log_probs
                            rj_log_probs = rj_out.log_probs
                            ch_log_probs = get_batch_logps(ch_log_probs, ch_lbls, average_log_prob=average_log_prob)
                            rj_log_probs = get_batch_logps(rj_log_probs, rj_lbls, average_log_prob=average_log_prob)
                            
                            # print("ch_logits", ch_logits.shape, ch_logits)
                            # print("rj_logits", rj_logits.shape, rj_logits)
                        else:
                            ch_logits = ch_out.logits
                            ch_logits.div_(temperature)
                            ch_log_probs = get_batch_logps_from_logits(ch_logits, ch_lbls, average_log_prob=average_log_prob)
                            # print("ch_logits", ch_logits.shape, ch_logits)
                            # print("ch_log_probs", ch_log_probs.shape, ch_log_probs)

                            rj_logits = rj_out.logits
                            rj_logits.div_(temperature)
                            rj_log_probs =  get_batch_logps_from_logits(rj_logits, rj_lbls, average_log_prob=average_log_prob)
                            # print("rj_logits", rj_logits.shape, rj_logits)
                            # print("rj_log_probs", rj_log_probs.shape, rj_log_probs)
                        dpo_loss, chosen_rewards, rejected_rewards = _dpo_loss(
                            chosen_logps=ch_log_probs,
                            rejected_logps=rj_log_probs,
                            beta=self.dpo_beta,
                            loss_type=self.dpo_loss_type,
                            label_smoothing=self.dpo_label_smoothing,
                            simpo_gamma=self.simpo_gamma
                        )
                        if torch.isfinite(dpo_loss).item():
                            loss = loss + self.dpo_coef * dpo_loss  
                            print("dpo_loss", dpo_loss )
                        else:                  
                            print("dpo_loss!!!", dpo_loss )
                    
                    
                    
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = loss * loss_scale_factor
                    else:
                        loss = loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    if self.use_dpo:
                        micro_batch_metrics["actor/dpo_loss"] = dpo_loss.detach().item()
                        micro_batch_metrics["actor/dpo_chosen_rewards"]  =  chosen_rewards.mean().item()
                        micro_batch_metrics["actor/dpo_rejected_rewards"]  =  rejected_rewards.mean().item()
                        micro_batch_metrics["actor/reward_accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
                        
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
