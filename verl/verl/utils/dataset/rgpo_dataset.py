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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RGPODataset(Dataset):
    """
    Load and preprocess RGPO data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        print("Processing RGPO data")
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        
        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.system_prompt = config.get("system_prompt", "You are a helpful assistant")
        try:
            from pathlib import Path
            self.system_prompt = Path(self.system_prompt.strip()).read_text(encoding="utf-8")
        except:
            pass
        print("system_prompt", {self.system_prompt})
        
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # ==== DPO-related (optional) ====
        self.response_key = config.get("response_key", "response")
        # cap response length in tokens (None = no cap)
        self.max_response_length = config.get("max_response_length", None)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts and (optionally) too long prompt+response
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        total_max = int(self.max_prompt_length) + int(self.max_response_length)

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

        def _lens(doc):
            """Return (prompt_len, full_len_or_None) in tokens. Uses a COPY to avoid mutating HF rows."""
            row = dict(doc)  # avoid side-effects from _build_messages pop()
            messages = self._build_messages(row)

            if processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                images = [process_image(img) for img in doc[image_key]] if (image_key in doc and doc[image_key]) else None
                videos = [process_video(v) for v in doc[video_key]] if (video_key in doc and doc[video_key]) else None

                prompt_len = len(self.processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

                resp_text = doc.get(self.response_key, None)
                if isinstance(resp_text, str) and len(resp_text) > 0:
                    full_messages = messages + [{"role": "assistant", "content": resp_text}]
                    full_text = self.processor.apply_chat_template(
                        full_messages, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    full_len = len(self.processor(text=[full_text], images=images, videos=videos)["input_ids"][0])
                else:
                    full_len = None

            else:
                raw_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                # IMPORTANT: count tokens, not characters
                prompt_len = len(tokenizer(raw_prompt, add_special_tokens=False)["input_ids"])

                resp_text = doc.get(self.response_key, None)
                if isinstance(resp_text, str) and len(resp_text) > 0:
                    full_messages = messages + [{"role": "assistant", "content": resp_text}]
                    full_text = tokenizer.apply_chat_template(
                        full_messages, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    full_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
                else:
                    full_len = None

            return prompt_len, full_len

        def _keep(doc):
            p_len, f_len = _lens(doc)
            if p_len > self.max_prompt_length:
                return False
            # If a response exists, also enforce total cap
            if f_len is not None and f_len > total_max:
                return False
            return True

        dataframe = dataframe.filter(
            _keep,
            num_proc=self.num_workers,
            desc=f"Filtering prompts>{self.max_prompt_length} and full>{total_max} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        prompt = example.pop(self.prompt_key)
        if isinstance(prompt, list):
            prompt = prompt[0]["content"]
            print("prompt", prompt)
        messages = [{"role": 'system', "content": self.system_prompt},{"role": "user", "content": prompt}]
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        resp_text = row_dict[self.response_key]
        messages = self._build_messages(row_dict)
        full_messages = messages + [{"role": "assistant", "content": resp_text}]
        model_inputs = {}
        images, videos = None, None
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

            full_text = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            full_model_inputs = self.processor(
                text=[full_text], images=images, videos=videos, return_tensors="pt"
            )
            full_input_ids = full_model_inputs.pop("input_ids")
            full_attention_mask = full_model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in full_model_inputs:
                full_model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["full_multi_modal_inputs"] = dict(full_model_inputs)

                # second_per_gri d_ts isn't used for training, just for mrope
                row_dict["full_multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            full_text = self.tokenizer.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            full_model_inputs = self.tokenizer(
                full_text, return_tensors="pt", add_special_tokens=False
            )
            full_input_ids = full_model_inputs.pop("input_ids")
            full_attention_mask = full_model_inputs.pop("attention_mask")
            


        row_dict["full_messages"] = full_messages

        
        full_input_ids, full_attention_mask = verl_F.postprocess_data(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            max_length=self.max_prompt_length+self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
                
            
            # # # # # # # # # #
            full_vision_position_ids = get_rope_index(
                self.processor,
                input_ids=full_input_ids[0],
                image_grid_thw=full_model_inputs.get("image_grid_thw"),
                video_grid_thw=full_model_inputs.get("video_grid_thw"),
                second_per_grid_ts=full_model_inputs.get("second_per_grid_ts"),
                attention_mask=full_attention_mask[0],
            )  # (3, seq_length)
            full_valid_mask = full_attention_mask[0].bool()
            full_text_position_ids = torch.ones((1, len(full_input_ids[0])), dtype=torch.long)
            full_text_position_ids[0, full_valid_mask] = torch.arange(full_valid_mask.sum().item())
            full_position_ids = [torch.cat((full_text_position_ids, full_vision_position_ids), dim=0)]  # (1, 4, seq_length)
                         
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            raise NotImplementedError
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
            full_position_ids = compute_position_id_with_mask(full_attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        row_dict["full_input_ids"] = full_input_ids[0]
        row_dict["full_attention_mask"] = full_attention_mask[0]
        row_dict["full_position_ids"] = full_position_ids[0]
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
         
        
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
            row_dict["full_raw_prompt"] = full_messages
            

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
