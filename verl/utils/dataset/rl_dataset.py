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
import json
import logging
import os
import re
from collections import defaultdict
from typing import Optional, Union

import datasets
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.format_adapter import FormatAdapter

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
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Universal dataset that supports multiple data sources and formats.

    - Supports local files (parquet, json, jsonl, csv) and HuggingFace datasets
    - Automatically adapts various formats to unified input/label/meta structure
    - Caches files locally
    - Filters prompts over a max length
    - Supports resuming from checkpoints

    Args:
        data_files (str or list): Path(s) to data files or HuggingFace dataset names.
            For backward compatibility, this parameter is kept as 'data_files'.
            Examples:
                - "train.parquet" (local file)
                - "openai/gsm8k" (HuggingFace dataset)
                - ["file1.json", "file2.jsonl"] (multiple files)
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, field_mapping, max_prompt_length, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, list[str]],  # Keep parameter name for backward compatibility
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # Support both single and multiple data sources
        if not isinstance(data_files, list | ListConfig):
            data_sources = [data_files]
        else:
            data_sources = data_files
            
        self.data_sources = copy.deepcopy(data_sources)
        self.original_data_sources = copy.deepcopy(data_sources)  # for resume
        
        # For backward compatibility
        self.data_files = self.data_sources
        self.original_data_files = self.original_data_sources
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # Initialize format adapter
        self.format_adapter = FormatAdapter(config)

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        # Load data based on source type
        self._load_data()
        
        # Apply format adaptation
        self._adapt_formats()
        
        # Filter overlong prompts if needed
        if self.filter_overlong_prompts:
            self._filter_by_length()

    def _load_data(self):
        """Load data from various sources"""
        all_data = []
        
        for source in self.data_sources:
            if isinstance(source, str):
                # Determine source type
                if self._is_huggingface_dataset(source):
                    data = self._load_huggingface_dataset(source)
                else:
                    # Local file
                    data = self._load_local_file(source)
            else:
                raise ValueError(f"Unsupported data source type: {type(source)}")
            
            all_data.append(data)
        
        # Concatenate all datasets
        if len(all_data) == 1:
            self.dataframe = all_data[0]
        else:
            self.dataframe = datasets.concatenate_datasets(all_data)
        
        logger.info(f"Loaded dataset with {len(self.dataframe)} examples")
    
    def _is_huggingface_dataset(self, source: str) -> bool:
        """Check if source is a HuggingFace dataset name"""
        # Simple heuristic: HF datasets usually contain '/' and don't exist as local files
        return '/' in source and not os.path.exists(source)
    
    def _load_huggingface_dataset(self, dataset_name: str) -> datasets.Dataset:
        """Load dataset from HuggingFace"""
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        
        # Parse dataset name and optional config/split
        parts = dataset_name.split(':')
        name = parts[0]
        
        # Default config and split
        config_name = None
        split = self.config.get("split", "train")
        
        # Parse config:split if provided
        if len(parts) > 1:
            if '/' in parts[1]:
                config_name, split = parts[1].split('/', 1)
            else:
                split = parts[1]
        
        # Load dataset
        dataset = datasets.load_dataset(name, config_name, split=split)
        
        # If it's a DatasetDict, get the specified split
        if isinstance(dataset, datasets.DatasetDict):
            dataset = dataset[split]
            
        return dataset
    
    def _load_local_file(self, file_path: str) -> datasets.Dataset:
        """Load dataset from local file"""
        from verl.utils.fs import copy_to_local
        
        # Expand user path
        file_path = os.path.expanduser(file_path)
        
        # Copy to cache if needed (for HDFS etc.)
        if not os.path.exists(file_path):
            file_path = copy_to_local(src=file_path, cache_dir=self.cache_dir, use_shm=self.use_shm)
        
        logger.info(f"Loading local file: {file_path}")
        
        # Determine file format
        if file_path.endswith('.parquet'):
            dataset = datasets.load_dataset('parquet', data_files=file_path)['train']
        elif file_path.endswith('.json'):
            dataset = datasets.load_dataset('json', data_files=file_path)['train']
        elif file_path.endswith('.jsonl'):
            dataset = datasets.load_dataset('json', data_files=file_path)['train']
        elif file_path.endswith('.csv'):
            dataset = datasets.load_dataset('csv', data_files=file_path)['train']
        else:
            # Try to auto-detect format
            dataset = datasets.load_dataset('text', data_files=file_path)['train']
            
        return dataset
    
    def _adapt_formats(self):
        """Apply format adaptation to convert data to unified structure"""
        adapted_data = []
        
        for idx in range(len(self.dataframe)):
            item = self.dataframe[idx]
            
            # Apply format adaptation
            adapted_item = self.format_adapter.adapt(item)
            
            # For backward compatibility, keep original fields in the adapted data
            # The original 'prompt' field will be in meta if it wasn't used as input
            if self.prompt_key in item and "prompt" not in adapted_item:
                adapted_item["prompt"] = adapted_item["input"]
                
            adapted_data.append(adapted_item)
        
        # Convert back to HuggingFace Dataset
        self.dataframe = datasets.Dataset.from_list(adapted_data)
        
    def _filter_by_length(self):
        """Filter out examples that are too long"""
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_sources")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            # Reload from original sources
            self.data_sources = copy.deepcopy(self.original_data_sources)
            self._load_data()
            self._adapt_formats()
            if self.filter_overlong_prompts:
                self._filter_by_length()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        """Build messages from the unified format"""
        # Handle new unified format
        if "input" in example:
            input_data = example["input"]
            
            # If input is already in message format
            if isinstance(input_data, list) and all(isinstance(m, dict) and "role" in m for m in input_data):
                messages = input_data
            # If input is a string, convert to message format
            elif isinstance(input_data, str):
                messages = [{"role": "user", "content": input_data}]
            else:
                # Try to get from the old prompt_key for backward compatibility
                messages = example.get(self.prompt_key, [])
                if not isinstance(messages, list):
                    messages = [{"role": "user", "content": str(messages)}]
        else:
            # Fallback to old behavior
            messages = example.get(self.prompt_key, [])
            
        # Handle multimodal content
        meta = example.get("meta", {})
        if self.image_key in meta or self.video_key in meta:
            for message in messages:
                if isinstance(message.get("content"), str):
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
        row_dict: dict = copy.deepcopy(dict(self.dataframe[item]))
        
        # Extract meta information
        meta = row_dict.get("meta", {})
        
        # Build messages from the unified format
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            # Check both in row_dict and meta for backward compatibility
            if self.image_key in meta and meta.get(self.image_key, None) is not None:
                images = [process_image(image) for image in meta.pop(self.image_key)]
            elif self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            # Check both in row_dict and meta for backward compatibility
            if self.video_key in meta and meta.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in meta.pop(self.video_key)]
            elif self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

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

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

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

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # Extract information from meta or extra_info for backward compatibility
        extra_info = meta.get("extra_info", row_dict.get("extra_info", {}))
        
        # add index for each prompt
        index = extra_info.get("index", meta.get("index", 0))
        tools_kwargs = extra_info.get("tools_kwargs", meta.get("tools_kwargs", {}))
        interaction_kwargs = extra_info.get("interaction_kwargs", meta.get("interaction_kwargs", {}))
        need_tools_kwargs = extra_info.get("need_tools_kwargs", meta.get("need_tools_kwargs", self.need_tools_kwargs))
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        
        # Add all meta fields to the result (except those already processed)
        for key, value in meta.items():
            if key not in row_dict and key not in ["extra_info", self.image_key, self.video_key]:
                row_dict[key] = value
                
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
