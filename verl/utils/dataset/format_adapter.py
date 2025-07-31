# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Format adapter for converting various data formats to unified structure
"""

from typing import Any, Dict, List, Union, Optional


class FormatAdapter:
    """Convert various formats to unified input/label/meta structure"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Default field mappings for common formats
        self.input_keys = config.get("input_keys", [
            "prompt", "question", "instruction", "messages", "query", "input", "text"
        ])
        self.label_keys = config.get("label_keys", [
            "response", "answer", "output", "completion", "label", "target"
        ])
        
        # Custom field mapping from config
        self.custom_mapping = config.get("field_mapping", {})
        
        # Whether to auto-detect format
        self.auto_detect = config.get("auto_detect", True)
    
    def adapt(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a data item to unified format with input/label/meta fields
        
        Args:
            item: Original data item
            
        Returns:
            Dict with 'input', 'label' (optional), and 'meta' fields
        """
        result = {"meta": {}}
        
        # 1. Handle input field
        input_value = self._extract_input(item)
        if input_value is not None:
            result["input"] = input_value
        
        # 2. Handle label field (optional)
        label_value = self._extract_label(item)
        if label_value is not None:
            result["label"] = label_value
        
        # 3. Put all other fields into meta
        used_keys = set()
        if "input" in result:
            # Find which key was used for input
            for key in self.input_keys + list(self.custom_mapping.values()):
                if key in item and item[key] == result["input"]:
                    used_keys.add(key)
                    break
        
        if "label" in result:
            # Find which key was used for label
            for key in self.label_keys + list(self.custom_mapping.values()):
                if key in item and item[key] == result["label"]:
                    used_keys.add(key)
                    break
        
        # Add all unused fields to meta
        for key, value in item.items():
            if key not in used_keys:
                result["meta"][key] = value
        
        # Ensure we have at least an input field
        if "input" not in result:
            raise ValueError(f"Could not find input field in item. Available keys: {list(item.keys())}")
        
        return result
    
    def _extract_input(self, item: Dict[str, Any]) -> Optional[Union[str, List[Dict[str, str]]]]:
        """Extract input field from item"""
        # First check custom mapping
        if "input" in self.custom_mapping:
            custom_key = self.custom_mapping["input"]
            if custom_key in item:
                return item[custom_key]
        
        # Then check default keys if auto_detect is enabled
        if self.auto_detect:
            for key in self.input_keys:
                if key in item:
                    return item[key]
        
        return None
    
    def _extract_label(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract label field from item"""
        # First check custom mapping
        if "label" in self.custom_mapping:
            custom_key = self.custom_mapping["label"]
            if custom_key in item:
                return item[custom_key]
        
        # Then check default keys if auto_detect is enabled
        if self.auto_detect:
            for key in self.label_keys:
                if key in item:
                    return item[key]
        
        return None
    
    @staticmethod
    def detect_format(item: Dict[str, Any]) -> str:
        """
        Detect the format of a data item
        
        Returns:
            Format type: 'openai', 'alpaca', 'sharegpt', 'unified', 'unknown'
        """
        if "messages" in item and isinstance(item["messages"], list):
            return "openai"
        elif "instruction" in item and "output" in item:
            return "alpaca"
        elif "conversations" in item and isinstance(item["conversations"], list):
            return "sharegpt"
        elif "input" in item and ("label" in item or "meta" in item):
            return "unified"
        else:
            return "unknown"