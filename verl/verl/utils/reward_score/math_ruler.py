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

import os, re
from datetime import datetime
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(solution_str: str, ground_truth: str, timeout_score: float = 0, prompt_str=None, data_source="unknown", format_weight: float = 0.1, **kwargs) -> bool:

    response = re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)  # handle qwen2.5vl-32b format
    format_score = format_reward(response)
    accuracy_score = accuracy_reward(response, ground_truth)
    score = (1 - format_weight) * accuracy_score + format_weight * format_score
    solution_extracted = extract_boxed_content(response)
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Accuracy reward: {score} -------------\n")
            f.write(f"Prompt: {prompt_str}\n")
            f.write(f"Response: {solution_str}\n")
            f.write(f"Solution: {solution_extracted}\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Format score: {format_score}\n")
            f.write(f"Accuracy score: {accuracy_score}\n")
            f.write(f"Overall score: {score}\n")
            
    return score

if __name__ == "__main__":
    print(compute_score("D", "D"))
    
    print(compute_score("4 \pi", "$4 \pi$"))
    print(compute_score("4*\pi", "$4 \pi$"))
    print(compute_score("4\pi", "$4 \pi$"))