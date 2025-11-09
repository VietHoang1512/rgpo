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
import os 
from datetime import datetime
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def compute_score(solution_str: str, ground_truth: str, timeout_score: float = 0, prompt_str=None, **kwargs) -> bool:


    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    try:
        ret_score, _ = verify_func([ground_truth], [solution_str])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} -------------\n")
            f.write(f"Prompt: {prompt_str}\n")
            f.write(f"Response: {solution_str}\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Overall score: {ret_score}\n")
    return ret_score

if __name__ == "main":
    print(compute_score(input(), "\\boxed{\frac{3\sqrt{3}}{2}}"))