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
from mathruler.grader import grade_answer



def compute_score(solution_str: str, ground_truth: str, timeout_score: float = 0, prompt_str=None, format_weight = 0.1, **kwargs) -> bool:

    solution_str = solution_str.replace("\x0c", "\\f").strip()   
    sol_match = re.search(r'<answer>(.*?)</answer>', solution_str)
    format_score = 0.
    solution_extracted = solution_str.strip()
    if sol_match:
        solution_extracted = sol_match.group(1).strip()  
        format_score = 1.
    ground_truth = ground_truth.replace("\x0c", "\\f").strip()  
    
    accuracy_score = 0
    if grade_answer(ground_truth, solution_extracted):
        accuracy_score = 1.

    score = (1 - format_weight) * accuracy_score + format_weight * format_score

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} -------------\n")
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
    print(compute_score("\\boxed{4\pi}", "$4 \pi$"))
    print(compute_score("80%", "\frac{4}{5}"))
    print(compute_score("8/10", "\frac{4}{5}"))
    print(compute_score("0.8", "\frac{4}{5}"))
    print(compute_score(".8", "$\frac{4}{5}$"))
    