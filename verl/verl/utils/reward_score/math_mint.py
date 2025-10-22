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
from typing import Tuple, List

STRUCTURE_RE = re.compile(
    r"""
    ^\s*\#\#\#\s*Image\ Description:\s*\r?\n      # 1) Image Description header
    .*?                                          #    any content (non-greedy)
    ^\#\#\#\s*Rationales:\s*\r?\n                # 2) Rationales header
    .*?                                          #    any content
    ^\#\#\#\s*Let's\ think\ step\ by\ step\.\s*\r?\n  # 3) "Let's think..." header (with period)
    (?:^\#\#\#\s*Step\s*\d+:\s*\r?\n.*?)+        # 4) One or more Step N: sections
    ^\#\#\#\s*The\ final\ answer\ is:\s*.*$        # 5) Final answer header
    """,
    re.DOTALL | re.MULTILINE | re.VERBOSE
)

def check_instruction_structure(text: str):
    return 1. if STRUCTURE_RE.search(text) is not None else 0.

def compute_score(solution_str: str, ground_truth: str, timeout_score: float = 0, prompt_str=None, format_weight = 0.1, **kwargs) -> bool:

    solution_str = solution_str.replace("\x0c", "\\f").strip()  
    sol_match = re.search(r'### The final answer is:(.+)', solution_str)
    format_score = 0.
    solution_extracted = solution_str.strip()  
    if sol_match:
        solution_extracted = sol_match.group(1).strip()  
    if check_instruction_structure(solution_str):
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
    import pandas as pd
    print(compute_score("4 \pi", "$4 \pi$"))
    print(compute_score("4*\pi", "$4 \pi$"))
    print(compute_score("4\pi", "$4 \pi$"))
    print(compute_score("\\boxed{4\pi}", "$4 \pi$"))
    print(compute_score("80%", "\frac{4}{5}"))
    print(compute_score("8/10", "\frac{4}{5}"))
    print(compute_score("0.8", "\frac{4}{5}"))
    print(compute_score(".8", "$\frac{4}{5}$"))
    minimal_ok = """### Image Description:\nsome description\n### Rationales:\nsome rationale\n### Let's think step by step.\nwhy the approach works\n### Step 1:\ndo this\n### Step 2:\ndo that\n### The final answer is:\nThe final answer is: 42."""
    # print(check_instruction_structure(minimal_ok))

    # ---------- Build test cases ----------
    valid_single_step = """### Image Description:
    An image of a cat.

    ### Rationales:
    We analyze.

    ### Let's think step by step.
    ### Step 1:
    First step details

    ### The final answer is:
    """

    valid_multi_step = """### Image Description:
    Details here

    ### Rationales:
    Reasoning content

    ### Let's think step by step.
    ### Step 1:
    Do this
    ### Step 2:
    Do that
    ### Step 10:
    Wrap up

    ### The final answer is:
    """

    missing_period_in_lets_think = """### Image Description:
    Something

    ### Rationales:
    Something

    ### Let's think step by step
    ### Step 1:
    Content

    ### The final answer is:
    """

    no_steps = """### Image Description:
    Content

    ### Rationales:
    More content

    ### Let's think step by step.
    (no steps here)

    ### The final answer is:
    """

    wrong_order = """### Rationales:
    Oops came first

    ### Image Description:
    Came second

    ### Let's think step by step.
    ### Step 1:
    ok

    ### The final answer is:
    """

    extra_text_after_final = """### Image Description:
    x

    ### Rationales:
    y

    ### Let's think step by step.
    ### Step 1:
    z

    ### The final answer is:
    After-final stray content
    """

    win_line_endings = "### Image Description:\r\nA\r\n\r\n### Rationales:\r\nB\r\n\r\n### Let's think step by step.\r\n### Step 1:\r\nC\r\n\r\n### The final answer is:\r\n"

    cases = [
        ("valid_single_step", valid_single_step),
        ("valid_multi_step", valid_multi_step),
        ("missing_period_in_lets_think", missing_period_in_lets_think),
        ("no_steps", no_steps),
        ("wrong_order", wrong_order),
        ("extra_text_after_final", extra_text_after_final),
        ("win_line_endings", win_line_endings),
    ]

    results = []
    for name, text in cases:
        results.append({
            "case": name,
            "follows_structure": check_instruction_structure(text)
        })

    df = pd.DataFrame(results)
    print("Regex structure check results", df)

    # Also print them plainly for quick glance
    for row in results:
        print(f"{row['case']}: {row['follows_structure']}")


    