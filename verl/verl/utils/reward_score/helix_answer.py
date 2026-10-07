from datetime import datetime
import os, re, regex  # Import regex module to support recursive matching of balanced brackets

from verl.utils.reward_score.mmhelix.metrics import metrics
# from verl.utils.reward_score.mmhelix.parser import parser

def str_to_dict(input_str):
    if isinstance(input_str, str):
        try:
            import json
            input_str = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            try:
                import ast
                input_str = ast.literal_eval(input_str)
            except (ValueError, SyntaxError):
                pass
    return input_str

def parse(response):
    if not response:
        return ""

    # Try to parse <|begin_of_box|>...</|end_of_box|> format - match from back to front
    box_pattern = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    matches = re.findall(box_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Take the last match

    # Then try to parse <answer></answer> format - match from back to front
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Take the last match

    # First try to parse \boxed{} format - match from back to front, use regex to support nested brackets
    boxed_pattern = r'\\boxed\{((?:[^{}]|\{[^}]*\})*)\}'
    matches = regex.findall(boxed_pattern, response)
    if matches:
        last_match = matches[-1]  # Take the last match
        # Find and remove all \text{} format in the last match
        text_pattern = r'\\text\{([^}]*)\}'
        text_matches = regex.findall(text_pattern, last_match)
        if text_matches:
            return text_matches[-1].strip()

        # 2. Handle truncated cases: ext{content} (missing \t)
        elif regex.search(r'ext\{[^}]*\}', last_match):
            ext_pattern = r'ext\{([^}]*)\}'
            ext_matches = regex.findall(ext_pattern, last_match)
            if ext_matches:
                return ext_matches[-1].strip()

        # 3. Handle other possible text variants
        elif 'text{' in last_match:
            # Remove any form of text{...}
            cleaned = regex.sub(r'[\\]*text\{([^}]*)\}', r'\1', last_match)
            if cleaned.strip() != last_match.strip():
                return cleaned.strip()

        # Try to parse \begin{array}...\end{array} format - use regex matching
        array_pattern = r'\\begin\{array\}((?:.|\n)*?)\\end\{array\}'
        array_matches = regex.findall(array_pattern, last_match)
        if array_matches:
            return array_matches[-1].strip()  # Take the last match

        # Try to parse \begin{bmatrix}...\end{bmatrix} format - use regex matching
        bmatrix_pattern = r'\\begin\{bmatrix\}((?:.|\n)*?)\\end\{bmatrix\}'
        bmatrix_matches = regex.findall(bmatrix_pattern, last_match)
        if bmatrix_matches:
            return bmatrix_matches[-1].strip()  # Take the last match
        return last_match.strip()

    # Finally try to parse Answer: format - match from back to front
    answer_matches = re.findall(r'Answer[:：]\s*(.*)', response, re.IGNORECASE | re.DOTALL)
    if answer_matches:
        return answer_matches[-1].strip()  # Take the last match

    return response  # return the original response

def compute_score(predicted_answer, ground_truth, initial_state, score_function, params=None):
    if score_function in metrics:
        evaluator = metrics[score_function]
        print(f"Evaluating with {score_function} metric...")
        return evaluator.evaluate(predicted_answer, ground_truth, initial_state, params)
    else:
        raise ValueError(f"Score function '{score_function}' not found in metrics.")


def format_reward(response: str) -> float:
    pattern = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)
    # strip() so a leading/trailing newline or space (common under sampling) doesn't
    # zero the format bonus for otherwise well-formatted responses.
    format_match = re.fullmatch(pattern, response.strip())
    return 1.0 if format_match else 0.0

def compute_score(solution_str: str, ground_truth: str,  data_source="unknown", format_weight: float = 0.1, prompt_str=None, extra_info=None, **kwargs) -> bool:
    # print(f"Extra info for scoring: {extra_info}")
    score_function = extra_info["score_function"]
    initial_state = str_to_dict(extra_info["initial_state"])
    if not isinstance(initial_state, dict):
        initial_state = None
    format_score = format_reward(solution_str)
    solution_extracted = parse(solution_str)
    if score_function in metrics:
        evaluator = metrics[score_function]
        # TODO: feedback for all metrics
        accuracy_score, feedback = evaluator.evaluate(solution_extracted, ground_truth, initial_state)
        accuracy_score = float(accuracy_score)
    else:
        raise ValueError(f"Score function '{score_function}' not found in metrics.")
    score = (1 - format_weight) * accuracy_score + format_weight * format_score
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} -------------\n")
            f.write(f"Data source: {data_source}\n")
            f.write(f"Prompt: {prompt_str}\n")
            f.write(f"Initial state: {initial_state}\n")
            f.write(f"Response: {solution_str}\n")
            f.write(f"Solution: {solution_extracted}\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Format score: {format_score}\n")
            f.write(f"Accuracy score: {accuracy_score}\n")
            f.write(f"Overall score: {score}\n")
            f.write(f"Score function: {score_function}\n")
            f.write(f"Feedback: {feedback}\n")
    # FIXME: debug
    # feedback = ""
    return {
        "score": score,
        "format_score": format_score,
        "accuracy_score": accuracy_score,
        "feedback": f"{feedback}\nFormat score: {format_score}/1.0\nAccuracy score: {accuracy_score}/1.0\nOverall score: {score}/1.0\nYou should maximize the overall score by improving both the format and accuracy of your answer.",
    }
if __name__ == "__main__":
    # Smoke tests for format_reward: returns 1.0 iff the (stripped) response
    # contains a literal \boxed{...}. re.DOTALL lets .* span newlines, and the
    # .strip() keeps leading/trailing whitespace from zeroing a valid response.
    cases = [
        (r"\boxed{42}", 1.0),
        (r"The answer is \boxed{42}.", 1.0),
        ("  \\boxed{42}  ", 1.0),                 # surrounding whitespace
        ("\n\\boxed{5}\n", 1.0),                  # surrounding newlines
        ("Step 1.\nStep 2.\n\\boxed{7}", 1.0),    # multiline reasoning
        (r"\boxed{}", 1.0),                        # empty box still matches
        (r"a \boxed{x} b \boxed{y} c", 1.0),      # multiple boxes
        ("no box here", 0.0),
        ("", 0.0),
        (r"\boxed{42", 0.0),                       # missing closing brace
        ("boxed{42}", 0.0),                        # missing leading backslash
    ]
    passed = 0
    for i, (resp, expected) in enumerate(cases):
        got = format_reward(resp)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] case {i}: expected {expected}, got {got}  |  {resp!r}")
    print(f"\n{passed}/{len(cases)} passed")
    assert passed == len(cases), "format_reward tests failed"

    # Smoke tests for parse(): extraction precedence is
    # <|begin_of_box|> > <answer> > \boxed{} (with \text/ext/array/bmatrix
    # sub-handling) > "Answer:" > original response; last match wins per tier.
    parse_cases = [
        ("x <|begin_of_box|>42<|end_of_box|> y", "42", "begin_of_box basic"),
        ("<|begin_of_box|>A<|end_of_box|> <answer>B</answer> \\boxed{C}", "A", "box beats answer+boxed"),
        ("<|begin_of_box|>first<|end_of_box|> <|begin_of_box|>second<|end_of_box|>", "second", "box last-match"),
        ("reasoning <answer>the answer</answer>", "the answer", "answer basic"),
        ("<answer>A</answer> \\boxed{B}", "A", "answer beats boxed"),
        ("<answer>one</answer><answer>two</answer>", "two", "answer last-match"),
        (r"The answer is \boxed{42}", "42", "boxed basic"),
        (r"\boxed{f(x) = {a}}", "f(x) = {a}", "boxed single-level nesting"),
        (r"\boxed{\text{hello}}", "hello", "boxed \\text{}"),
        (r"\boxed{ext{world}}", "world", "boxed truncated ext{}"),
        (r"\boxed{\begin{array}1 & 2\end{array}}", "1 & 2", "boxed array"),
        (r"\boxed{\begin{bmatrix}1 & 2\end{bmatrix}}", "1 & 2", "boxed bmatrix"),
        ("The answer. Answer: 42", "42", "Answer: basic"),
        ("Answer：99", "99", "Answer fullwidth colon"),
        (r"Answer: 10 and \boxed{20}", "20", "boxed beats Answer:"),
        ("just some text with no answer marker", "just some text with no answer marker", "no marker -> original"),
        ("", "", "empty -> ''"),
        (None, "", "None -> ''"),
    ]
    p_passed = 0
    for i, (resp, expected, label) in enumerate(parse_cases):
        got = parse(resp)
        ok = got == expected
        p_passed += ok
        print(resp)
        print(f"[{'PASS' if ok else 'FAIL'}] parse {i:2d} ({label}): expected {expected!r}, got {got!r}")
    print(f"\n{p_passed}/{len(parse_cases)} parse cases passed")
    assert p_passed == len(parse_cases), "parse tests failed"
