import re
from collections import Counter
from typing import Dict, Any, Union, List, Tuple


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class TwentyFourPointsEvaluator(BaseEvaluator):
    """
    评估24点游戏解答的评估器
    验证模型输出的表达式是否：
    1. 正确使用了所有给定的数字（每个数字恰好使用一次）
    2. 计算结果是否等于24
    """

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """准备发送给模型的提示词"""
        prompt = (
            "Use these numbers exactly once, and combine them with +, -, ×, ÷, and parentheses to make 24.\n"
            "Please provide your answer as an expression that includes only numbers, operators, and parentheses.\n"
            "Example answer format: (9 - 3) × 8 ÷ 2."
        )
        return prompt

    def extract_answer(self, model_output: str) -> str:
        """从模型输出中提取表达式答案，优先提取最终答案"""
        if isinstance(model_output, dict) and "text" in model_output:
            model_output = model_output["text"]

        # 先预处理，将可能的LaTeX符号表示修复
        # 处理 \times 被解释为制表符的情况
        processed_output = model_output.replace('\times', r'\times')
        processed_output = processed_output.replace('\\div', r'\div')
        processed_output = processed_output.replace('\\cdot', r'\cdot')

        # 查找包含LaTeX符号的完整表达式
        # 匹配包含LaTeX符号的表达式，包括完整的符号和残余符号
        # 使用更宽松的匹配来获取完整表达式
        latex_patterns = [
            # 匹配包含完整\times或残余imes的表达式
            r'[\(\)\d\s\+\-×÷\*/\\timesa-z]*(?:\\times|imes)[\(\)\d\s\+\-×÷\*/\\timesa-z]*',
            # 匹配包含完整\div或单独div的表达式
            r'[\(\)\d\s\+\-×÷\*/\\diva-z]*(?:\\div|div)[\(\)\d\s\+\-×÷\*/\\diva-z]*',
            # 匹配包含完整\cdot或残余cdot的表达式
            r'[\(\)\d\s\+\-×÷\*/\\cdota-z]*(?:\\cdot|cdot)[\(\)\d\s\+\-×÷\*/\\cdota-z]*'
        ]

        for pattern in latex_patterns:
            matches = re.findall(pattern, model_output)
            if matches:
                # 找到最长的匹配项
                longest_match = max(matches, key=len)
                # 检查是否包含足够的数字
                numbers_in_match = re.findall(r'\d+', longest_match)
                if len(numbers_in_match) >= 3:  # 至少3个数字才可能是完整的24点表达式
                    return longest_match.strip()

        # 如果上面没有找到，尝试更通用的方法：查找整个输入中的完整表达式
        # 如果输入本身看起来就是一个表达式，直接使用
        if (re.search(r'\d+', model_output)
                and any(keyword in model_output for keyword in [
                    'imes', 'div', 'cdot', '+', '-', '*', '/', '×', '÷', '\\times', '\\div', '\\cdot'])):
            return model_output.strip()

        # 定义可接受的字符模式（不包含量词）
        # 包括：数字、括号、空格、各种运算符（标准符号、Unicode符号、反斜杠、字母）
        expression_chars = r'[\(\)\d\s\+\-×÷\*/\\a-zA-Z]'

        # 1. 查找显式标记的最终答案
        final_answer_patterns = [
            rf'(?:final answer|answer is|the answer is)[^\n]*?[=:]?\s*({expression_chars}+)',
            rf'(?:so|thus|hence)[^\n]*?[=:]?\s*({expression_chars}+)\s*=\s*24',
            rf'That\'s it!\s*(?:The)?\s*(?:answer|expression)\s*(?:is)?\s*:?\s*({expression_chars}+)'
        ]

        for pattern in final_answer_patterns:
            matches = re.findall(pattern, processed_output, re.IGNORECASE)
            if matches:
                return matches[-1].strip()  # 返回最后一个匹配（通常是最终答案）

        # 2. 查找等于24的表达式（优先选择文本最后出现的）
        expressions_with_24 = re.findall(rf'({expression_chars}+)\s*=\s*24', processed_output)
        if expressions_with_24:
            return expressions_with_24[-1].strip()

        # 3. 提取模型提供的最后一个完整表达式
        # 先按行分割文本
        lines = processed_output.split('\n')
        for line in reversed(lines):  # 从后往前检查
            # 查找包含数字和运算符的表达式（包括LaTeX格式和文字运算符）
            # 长度至少7个字符的表达式
            expr_matches = re.findall(rf'({expression_chars}{{7,}})', line)
            if expr_matches:
                # 过滤出格式有效的表达式
                valid_expressions = [expr for expr in expr_matches
                                     if self._is_valid_expression_format(expr)]
                if valid_expressions:
                    return valid_expressions[-1].strip()

        # 4. 如果上述方法都失败，提取整个文本中最可能的表达式
        all_expressions = re.findall(rf'({expression_chars}{{7,}})', processed_output)
        valid_expressions = [expr for expr in all_expressions
                             if self._is_valid_expression_format(expr)]

        if valid_expressions:
            # 按照启发式规则排序：优先选择包含括号、长度适中的表达式
            sorted_expressions = sorted(
                valid_expressions,
                key=lambda x: (
                    '(' in x and ')' in x,  # 优先有括号的
                    len(re.findall(r'\d+', x)),  # 优先包含更多数字的
                    -abs(len(x) - 15)  # 优先长度接近15个字符的（启发式值）
                ),
                reverse=True
            )
            return sorted_expressions[0].strip()

        # 5. 最后手段：如果输入本身就是一个表达式，直接返回
        if self._is_valid_expression_format(processed_output):
            return processed_output.strip()

        # 6. 返回最后一行非空文本
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return processed_output.strip()

    def _is_valid_expression_format(self, expr: str) -> bool:
        """检查表达式格式是否有效（包含数字和运算符）"""
        # 确保表达式包含至少一个数字和一个运算符
        has_number = bool(re.search(r'\d', expr))

        # 检查是否包含运算符（包括LaTeX符号和文字运算符）
        operator_patterns = [
            r'[\+\-\×\÷\*/]',  # 标准符号
            r'\\times|\\div|\\cdot',  # 完整的LaTeX符号
            r'\bimes\b|\bdiv\b|\bcdot\b',  # LaTeX符号的残余部分
            r'\bmul\b|\btimes\b|\bplus\b|\bminus\b'  # 文字运算符
        ]
        has_operator = any(re.search(pattern, expr, re.IGNORECASE) for pattern in operator_patterns)

        # 还要检查括号是否匹配
        open_brackets = expr.count('(')
        close_brackets = expr.count(')')
        brackets_match = open_brackets == close_brackets

        return has_number and has_operator and brackets_match

    def evaluate(self, output: str, ground_truth: Any, params: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Evaluate a 24-point answer and return ``(is_correct, feedback)``.

        The feedback is written to be *instructive* so it can be shown back to the
        model as a learning signal: it diagnoses exactly what went wrong — which
        numbers were mis-used, what the expression evaluated to, and in which
        direction it missed 24 — so the model can self-correct on the next
        attempt. It never reveals the reference solution.
        """
        input_numbers = self._get_input_numbers(ground_truth, params)
        nums_str = self._fmt_list(input_numbers) if input_numbers else "the given numbers"
        goal_hint = (
            f"Combine {nums_str}, using each number exactly once, with +, -, ×, ÷ and "
            f"parentheses to make 24, and give the final expression in \\boxed{{}}."
        )

        predicted_answer = self.extract_answer(output)
        expression = self._normalize_expression(predicted_answer) if predicted_answer else ""

        # No usable arithmetic expression found: guide the model to the format.
        if not expression or not re.search(r'[+\-*/]', expression):
            return False, f"No arithmetic expression was found in your answer. {goal_hint}"

        # If we somehow lack the puzzle's numbers, fall back to a plain value check.
        if not input_numbers:
            try:
                value = self._evaluate_expression(expression)
            except Exception:
                return False, f"Your expression could not be evaluated. {goal_hint}"
            if abs(value - 24) < 1e-6:
                return True, "Correct — your expression equals 24."
            return False, f"Your expression equals {self._fmt(value)}, not 24. {goal_hint}"

        try:
            # 1) Each given number must be used exactly once (multiset comparison).
            used_numbers = self._extract_numbers(expression)
            if Counter(used_numbers) != Counter(input_numbers):
                usage = self._describe_number_usage(used_numbers, input_numbers)
                return False, (
                    f"Number-usage error: {usage} You must use each of {nums_str} "
                    f"exactly once and use no other numbers."
                )

            # 2) The value must equal 24; otherwise report the gap and direction.
            value = self._evaluate_expression(expression)
            if abs(value - 24) < 1e-6:
                return True, (
                    f"Correct — your expression equals 24 using each of {nums_str} exactly once."
                )

            expr_echo = self._pretty_expression(expression)
            expr_part = f" {expr_echo}" if len(expr_echo) <= 60 else ""
            gap = 24 - value
            if gap > 0:
                direction = f"{self._fmt(gap)} short of 24 — make the result larger"
            else:
                direction = f"{self._fmt(-gap)} above 24 — make the result smaller"
            return False, (
                f"You used the right numbers, but your expression{expr_part} equals "
                f"{self._fmt(value)}, which is {direction}."
            )

        except ZeroDivisionError:
            return False, (
                f"Your expression divides by zero. Rearrange it so that no division "
                f"by zero occurs. {goal_hint}"
            )
        except Exception:
            hint = f" '{predicted_answer}'" if len(predicted_answer) <= 60 else ""
            return False, (
                f"Your expression{hint} could not be evaluated. Use only {nums_str} with "
                f"+, -, ×, ÷ and balanced parentheses, and nothing else."
            )

    @staticmethod
    def _fmt(n: Any) -> str:
        """Render a number without a trailing '.0' when it is integral."""
        try:
            f = float(n)
        except (TypeError, ValueError):
            return str(n)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.4g}"

    def _fmt_list(self, nums: List[Any]) -> str:
        return "[" + ", ".join(self._fmt(n) for n in nums) + "]"

    def _get_input_numbers(self, ground_truth: Any, params: Any) -> List:
        """Best-effort extraction of the puzzle's numbers from either source.

        For MM-HELIX 24Points, ``params`` is the parsed initial_state dict
        ``{'numbers': [...]}`` and ``ground_truth`` is a reference expression
        string. Stay robust to either being a dict, string, or None.
        """
        for src in (params, ground_truth):
            if isinstance(src, dict):
                if isinstance(src.get("numbers"), list):
                    return list(src["numbers"])
                inner = src.get("initial_state")
                if isinstance(inner, dict) and isinstance(inner.get("numbers"), list):
                    return list(inner["numbers"])
        return []

    def _describe_number_usage(self, used: List[int], expected: List[int]) -> str:
        """Describe precisely how the used numbers differ from what is required."""
        used_c = Counter(used)
        exp_c = Counter(expected)
        missing = exp_c - used_c   # needed more often than it was used
        extra = used_c - exp_c     # used too often, or not allowed at all
        msgs = []
        for num in sorted(missing):
            if used_c.get(num, 0) == 0:
                msgs.append(f"you did not use {self._fmt(num)}")
            else:
                msgs.append(
                    f"you used {self._fmt(num)} {used_c[num]}x but it should appear {exp_c[num]}x"
                )
        for num in sorted(extra):
            if exp_c.get(num, 0) == 0:
                msgs.append(f"you used {self._fmt(num)}, which is not one of the given numbers")
            else:
                msgs.append(
                    f"you used {self._fmt(num)} {used_c[num]}x but only {exp_c[num]} is available"
                )
        if not msgs:
            return "the set of numbers you used does not match the given numbers."
        return "; ".join(msgs) + "."

    @staticmethod
    def _pretty_expression(expression: str) -> str:
        """Convert a normalized expression back to human-friendly operators."""
        return expression.replace('*', '×').replace('/', '÷')

    def _normalize_expression(self, expression: str) -> str:
        """标准化表达式，统一运算符符号"""
        # 替换LaTeX格式的数学符号（注意：需要处理字面的反斜杠字符串）
        expression = expression.replace('\\times', '*')
        expression = expression.replace('\\div', '/')
        expression = expression.replace('\\cdot', '*')

        # 处理可能的转义序列残余
        expression = expression.replace('\times', '*')  # 制表符+imes -> *
        expression = expression.replace('imes', '*')     # 单独的imes -> *

        # 替换Unicode乘除符号为Python可以计算的符号
        expression = expression.replace('×', '*').replace('÷', '/')

        # 替换文字形式的运算符
        expression = re.sub(r'\bdiv\b', '/', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\bmul\b', '*', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\btimes\b', '*', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\bplus\b', '+', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\bminus\b', '-', expression, flags=re.IGNORECASE)

        # 移除空格
        expression = expression.replace(' ', '')

        # 移除可能包含的"="和之后的内容
        expression = re.sub(r'=.*$', '', expression)

        return expression

    def _extract_numbers(self, expression: str) -> List[int]:
        """从表达式中提取所有使用的数字"""
        return [int(num) for num in re.findall(r'\d+', expression)]

    def _evaluate_expression(self, expression: str) -> float:
        """
        计算表达式的值

        注意：使用eval函数存在安全风险，但在这个受控的评估环境中是可以接受的
        """
        # 检查表达式中是否只包含允许的字符
        if not re.match(r'^[\d\+\-\*/\(\)\.]+$', expression):
            raise ValueError("Expression contains invalid characters")

        # 计算表达式值
        return eval(expression)

if __name__ == "__main__":
    evaluator = TwentyFourPointsEvaluator()
    numbers = [2, 5, 10, 12]
    initial_state = {"numbers": numbers}
    reference = "(12 - 5) × 2 + 10"  # ground_truth is a reference expression string

    cases = [
        ("(12 - 5) \\times 2 + 10", "correct → 24"),
        ("(12 - 10) \\times 5 + 2", "right numbers, wrong value (too small)"),
        ("12 \\times 5 - 10 - 2", "right numbers, wrong value (too big)"),
        ("12 + 10 + 5", "missing a number (2)"),
        ("2 \\times 2 \\times 5 \\times 12", "used 2 twice, missing 10"),
        ("2 + 5 + 10 + 12 + 9", "used an extra number not given"),
        ("I think the answer is probably 24", "no expression"),
        ("5 \\div (12 - 10 - 2)", "division by zero"),
    ]

    for answer, label in cases:
        is_correct, feedback = evaluator.evaluate(answer, reference, initial_state)
        print(f"[{label}]")
        print(f"  input   : {answer}")
        print(f"  correct : {is_correct}")
        print(f"  feedback: {feedback}\n")