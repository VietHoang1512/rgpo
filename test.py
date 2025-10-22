
# Fix: escape literal '#' characters because re.VERBOSE treats '#' as comment starters.
import re
import pandas as pd

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



def follows_structure(text: str) -> bool:
    return 1. if STRUCTURE_RE.search(text) is not None else 0.

# ---------- Build test cases ----------
valid_single_step = """### Image Description:
The image shows a circle with a diameter $\overline{AB}$. Points $A$ and $B$ are on the circumference, and point $C$ is on the circle such that $AC$ and $BC$ are chords. The lengths of $AC$ and $BC$ are given as 8 inches and 15 inches, respectively. The question asks to find the radius of the circle.

### Rationales:
1. Since $\overline{AB}$ is a diameter, the length of $\overline{AB}$ is twice the radius of the circle.
2. The triangle $ABC$ is a right triangle because $\overline{AB}$ is the diameter, making $\angle ACB$ a right angle (by the Thales' theorem).
3. Using the Pythagorean theorem in $\triangle ABC$, we can find the length of $\overline{AB}$.
4. Once we have the length of $\overline{AB}$, we can find the radius by dividing the diameter by 2.

### Let's think step by step.
### Step 1:
Identify the right triangle $\triangle ABC$ with $\angle ACB = 90^\circ$.
### Step 2:
Apply the Pythagorean theorem to find the length of $\overline{AB}$:
\[
AB^2 = AC^2 + BC^2
\]
\[
AB^2 = 8^2 + 15^2
\]
\[
AB^2 = 64 + 225
\]
\[
AB^2 = 289
\]
\[
AB = \sqrt{289} = 17 \text{ inches}
\]
### Step 3:
Since $\overline{AB}$ is the diameter, the radius $r$ is half of the diameter:
\[
r = \frac{AB}{2} = \frac{17}{2} = 8.5 \text{ inches}
\]

### The final answer is: C
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
        "follows_structure": follows_structure(text)
    })

df = pd.DataFrame(results)
print("Regex structure check results", df)

# Also print them plainly for quick glance
for row in results:
    print(f"{row['case']}: {row['follows_structure']}")
