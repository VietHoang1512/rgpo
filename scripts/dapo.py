import json

from datasets import Dataset
from tqdm.auto import tqdm

data = []
idx = 0
with open("data/dapo-selected-30k.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
        data[-1]["id"] = len(data) 
        data[-1]["data_source"] = "dapo-selected-30k"
        data[-1]["ground_truth"] = data[-1]["answer"]
print(data[0])
# {'question': 
# 'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.',
#  'answer':
#  '34', 
# 'abstract_hint': 
# 'Think about the properties of angle bisectors and the given trigonometric value. How might the angle bisector theorem and the Pythagorean theorem help you relate the sides of the triangle? What relationship can you infer from the given ratio $\\frac{BD}{CD} = \\frac{3}{2}$ and the right angle at $\\angle BDC$?',
#  'medium_hint': 
# 'First, identify the given values: $\\sin \\angle A = \\frac{4}{5}$, $AD = 1$, and $\\frac{BD}{CD} = \\frac{3}{2}$. Since $\\angle A < 90^\\circ$, you can determine $\\cos \\angle A$ using the Pythagorean identity. Next, consider the angle bisector property and the right angle at $\\angle BDC$. Try expressing the lengths $BD$ and $CD$ in terms of a single variable using the given ratio. What trigonometric relationships can you use to relate these lengths to the sides of triangle $ABC$?', 
# 'direct_hint':
#  '### Solution:\n\n**Step 1: Determine the cosine of \\(\\angle A\\).**\nGiven \\(\\sin \\angle A = \\frac{4}{5}\\) and \\(\\angle A < 90^\\circ\\), we can use the Pythagorean identity to find \\(\\cos \\angle A\\):\n\\[\n\\sin^2 \\angle A + \\cos^2 \\angle A = 1\n\\]\n\\[\n\\left(\\frac{4}{5}\\right)^2 + \\cos^2 \\angle A = 1\n\\]\n\\[\n\\frac{16}{25} + \\cos^2 \\angle A = 1\n\\]\n\\[\n\\cos^2 \\angle A = 1 - \\frac{16}{25} = \\frac{9}{25}\n\\]\n\\[\n\\cos \\angle A = \\frac{3}{5} \\quad (\\text{since } \\angle A < 90^\\circ)\n\\]\n\n**Step 2: Use the angle bisector theorem.**\nSince \\(\\angle BAD = \\angle DAC\\), \\(AD\\) is the angle bisector of \\(\\angle BAC\\). By the Angle Bisector Theorem:\n\\[\n\\frac{BD}{CD} = \\frac{AB}{AC} = \\frac{3}{2}\n\\]\nLet \\(AB = 3k\\) and \\(AC = 2k\\).\n\n**Step 3: Use the given condition \\(\\angle BDC = 90^\\circ\\).**\nSince \\(\\angle BDC = 90^\\circ\\), triangle \\(BDC\\) is a right triangle. We can use the Pythagorean theorem in \\(\\triangle BDC\\):\n\\[\nBD^2 + CD^2 = BC^2\n\\]\nGiven \\(\\frac{BD}{CD} = \\frac{3}{2}\\), let \\(BD = 3x\\) and \\(CD = 2x\\). Then:\n\\[\n(3x)^2 + (2x)^2 = BC^2\n\\]\n\\[\n9x^2 + 4x^2 = BC^2\n\\]\n\\[\n13x^2 = BC^2\n\\]\n\\[\nBC = x\\sqrt{13}\n\\]\n\n**Step 4: Use the Law of Cosines in \\(\\triangle ABD\\) and \\(\\triangle ACD\\).**\nWe know \\(AD = 1\\). Using the Law of Cosines in \\(\\triangle ABD\\):\n\\[\nAB^2 = AD^2 + BD^2 - 2 \\cdot AD \\cdot BD \\cdot \\cos \\angle BAD\n\\]\n\\[\n(3k)^2 = 1^2 + (3x)^2 - 2 \\cdot 1 \\cdot 3x \\cdot \\cos \\angle BAD\n\\]\n\\[\n9k^2 = 1 + 9x^2 - 6x \\cos \\angle BAD\n\\]\n\nSimilarly, using the Law of Cosines in \\(\\triangle ACD\\):\n\\[\nAC^2 = AD^2 + CD^2 - 2 \\cdot AD \\cdot CD \\cdot \\cos \\angle CAD\n\\]\n\\[\n(2k)^2 = 1^2 + (2x)^2 - 2 \\cdot 1 \\cdot 2x \\cdot \\cos \\angle CAD\n\\]\n\\[\n4k^2 = 1 + 4x^2 - 4x \\cos \\angle CAD\n\\]\n\nSince \\(\\angle BAD = \\angle CAD\\), \\(\\cos \\angle BAD = \\cos \\angle CAD\\). Let \\(\\cos \\angle BAD = \\cos \\angle CAD = y\\). Then:\n\\[\n9k^2 = 1 + 9x^2 - 6xy\n\\]\n\\[\n4k^2 = 1 + 4x^2 - 4xy\n\\]\n\n**Step 5: Solve the system of equations.**\nSubtract the second equation from the first:\n\\[\n9k^2 - 4k^2 = (1 + 9x^2 - 6xy) - (1 + 4x^2 - 4xy)\n\\]\n\\[\n5k^2 = 5x^2 - 2xy\n\\]\n\\[\nk^2 = x^2 - \\frac{2xy}{5}\n\\]\n\n**Step 6: Use the relationship between \\(k\\) and \\(x\\).**\nFrom the Angle Bisector Theorem, we have:\n\\[\n\\frac{AB}{AC} = \\frac{3}{2} \\implies \\frac{3k}{2k} = \\frac{3}{2}\n\\]\nThis is consistent, so we need to find \\(k\\) and \\(x\\). We use the fact that \\(AD = 1\\):\n\\[\nAD = \\sqrt{AB \\cdot AC \\cdot \\frac{1 - \\cos \\angle BAC}{1 + \\cos \\angle BAC}}\n\\]\n\\[\n1 = \\sqrt{3k \\cdot 2k \\cdot \\frac{1 - \\frac{3}{5}}{1 + \\frac{3}{5}}}\n\\]\n\\[\n1 = \\sqrt{6k^2 \\cdot \\frac{\\frac{2}{5}}{\\frac{8}{5}}}\n\\]\n\\[\n1 = \\sqrt{6k^2 \\cdot \\frac{1}{4}}\n\\]\n\\[\n1 = \\sqrt{\\frac{3k^2}{2}}\n\\]\n\\[\n1 = \\frac{\\sqrt{3}k}{\\sqrt{2}}\n\\]\n\\[\n\\sqrt{2} = \\sqrt{3}k\n\\]\n\\[\nk = \\frac{\\sqrt{2}}{\\sqrt{3}} = \\frac{\\sqrt{6}}{3}\n\\]\n\n**Step 7: Calculate \\(AB + AC\\).**\n\\[\nAB + AC = 3k + 2k = 5k = 5 \\cdot \\frac{\\sqrt{6}}{3} = \\frac{5\\sqrt{6}}{3}\n\\]\n\n**Step 8: Express the final answer.**\n\\[\na = 5, \\quad b = 6, \\quad c = 3\n\\]\n\\[\na + b + c = 5 + 6 + 3 = 14\n\\]\n\nThus, the final answer is:\n\\[\n\\boxed{14}\n\\]',
#  'id': 1,
#  'problem': 
# 'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.',
#  'data_source':
#  'dapo-selected-30k'}

ds = Dataset.from_list(data)
ds.to_parquet("data/dapo-selected-30k.parquet")
print(len(ds))