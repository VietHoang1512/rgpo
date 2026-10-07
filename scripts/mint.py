import json
import os
import re
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image as HFImage
from huggingface_hub import create_repo
from PIL import Image
from tqdm.auto import tqdm

with open("./data/MINT-CoT_interleave_sft_54k.json", "r", encoding="utf-8") as f:
    data = json.load(f)

FINAL_ANS_PATTERN = re.compile(r"(###\s*The final answer is:\s*\n)([^\n]*)(?=\n|$)", flags=re.IGNORECASE)

def replace_final_answer_block(answer_text: str, ground_truth: str) -> str:
    """Replace the content after the '### The final answer is:' line with normalized {ground_truth}.
    If the block doesn't exist, append one at the end."""
    replacement_line = r"\1 " + ground_truth
    if FINAL_ANS_PATTERN.search(answer_text):
        return FINAL_ANS_PATTERN.sub(replacement_line, answer_text, count=1) + "."
    # If no block found, append a well-formed section
    # Ensure the answer ends with a single newline
    tail = "" if answer_text.endswith("\n") else "\n"
    return answer_text + f"{tail}### The final answer is:\n {ground_truth}."

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False  
def resize_img(img, max_size=512):
    w, h = img.size
    if max(w, h) <= max_size:    # already small enough
        return img
    print(f"Resizing image from ({w}, {h}) to fit within {max_size}x{max_size}")
    scale = max_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)
    
cnt=0     
rows = []
for i, datum in tqdm(enumerate(data), total=len(data)):
    assert len(datum['messages'])==2
    question = datum['messages'][0]['content'].strip()
    answer = datum['messages'][1]['content'].replace("\nrac{36}{13}", "\nfrac{36}{13}").strip().strip(".").strip() + "."
    sol_match = re.search(r'### The final answer is:\n(.+)', answer)
    if sol_match:
        ground_truth = sol_match.group(1).strip().strip(".").strip() 
        
    else:
        ground_truth = sol.strip().strip(".").strip()
    
    option_match = re.search(r'option (.+)', ground_truth)
    ground_truth = option_match.group(1).strip().strip(".").strip() if option_match else ground_truth.strip().strip(".").strip()
    if ground_truth.startswith("A. "):
        ground_truth = "A"
    elif ground_truth.startswith("B. "):
        ground_truth = "B"
    elif ground_truth.startswith("C. "):
        ground_truth = "C"
    elif ground_truth.startswith("D. "):
        ground_truth = "D"      
    elif ground_truth.startswith("E. "):
        ground_truth = "E"     
    # if sol_match:    
    #     if  ground_truth != 
    #     answer = answer.replace()           
    # normalized_ground_truth = ground_truth
    
    # normalized_answer = replace_final_answer_block(answer, ground_truth)
    # print("_"*50)
    # print("answer", answer)
    # print("normalized_answer", normalized_answer)
    if len(ground_truth)>1 and not is_float(ground_truth):
        
        # print("*"*20)
        # print(i)
        # print("_")
        # print(question)
        # print("_")
        # print(ground_truth)
        cnt += 1
    images = [resize_img(Image.open(path).convert("RGB")) for path in datum["images"].split(",")]
    problem = question.replace("Generate an image description based on the question.\nThen, provide a rationale to analyze the question.\nNext, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.\nFinally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx.\n\nFormat your response with the following sections, separated by ###:\n### Image Description:\n### Rationales:\n### Let's think step by step.\n### Step 1:\n### Step 2:\n...\n### The final answer is: \n\nQuestion: ", "").strip()
    rows.append({
        "id": i,
        "images": images,  # PIL image or path
        "problem" : problem,
        "ground_truth": ground_truth,
        # "answer": normalized_answer,
        "answer": answer,
        
        "question_original": question,
        "answer_original": answer,
        "data_source": "mint_cot_r1_512",
    })
    # print(rows[-1])


ds = Dataset.from_list(rows)
ds.to_parquet("data/mint_cot_r1_512.parquet")
# ds.to_json(f"data/mint_cot_r1.jsonl", lines=True, force_ascii=False)  # JSONL
# dsd = DatasetDict({"train": ds})
# create_repo("phanviethoang1512/MINT-CoT-R1", repo_type="dataset", private=False, exist_ok=True)
# dsd.save_to_disk("MINT-CoT-R1")
# dsd.push_to_hub("phanviethoang1512/MINT-CoT-R1", private=False)
# print(f"Pushed {len(ds)} rows")
