from datasets import load_dataset, Dataset


data = load_dataset("PAPOGalaxy/PAPO_ViRL39K_train")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import io
import sys
import json
import time
import base64
import argparse
from typing import Dict, Iterator, Tuple, Optional

from PIL import Image
from tqdm import tqdm

# Google Gen AI SDK (official)
# pip install -U google-genai
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
from google.genai import types
DEFAULT_MODEL = "gemini-2.5-pro"   # Good price/latency; change to 2.5 Pro for max reasoning.
DEFAULT_MAX_RETRIES = 5
DEFAULT_RATE_LIMIT_DELAY = 0.2  # seconds between requests (tune for your quota/project)

PROMPT_TEMPLATE = (
    "You are a helpful assistant. When responding to any user query, first provide a clear, step-by-step thinking trace explaining your reasoning process. Then, output only the final answer enclosed <answer> </answer> tags. With multiple choice questions, provide the correct option, e.g., A, B, C, Yes, No, at the end. Please strictly follow the format.\n"
    "Question: {question}"
)

def load_image_as_bytes(im) -> bytes:
    # Keep original encoding by re-saving losslessly as PNG in-memory
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def gemini_client():
    api_key = "AIzaSyCFr1X2Uv-lEiyEd2x0r1A1Wq1y6OIuexs"
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var. Get one from AI Studio and export it.")
    return genai.Client(api_key=api_key)

def build_multimodal_content(image_bytes: bytes, question: str) -> Content:
    """
    Gemini multimodal request: combine text prompt and image as parts.
    Docs: https://ai.google.dev/gemini-api/docs/image-understanding
    """
    # return Content(
    #     role="user",
    #     parts=[
    #         Part.text(PROMPT_TEMPLATE.format(question=question)),
    #         Part.inline_data(mime_type="image/png", data=image_bytes),
    #     ],
    # )
    return   [
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),
      PROMPT_TEMPLATE.format(question=question)
    ]

def call_gemini(client: genai.Client, model: str, content: Content, max_retries: int) -> str:
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=content,
                config=GenerateContentConfig(
                    temperature=0.1,  # Keep answers stable for evaluation
                ),
            )
            # Unified SDK returns candidates; resp.text is a convenient shortcut
            answer = (resp.text or "").strip()
            if not answer:
                answer = "unanswerable"
            return answer
        except Exception as e:
            if attempt == max_retries:
                return f"[ERROR] {e.__class__.__name__}: {e}"
            # simple exponential backoff
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)

def main():
    parser = argparse.ArgumentParser(description="Run VQA with Gemini Vision.")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Gemini model name (e.g., gemini-2.5-flash, gemini-2.5-pro)")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--rate_limit_delay", type=float, default=DEFAULT_RATE_LIMIT_DELAY)
    args = parser.parse_args()


    os.makedirs(args.output, exist_ok=True)

    # Build iterator

    client = gemini_client()

    n_total = 0
    n_missing = 0
    rows = []
    cnt = 0
    for datum in data["train"]:
        print("_"*30)
        # print('datum', datum)
        # print('datum["images"]', datum["images"])
        img = datum["images"][0]
        question = datum["problem"]
        n_total += 1
        answer = "Error"
        img_bytes = load_image_as_bytes(img)
        content = build_multimodal_content(img_bytes, question)
        answer = call_gemini(client, args.model, content, args.max_retries)
        # try:
        #     img_bytes = load_image_as_bytes(img)
        #     content = build_multimodal_content(img_bytes, question)
        #     answer = call_gemini(client, args.model, content, args.max_retries)

        # except Exception as e:
        #     print("e", e)
        #     n_missing += 1
        print("Question", question)
        print("Answer", answer)
        sol_match = re.search(r'<answer>(.*?)</answer>', answer)
        solution_extracted = sol_match.group(1).strip() if sol_match else answer.strip()  
        ground_truth = datum["answer"] 
        print("Extracted answer", solution_extracted)
        print("Ground truth", ground_truth)
        time.sleep(args.rate_limit_delay)
        datum["gemini_answer"] = answer
        rows.append(datum)
        if cnt==10:
            break
    ds = Dataset.from_list(rows)
    ds.to_parquet(f"{args.output}/ViRL39K.parquet")
    print(f"Done. Wrote: {args.output}")
    if n_missing:
        print(f"Warning: {n_missing} / {n_total} images missing.")

if __name__ == "__main__":
    main()
