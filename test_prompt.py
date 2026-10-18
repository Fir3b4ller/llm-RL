"""
View what the full prompt looks like after build_prompt goes through tokenizer.apply_chat_template
"""

import os
import random
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_dataset
from transformers import AutoTokenizer
from utils import build_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    question = ds[random.Random(0).randrange(len(ds))]["question"]   # Sample a fixed question

    prompt = build_prompt(tokenizer, question)

    print("=" * 90)
    print("origin_question:\n" + question)
    print("=" * 90)
    print("After apply_chat_template(repr):\n")
    print(repr(prompt))
    print("=" * 90)
    print("Readable form:\n")
    print(prompt)
    print("=" * 90)
    print("the number of prompt tokens:", len(tokenizer(prompt, add_special_tokens=False).input_ids))


if __name__ == "__main__":
    main()