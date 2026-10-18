"""
批量版：用原始 Qwen2.5-0.5B-Instruct 随机作答 n 道 GSM8K 题目，打印每道题及模型回答
"""
import os
import argparse
import random
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import build_prompt, extract_number, reward_fn

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs="?", default=32, help="随机抽取的题目数量")  # 1319
    parser.add_argument("--split", default="test", choices=["train", "test"], help="从哪个划分抽题")
    parser.add_argument("--max_tokens", type=int, default=1024, help="模型回答的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 表示贪心解码")
    parser.add_argument("--batch_size", type=int, default=32, help="一次并发生成的题目数，越大 GPU 占用率越高")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"                                # 批量生成必须左填充
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    idx = random.sample(range(len(dataset)), args.n)

    gen_kwargs = {"max_new_tokens": args.max_tokens, "pad_token_id": tokenizer.pad_token_id}
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)

    correct = 0
    done = 0
    for start in range(0, len(idx), args.batch_size):
        batch = idx[start:start + args.batch_size]
        prompts = [build_prompt(tokenizer, dataset[j]["question"]) for j in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(device)
        output = model.generate(**enc, **gen_kwargs)
        answers = tokenizer.batch_decode(
            output[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

        for j, answer in zip(batch, answers):
            done += 1
            question = dataset[j]["question"]
            gold = extract_number(dataset[j]["answer"])
            answer = answer.strip()

            ok = reward_fn(answer, gold) > 0.5
            correct += int(ok)

            print("=" * 90)
            print(f"[{done}/{args.n}] 题目: {question}")
            print(f"标准答案: {gold}")
            print(f"模型回答: {answer}")
            print(f"是否正确: {'正确' if ok else '错误'}")

    print("=" * 90)
    print(f"共 {args.n} 题，正确 {correct} 题，准确率 {correct / args.n:.2%}")


if __name__ == "__main__":
    main()
