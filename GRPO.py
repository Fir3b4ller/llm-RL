import os
import re
import random
from dataclasses import dataclass, field, asdict
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import swanlab


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "openai/gsm8k"
    output_dir: str = "grpo_gsm8k_lora"
    swanlab_project: str = "GRPO-GSM8K"
    seed: int = 42

    # GRPO
    group_size: int = 4
    batch_size: int = 4               # rollout = batch_size * group_size
    mini_batch_size: int = 4          # 训练前向的微批大小，省显存
    evaluation_batch_size: int = 64   # 评测时的批大小
    max_completion_length: int = 1024
    temperature: float = 1
    top_p: float = 1
    kl_coef: float = 0.04
    epsilon: float = 0.2              # PPO clif range
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_epochs: int = 1
    log_steps: int = 1

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


QUESTION_SUFFIX = " End your reply with the final numerical answer (no units) on its own line as '#### <number>'."
FEWSHOT_PREFIX = [
    {
        "role": "user",
        "content": (
            "A bakery sold 9 loaves in the morning and 6 loaves in the afternoon. "
            "How many loaves did it sell that day?" + QUESTION_SUFFIX
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Morning sales are 9 loaves and afternoon sales are 6 loaves, "
            "so the total is 9 + 6 = 15 loaves. "
            "#### 15"
        ),
    },
]


def normalize_answer(text: str) -> str:
    """GSM8K 答案做归一化，避免1,000和1000被判成不同。"""
    return text.replace(",", "")


def extract_number(text: str):
    """从文本中抽取数字答案：优先取 '####' 后第一个数，否则取第一个数字。"""
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[0] if nums else None


def reward_fn(completion: str, gold: str) -> float:
    """两边答案归一化后完全一致得 1.0，否则 0.0。"""
    pred = extract_number(completion)
    return 1.0 if pred is not None and normalize_answer(pred) == normalize_answer(gold) else 0.0


def build_prompt(tokenizer, question: str) -> str:
    messages = FEWSHOT_PREFIX + [{"role": "user", "content": question + QUESTION_SUFFIX}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def rollouts(model, tokenizer, prompt_texts, cfg, device):
    """对每个 prompt 采样 G 条 completion"""
    model.config.use_cache = True                                # 生成时开启 KV cache
    prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True, padding_side="left",
                    add_special_tokens=False,)
    prompt_len = prompt_tokens.input_ids.shape[1]
    input_ids = prompt_tokens.input_ids.repeat_interleave(cfg.group_size, dim=0).to(device)
    attention_mask = prompt_tokens.attention_mask.repeat_interleave(cfg.group_size, dim=0).to(device)

    output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=cfg.max_completion_length,
        do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False                               # 训练时关闭

    attention_mask = (output != tokenizer.pad_token_id).long()
    completion_mask = attention_mask.clone()
    completion_mask[:, :prompt_len] = 0                          # 只对生成部分算 loss
    completions = [
        tokenizer.decode(output[i, prompt_len:], skip_special_tokens=True)
        for i in range(output.shape[0])
    ]
    return output, attention_mask, completion_mask, completions


# GRPO 目标
def token_logprobs(model, input_ids, attention_mask):
    """近似计算softmax，省显存"""
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    gathered = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return gathered - logits.logsumexp(dim=-1)


def grpo_loss(new_logp, old_logp, ref_logp, advantages, mask, kl_coef, epsilon):
    ratio = (new_logp - old_logp).exp()
    adv = advantages.unsqueeze(-1)
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * adv
    pg_loss = -torch.minimum(unclipped, clipped)

    delta = ref_logp - new_logp                                  # k3 估计的无偏 KL
    kl = delta.exp() - delta - 1.0

    clip_count = ((ratio - 1.0).abs() > epsilon).float().mul(mask).sum()  # ratio 落在裁剪区间外的位置
    return ((pg_loss + kl_coef * kl) * mask).sum(), clip_count


@torch.no_grad()
def evaluate(model, tokenizer, questions, golds, cfg, device):
    model.eval()
    correct = 0
    for i in range(0, len(questions), cfg.evaluation_batch_size):
        batch_q = questions[i:i + cfg.evaluation_batch_size]
        batch_g = golds[i:i + cfg.evaluation_batch_size]
        prompts = [build_prompt(tokenizer, q) for q in batch_q]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left",
            add_special_tokens=False,
        ).to(device)
        output = model.generate(
            **enc, max_new_tokens=cfg.max_completion_length,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
        completions = tokenizer.batch_decode(output[:, enc.input_ids.shape[1]:],
                                             skip_special_tokens=True)
        for c, g in zip(completions, batch_g):
            correct += int(reward_fn(c, g) > 0.5)
    model.train()
    return correct / max(len(questions), 1)


def main():
    cfg = Config()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    swanlab.init(project=cfg.swanlab_project, config=asdict(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # data
    train_dataset = load_dataset(cfg.dataset_name, "main", split="train")
    test_dataset = load_dataset(cfg.dataset_name, "main", split="test")
    train_question = train_dataset["question"]
    train_gold = [extract_number(a) for a in train_dataset["answer"]]
    test_question = test_dataset["question"]
    test_gold = [extract_number(a) for a in test_dataset["answer"]]
    print(f"train={len(train_question)}  eval={len(test_question)}")

    group_size = cfg.group_size
    global_step = 0

    for epoch in range(cfg.num_epochs):
        order = list(range(len(train_question)))
        random.shuffle(order)
        progress = tqdm(range(0, len(order), cfg.batch_size), desc=f"epoch {epoch}")

        for start in progress:
            idx = order[start:start + cfg.batch_size]
            prompt_texts = [build_prompt(tokenizer, train_question[j]) for j in idx]
            golds = [train_gold[j] for j in idx]
            batch_size = len(idx)

            # rollout
            output_ids, attn_mask, comp_mask, completions = rollouts(
                model, tokenizer, prompt_texts, cfg, device)

            # reward and advantage
            rewards = torch.tensor(
                [[reward_fn(completions[b * group_size + g], golds[b]) for g in range(group_size)] for b in range(batch_size)],
                device=device,
            ) # shape: (batch_size, group_size)
            advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / \
                         (rewards.std(dim=1, keepdim=True) + 1e-4)
            advantages = advantages.reshape(-1)
            degenerate_frac = (rewards.std(dim=1) < 1e-6).float().mean().item()  # 组内奖励相同: 无优势信号

            mask = comp_mask[:, 1:].float()  # 对齐到 logprob 位置
            total_tokens = mask.sum()
            n = output_ids.shape[0]
            old_chunks, ref_chunks = [], []
            with torch.no_grad():
                for mb in range(0, n, cfg.mini_batch_size):
                    sl = slice(mb, mb + cfg.mini_batch_size)
                    old_chunks.append(token_logprobs(model, output_ids[sl], attn_mask[sl]))
                    with model.disable_adapter():
                        ref_chunks.append(token_logprobs(model, output_ids[sl], attn_mask[sl]))
            old_logp = torch.cat(old_chunks)
            ref_logp = torch.cat(ref_chunks)

            # 微批前向 + 反向
            optimizer.zero_grad()
            loss_sum = clip_count = 0.0
            for mb in range(0, n, cfg.mini_batch_size):
                sl = slice(mb, mb + cfg.mini_batch_size)
                new_logp = token_logprobs(model, output_ids[sl], attn_mask[sl])
                mb_loss, mb_clip = grpo_loss(
                    new_logp, old_logp[sl], ref_logp[sl], advantages[sl],
                    mask[sl], cfg.kl_coef, cfg.epsilon,
                )
                (mb_loss / total_tokens).backward()
                loss_sum += mb_loss.item()
                clip_count += mb_clip.item()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optimizer.step()

            # log
            with torch.no_grad():
                delta = ref_logp - old_logp
                kl = ((delta.exp() - delta - 1.0) * mask).sum() / total_tokens
            if global_step % cfg.log_steps == 0:
                reward_mean = rewards.mean().item()
                mean_len = mask.sum(dim=1).mean().item()
                swanlab.log({
                    "train/loss": loss_sum / total_tokens.item(),
                    "train/frac_degenerate": degenerate_frac,
                    "train/clip_frac": clip_count / total_tokens.item(),
                    "train/reward": reward_mean,
                    "train/kl": kl.item(),
                    "train/completion_len": mean_len,
                }, step=global_step)
                progress.set_postfix(reward=f"{reward_mean:.3f}", kl=f"{kl.item():.4f}")

            global_step += 1

    # evaluate
    acc = evaluate(model, tokenizer, test_question, test_gold, cfg, device)
    print(f"final eval accuracy on full test set = {acc:.4f}")

    # save LoRA adapters
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # save final eval accuracy
    acc_path = os.path.join(cfg.output_dir, "eval_accuracy.txt")
    with open(acc_path, "w", encoding="utf-8") as f:
        f.write(f"final eval accuracy on full test set = {acc:.6f}\n")

    swanlab.finish()
    print(f"saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()