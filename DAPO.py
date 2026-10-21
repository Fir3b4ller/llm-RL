import os
import random
from dataclasses import dataclass, field, asdict
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import swanlab

from utils import extract_number, reward_fn, build_prompt, rollouts, token_logprobs, evaluate


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "openai/gsm8k"
    output_dir: str = "dapo_gsm8k_lora"
    swanlab_project: str = "DAPO-GSM8K"
    seed: int = 42

    # DAPO
    group_size: int = 4
    batch_size: int = 4
    mini_batch_size: int = 4          # Mini-batch size for training forward passes to save memory
    evaluation_batch_size: int = 64
    max_completion_length: int = 512
    temperature: float = 1
    top_p: float = 1
    epsilon_low: float = 0.2          # Lower clipping bound
    epsilon_high: float = 0.28        # Upper clipping bound
    overlong_cache_length: int = 128  # Linear penalty range
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    steps: int = 1000                  # one step = one dynamic rollout
    log_steps: int = 1

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


def overlong_penalty(lengths, max_len, cache_len):
    """Apply DAPO's linear soft penalty to overlong completions."""
    soft = max_len - cache_len
    return torch.clamp((soft - lengths).float() / cache_len, min=-1.0, max=0.0)


@torch.no_grad()
def dynamic_rollout(model, tokenizer, questions, golds, order, pointer, cfg, device):
    """Generate samples until batch_size non-degenerate groups have been collected."""
    group_size = cfg.group_size
    kept, sampled, dropped = [], 0, 0                 # kept: (output_ids, attn, comp_mask, rewards)
    while len(kept) < cfg.batch_size and pointer < len(order):
        chunk = order[pointer: pointer + cfg.batch_size]
        pointer += len(chunk)
        prompts = [build_prompt(tokenizer, questions[j]) for j in chunk]
        chunk_gold = [golds[j] for j in chunk]

        output_ids, attn_mask, comp_mask, completions = rollouts(model, tokenizer, prompts, cfg, device)
        correct = torch.tensor(
            [[reward_fn(completions[b * group_size + g], chunk_gold[b]) for g in range(group_size)]
             for b in range(len(chunk))],
            device=device,
        )                                              # (batch_size, group_size)
        lengths = comp_mask.sum(dim=1).reshape(len(chunk), group_size)
        rewards = correct + overlong_penalty(lengths, cfg.max_completion_length, cfg.overlong_cache_length)

        correct_count = correct.sum(dim=1)
        for b in range(len(chunk)):
            sampled += 1
            if correct_count[b] == 0 or correct_count[b] == group_size:  # Discard groups that are entirely incorrect or correct
                dropped += 1
                continue
            sl = slice(b * group_size, (b + 1) * group_size)
            kept.append((output_ids[sl], attn_mask[sl], comp_mask[sl], rewards[b]))
    return kept, sampled, dropped, pointer


def dapo_loss(new_logp, old_logp, advantages, mask, epsilon_low, epsilon_high):
    ratio = (new_logp - old_logp).exp()
    adv = advantages.unsqueeze(-1)
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - epsilon_low, 1 + epsilon_high) * adv
    pg_loss = -torch.minimum(unclipped, clipped)

    clip_count = (clipped.detach() < unclipped.detach()).float().mul(mask).sum()
    return (pg_loss * mask).sum(), clip_count


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

    order = list(range(len(train_question)))
    random.shuffle(order)
    pointer = 0
    progress = tqdm(total=cfg.steps, desc="train")

    for global_step in range(cfg.steps):
        if pointer >= len(order):
            random.shuffle(order)
            pointer = 0

        # dynamic rollout
        kept, sampled, dropped, pointer = dynamic_rollout(
            model, tokenizer, train_question, train_gold, order, pointer, cfg, device)
        degenerate_frac = dropped / max(sampled, 1)          # Fraction of sampled groups discarded as degenerate
        progress.update(1)
        if not kept:
            continue

        max_len = max(k[0].shape[1] for k in kept)           # Right-pad chunks of different lengths before concatenation
        pad = torch.nn.functional.pad
        output_ids = torch.cat([pad(k[0], (0, max_len - k[0].shape[1])) for k in kept])
        attn_mask = torch.cat([pad(k[1], (0, max_len - k[1].shape[1])) for k in kept])
        comp_mask = torch.cat([pad(k[2], (0, max_len - k[2].shape[1])) for k in kept])
        rewards = torch.stack([k[3] for k in kept])          # (batch_size, group_size)
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / \
                     (rewards.std(dim=1, keepdim=True) + 1e-4)
        advantages = advantages.reshape(-1)

        mask = comp_mask[:, 1:].float()                      # Align with log-probability positions
        total_tokens = mask.sum()
        n = output_ids.shape[0]
        old_chunks = []
        with torch.no_grad():
            for mb in range(0, n, cfg.mini_batch_size):
                sl = slice(mb, mb + cfg.mini_batch_size)
                old_chunks.append(token_logprobs(model, output_ids[sl], attn_mask[sl]))
        old_logp = torch.cat(old_chunks)

        # Mini-batched forward and backward passes
        optimizer.zero_grad()
        loss_sum = clip_count = 0.0
        for mb in range(0, n, cfg.mini_batch_size):
            sl = slice(mb, mb + cfg.mini_batch_size)
            new_logp = token_logprobs(model, output_ids[sl], attn_mask[sl])
            mb_loss, mb_clip = dapo_loss(
                new_logp, old_logp[sl], advantages[sl],
                mask[sl], cfg.epsilon_low, cfg.epsilon_high,
            )
            (mb_loss / total_tokens).backward()
            loss_sum += mb_loss.item()
            clip_count += mb_clip.item()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        optimizer.step()

        # log
        if global_step % cfg.log_steps == 0:
            reward_mean = rewards.mean().item()
            mean_len = mask.sum(dim=1).mean().item()
            swanlab.log({
                "train/loss": loss_sum / total_tokens.item(),
                "train/frac_degenerate": degenerate_frac,
                "train/clip_frac": clip_count / total_tokens.item(),
                "train/reward": reward_mean,
                "train/completion_len": mean_len,
            }, step=global_step)
            progress.set_postfix(reward=f"{reward_mean:.3f}", degenerate=f"{degenerate_frac:.2f}")

    progress.close()

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
