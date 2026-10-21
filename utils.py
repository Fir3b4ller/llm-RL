import re
import torch


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
    """从文本中抽取数字答案：取 '####' 后第一个数"""
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


def token_logprobs(model, input_ids, attention_mask):
    """近似计算softmax，省显存"""
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    gathered = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return gathered - logits.logsumexp(dim=-1)


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
