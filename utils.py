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
    """The answers are normalized to avoid treating 1,000 and 1000 as different."""
    return text.replace(",", "")


def extract_number(text: str):
    """Extract the numerical answer from the text: Take the first number after '####' """
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[0] if nums else None


def reward_fn(completion: str, gold: str) -> float:
    """if the answers are exactly the same, the score is 1.0; otherwise 0.0"""
    pred = extract_number(completion)
    return 1.0 if pred is not None and normalize_answer(pred) == normalize_answer(gold) else 0.0


def build_prompt(tokenizer, question: str) -> str:
    messages = FEWSHOT_PREFIX + [{"role": "user", "content": question + QUESTION_SUFFIX}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def rollouts(model, tokenizer, prompt_texts, cfg, device):
    """Sample completions for each prompt"""
    model.config.use_cache = True                                # Enable KV cache during generation
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
    model.config.use_cache = False                               # Turn off during training

    attention_mask = (output != tokenizer.pad_token_id).long()
    completion_mask = attention_mask.clone()
    completion_mask[:, :prompt_len] = 0                          # Only calculate the loss for the generated part
    completions = [
        tokenizer.decode(output[i, prompt_len:], skip_special_tokens=True)
        for i in range(output.shape[0])
    ]
    return output, attention_mask, completion_mask, completions


def token_logprobs(model, input_ids, attention_mask):
    """Compute log probabilities via log-sum-exp to reduce memory usage."""
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
