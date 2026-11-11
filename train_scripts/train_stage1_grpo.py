"""6.6 Stage-1 GRPO 主训练 (自实现, 不依赖 trl).

每个 step 的逻辑:
  1) sample prompt
  2) actor (LoRA "default") 生成 G 条 rollout
  3) compute_reward 给每条 rollout 打分; 组内 reward 全相同 (std=0) 时无学习信号, 跳过该组
  4) group-relative advantage A_i = (r_i - mean) / (std + eps)
  5) ref (LoRA "ref", 冻结的 stage0 副本) 抽 assistant 段 token logprob
  6) actor forward 抽同样 token 的 logprob
  7) policy_loss = -mean(A_i * mean_logprob_actor)
     KL 用 k3 估计: exp(ref - actor) - (ref - actor) - 1, 恒 >= 0, 梯度把 actor 拉向 ref
  8) loss = policy_loss + beta * KL, backward, step

ref 必须是 stage0 SFT 模型而不是原始 base: 用 disable_adapter() 得到的是没学过 DPCoT 格式的 base,
KL 会把格式拉坏. 这里把 stage0 LoRA 再加载一份为冻结的 "ref" adapter, 共享 base 权重.
保存时只存 "default", 所以 stage1 ckpt 仍是 "base + 单个 LoRA", 评测脚本可直接加载.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from torch.optim import AdamW
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train_scripts"))
from dpcot import (build_user_prompt, build_system_prompt, parse_dpcot,
                   compute_reward)

MODEL_PATH = "/root/autodl-tmp/huggingface/Qwen3-VL-4B-Thinking"
TRAIN = ROOT / "data_build" / "annotated_train_20261105-0935.jsonl"
DEV = ROOT / "data_build" / "annotated_test_20261105-0935.jsonl"
STAGE0_CKPT = ROOT / "train_ckpt" / "stage0" / "final"
CKPT_DIR = ROOT / "train_ckpt" / "stage1"
LOG_PATH = ROOT / "train_logs" / "stage1.jsonl"
ROLLOUT_LOG_PATH = ROOT / "train_logs" / "stage1_rollouts.jsonl"
# 与 stage0 保持一致, 否则 LoRA 在不同视觉 token 分布上训练/评测
MAX_PIXELS = 1024 * 32 * 32
MIN_PIXELS = 64 * 32 * 32


def gpu_mem() -> str:
    a = torch.cuda.memory_allocated() / 2**30
    return f"{a:.1f}G"


def encode_prompt(processor, sample: dict):
    img = Image.open(ROOT / sample["image"]).convert("RGB")
    msgs = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": build_user_prompt(sample["question"])},
        ]},
    ]
    text = processor.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return processor(text=text, images=img, return_tensors="pt"), img


@torch.no_grad()
def rollout_one(model, processor, prompt_inputs, max_new_tokens=500,
                temperature=1.0, top_p=1.0):
    inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    full_ids = out[0]  # prompt + gen
    return text, full_ids, prompt_len


VISION_KEYS = ("pixel_values", "image_grid_thw")


@contextmanager
def ref_adapter(model):
    """临时切到冻结的 ref adapter. set_adapter 会同时改 requires_grad, 退出时切回 default 即恢复."""
    model.set_adapter("ref")
    try:
        yield
    finally:
        model.set_adapter("default")


def compute_logprobs(model, full_ids, prompt_len, vision_inputs: dict,
                     temperature: float, with_entropy: bool = False):
    """forward 一次, 返回 assistant 段每个 token 的 logprob (以及可选的平均 token 熵).

    logits 除以采样温度, 使 logprob 与 rollout 的采样分布一致.
    attention_mask 必须显式传: Qwen3-VL (transformers 4.57) 在无 mask 且无 KV cache 时
    (train 模式 + gradient checkpointing 会关掉 cache) 前向结果错误, 每 token logprob 偏差可达 20+ nats.
    """
    ids = full_ids.unsqueeze(0).to(model.device)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    inputs.update({k: v.to(model.device) for k, v in vision_inputs.items()})
    out = model(**inputs)
    # logits[t] 预测 input_ids[t+1]; 只取 assistant 段, 避免对整段 prompt 做 log_softmax
    logits = out.logits[0, prompt_len - 1:-1].float() / temperature
    target = full_ids[prompt_len:].to(model.device)
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(1, target.unsqueeze(-1)).squeeze(-1)
    if not with_entropy:
        return token_logp, None
    with torch.no_grad():
        entropy = -(logp.exp() * logp).sum(-1).mean()
    return token_logp, entropy


def grpo_step(model, processor, sample, args):
    """对一个 prompt 做 G 次 rollout, 返回 (loss 或 None, metrics, rollouts).

    loss 为 None 表示组内 reward 全相同, 该组不做更新.
    """
    prompt_inputs, _ = encode_prompt(processor, sample)
    vision_inputs = {k: prompt_inputs[k] for k in VISION_KEYS if k in prompt_inputs}

    rollouts = []  # list of (text, full_ids, prompt_len, reward_dict)
    model.eval()
    for _ in range(args.G):
        text, full_ids, plen = rollout_one(
            model, processor, prompt_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
        )
        rs = compute_reward(text, sample)
        rollouts.append((text, full_ids, plen, rs))

    rewards = torch.tensor([r[3]["R"] for r in rollouts], dtype=torch.float32)
    gen_lens = [full_ids.shape[0] - plen for _, full_ids, plen, _ in rollouts]
    metrics = {
        "mean_R": rewards.mean().item(),
        "max_R": rewards.max().item(),
        "min_R": rewards.min().item(),
        "std_R": rewards.std().item(),
        "mean_Rf": sum(r[3]["R_format"] for r in rollouts) / args.G,
        "mean_Rt": sum(r[3]["R_tag"] for r in rollouts) / args.G,
        "mean_Rb": sum(r[3]["R_behavior"] for r in rollouts) / args.G,
        "mean_len": sum(gen_lens) / args.G,
        "trunc_rate": sum(n >= args.max_new_tokens for n in gen_lens) / args.G,
    }

    if metrics["std_R"] < 1e-6:
        model.train()
        return None, metrics, rollouts

    std = rewards.std() + 1e-4
    advantages = (rewards - rewards.mean()) / std   # [G]

    with torch.no_grad(), ref_adapter(model):
        ref_logps = [
            compute_logprobs(model, full_ids, plen, vision_inputs, args.temperature)[0]
            for _, full_ids, plen, _ in rollouts
        ]
    model.train()

    policy_losses = []
    kl_terms = []
    entropies = []
    for i, (text, full_ids, plen, rs) in enumerate(rollouts):
        adv = advantages[i].to(model.device)
        logp_actor, entropy = compute_logprobs(
            model, full_ids, plen, vision_inputs, args.temperature, with_entropy=True)
        n_tok = max(logp_actor.numel(), 1)
        # 论文: L = E[A_i * log pi_theta(o_i|q)] - beta * KL
        # 我们 minimize: -A_i * mean_logp + beta * KL
        policy_losses.append(-adv * logp_actor.sum() / n_tok)
        log_ratio = ref_logps[i] - logp_actor
        kl_tok = torch.exp(log_ratio) - log_ratio - 1
        kl_terms.append(kl_tok.sum() / n_tok)
        entropies.append(entropy.item())

    policy_loss = torch.stack(policy_losses).mean()
    kl = torch.stack(kl_terms).mean()
    loss = policy_loss + args.beta * kl

    metrics.update({
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "entropy": sum(entropies) / len(entropies),
    })
    return loss, metrics, rollouts


def dump_rollouts(fp, step, n_groups, sample, rollouts, skipped):
    fp.write(json.dumps({
        "step": step, "group": n_groups, "skipped": skipped,
        "qid": sample["question_id"],
        "gt_category": sample["gt_category"], "gt_action": sample["gt_action"],
        "rollouts": [{
            "R": rs["R"], "Rf": rs["R_format"], "Rt": rs["R_tag"],
            "Rb": rs["R_behavior"], "len": int(full_ids.shape[0] - plen),
            "text": text[:3000],
        } for text, full_ids, plen, rs in rollouts],
    }, ensure_ascii=False) + "\n")
    fp.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=300,
                    help="优化器更新次数, 跳过的组不计入")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="<1 时采样分布与 logprob 不再严格一致")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--sample-every", type=int, default=10,
                    help="每 N 个 prompt 组把 rollout 原文写入 stage1_rollouts.jsonl")
    ap.add_argument("--max-consec-skip", type=int, default=50,
                    help="连续这么多组 std=0 则判定崩溃并提前停止")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-stage0", type=str, default=str(STAGE0_CKPT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_path, rollout_log_path = LOG_PATH, ROLLOUT_LOG_PATH
    if args.dry_run:
        args.max_steps = 3
        args.save_every = 100
        args.sample_every = 1
        log_path = LOG_PATH.with_name("stage1_dryrun.jsonl")
        rollout_log_path = ROLLOUT_LOG_PATH.with_name("stage1_rollouts_dryrun.jsonl")

    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
    random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"[load] processor + base {gpu_mem()}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    print(f"[load LoRA] actor + ref from {args.from_stage0}", flush=True)
    model = PeftModel.from_pretrained(base, args.from_stage0, is_trainable=True)
    model.load_adapter(args.from_stage0, adapter_name="ref", is_trainable=False)
    # load_adapter 会先按 base 的 bf16 建层再加载, ref 权重被舍入; 直接从 default 复制保证二者初始完全一致
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, p in params.items():
            if ".ref." in name:
                p.copy_(params[name.replace(".ref.", ".default.")])
    model.set_adapter("default")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ready] dt={time.time()-t0:.1f}s lora={n_train/1e6:.1f}M {gpu_mem()}",
          flush=True)

    train = [json.loads(l) for l in TRAIN.open()]
    random.shuffle(train)
    print(f"[data] train={len(train)}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)

    log = log_path.open("a")
    rollout_log = rollout_log_path.open("a")
    log.write(json.dumps({"event": "start", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "args": vars(args)}) + "\n")
    log.flush()

    step = 0
    n_groups = 0
    n_skip = 0
    consec_skip = 0
    aborted = False
    t_start = time.time()
    while step < args.max_steps and not aborted:
        random.shuffle(train)
        for s in train:
            try:
                loss, metrics, rollouts = grpo_step(model, processor, s, args)
            except torch.cuda.OutOfMemoryError as e:
                opt.zero_grad()
                torch.cuda.empty_cache()
                print(f"  [OOM] qid={s['question_id']} skip: {e}", flush=True)
                continue
            except Exception as e:
                opt.zero_grad()
                print(f"  [err] qid={s['question_id']} skip: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue
            n_groups += 1
            if n_groups % args.sample_every == 0:
                dump_rollouts(rollout_log, step, n_groups, s, rollouts, loss is None)

            if loss is None:
                n_skip += 1
                consec_skip += 1
                print(f"  [skip] std=0 R={metrics['mean_R']:.3f} Rf={metrics['mean_Rf']:.2f} "
                      f"len={metrics['mean_len']:.0f} (连续 {consec_skip}, 累计 {n_skip}/{n_groups})",
                      flush=True)
                log.write(json.dumps({"event": "skip", "step": step, "group": n_groups,
                                      **metrics}) + "\n")
                log.flush()
                if consec_skip >= args.max_consec_skip:
                    print(f"[abort] 连续 {consec_skip} 组 reward 无方差, 疑似崩溃, 停止训练",
                          flush=True)
                    aborted = True
                    break
                continue
            consec_skip = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            dt = time.time() - t_start
            if step % args.log_every == 0:
                print(f"[step {step}/{args.max_steps}] R={metrics['mean_R']:.3f} "
                      f"(min={metrics['min_R']:.2f} max={metrics['max_R']:.2f}) "
                      f"Rf={metrics['mean_Rf']:.2f} Rt={metrics['mean_Rt']:.2f} "
                      f"Rb={metrics['mean_Rb']:.2f} kl={metrics['kl']:.4f} "
                      f"ent={metrics['entropy']:.3f} len={metrics['mean_len']:.0f} "
                      f"trunc={metrics['trunc_rate']:.2f} loss={metrics['loss']:.3f} "
                      f"skip={n_skip}/{n_groups} {gpu_mem()} dt={dt:.0f}s",
                      flush=True)
            log.write(json.dumps({"event": "step", "step": step, "group": n_groups,
                                  **metrics}) + "\n")
            log.flush()

            if (not args.dry_run) and step % args.save_every == 0:
                save = CKPT_DIR / f"step_{step}"
                model.save_pretrained(save, selected_adapters=["default"])
                print(f"  [save] -> {save}", flush=True)

            if step >= args.max_steps:
                break

    print(f"[done] steps={step} groups={n_groups} skipped={n_skip} "
          f"total={time.time()-t_start:.0f}s", flush=True)
    if not args.dry_run:
        save = CKPT_DIR / "final"
        model.save_pretrained(save, selected_adapters=["default"])
        print(f"[final] -> {save}", flush=True)
    log.close()
    rollout_log.close()


if __name__ == "__main__":
    sys.exit(main())
