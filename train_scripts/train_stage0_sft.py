"""6.5 Stage-0 SFT 冷启动训练 (LoRA on Intern-S1-mini).

目标: 让模型学会 DPCoT 输出格式 (R_format -> 0.9+)
做法: 监督 assistant 段, mask user/system, AdamW + bf16 + grad checkpointing.

用法:
  python train_scripts/train_stage0_sft.py                # 默认配置
  python train_scripts/train_stage0_sft.py --max-steps 50 --dry-run   # 冒烟
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train_scripts"))
from dpcot import (build_user_prompt, build_system_prompt, build_sft_target,
                   parse_dpcot, compute_reward)

MODEL_PATH = "/root/autodl-tmp/huggingface/Qwen3-VL-4B-Thinking"
TRAIN = ROOT / "data_build" / "annotated_train_20261105-0935.jsonl"
DEV = ROOT / "data_build" / "annotated_test_20261105-0935.jsonl"
CKPT_DIR = ROOT / "train_ckpt" / "stage0"
LOG_PATH = ROOT / "train_logs" / "stage0.jsonl"
# 默认 longest_edge=16M 像素, 10MP 图会产生 ~9.7k 视觉 token 直接 OOM; 每 32x32 像素 = 1 token
MAX_PIXELS = 1024 * 32 * 32
MIN_PIXELS = 64 * 32 * 32


def gpu_mem() -> str:
    a = torch.cuda.memory_allocated() / 2**30
    return f"alloc={a:.1f}G"


def encode_sample(processor, sample: dict, train_mode: bool = True):
    """构造 (input_ids, labels, pixel_values...). 在 train_mode 下 mask user/system."""
    img = Image.open(ROOT / sample["image"]).convert("RGB")
    sys_p = build_system_prompt()
    user_p = build_user_prompt(sample["question"])
    target = build_sft_target(sample)
    msgs_full = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_p},
        ]},
        {"role": "assistant", "content": target},
    ]
    msgs_prompt = msgs_full[:-1]

    full_text = processor.tokenizer.apply_chat_template(
        msgs_full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt_text = processor.tokenizer.apply_chat_template(
        msgs_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    full_inputs = processor(text=full_text, images=img, return_tensors="pt")
    if train_mode:
        prompt_inputs = processor(text=prompt_text, images=img, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full_inputs["labels"] = labels
    return full_inputs, img


def eval_dev(model, processor, dev_samples: list, n: int = 8) -> dict:
    """在 dev 上跑 n 条做生成, 计算 R_format / R_tag / R / hit_count."""
    model.eval()
    records = []
    for s in dev_samples[:n]:
        img = Image.open(ROOT / s["image"]).convert("RGB")
        sys_p = build_system_prompt()
        user_p = build_user_prompt(s["question"])
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_p},
            ]},
        ]
        text = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=600, do_sample=False,
                                 pad_token_id=processor.tokenizer.eos_token_id)
        gen = processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rs = compute_reward(gen, s)
        records.append({"qid": s["question_id"], **{k: v for k, v in rs.items()
                                                    if k != "parsed"}})
    model.train()
    if not records:
        return {}
    avg = {k: sum(r[k] for r in records) / len(records)
           for k in ["R", "R_format", "R_tag", "R_behavior", "R_logic", "R_knowledge"]}
    avg["n"] = len(records)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="只跑 5 步, 不存盘, 不评测")
    args = ap.parse_args()

    if args.dry_run:
        args.max_steps = 5
        args.save_every = 100
        args.eval_every = 100

    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

    random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"[load] processor + model {gpu_mem()}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    target_modules = [
        n for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and n.startswith("model.language_model.")
        and n.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
    ]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ready] dt={time.time()-t0:.1f}s lora={n_train/1e6:.1f}M targets={len(target_modules)} {gpu_mem()}",
          flush=True)

    train_samples = [json.loads(l) for l in TRAIN.open()]
    dev_samples = [json.loads(l) for l in DEV.open()]
    random.shuffle(train_samples)
    random.shuffle(dev_samples)
    print(f"[data] train={len(train_samples)} dev={len(dev_samples)}", flush=True)

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)

    log = LOG_PATH.open("a")
    model.train()
    step = 0
    micro = 0
    epoch = 0
    loss_sum = 0.0
    t_start = time.time()
    while step < args.max_steps:
        epoch += 1
        random.shuffle(train_samples)
        for s in train_samples:
            try:
                inputs, _ = encode_sample(processor, s, train_mode=True)
            except Exception as e:
                print(f"  [skip] qid={s['question_id']} encode err: {e}", flush=True)
                continue
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            try:
                out = model(**inputs)
                loss = out.loss / args.grad_accum
                loss.backward()
            except torch.OutOfMemoryError:
                print(f"  [skip] qid={s['question_id']} OOM "
                      f"seq_len={inputs['input_ids'].shape[1]}", flush=True)
                out = loss = None
                torch.cuda.empty_cache()
                continue
            loss_sum += loss.item() * args.grad_accum
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                avg_loss = loss_sum / args.grad_accum
                loss_sum = 0.0
                dt = time.time() - t_start
                print(f"[step {step}/{args.max_steps}] loss={avg_loss:.3f} "
                      f"epoch={epoch} {gpu_mem()} dt={dt:.0f}s", flush=True)
                log.write(json.dumps({"event": "step", "step": step,
                                      "loss": avg_loss, "epoch": epoch}) + "\n")
                log.flush()

                if (not args.dry_run) and step > 0 and step % args.eval_every == 0:
                    print(f"[eval@{step}] ...", flush=True)
                    ev = eval_dev(model, processor, dev_samples, args.eval_n)
                    print(f"  dev: R={ev.get('R',0):.3f} Rf={ev.get('R_format',0):.2f} "
                          f"Rt={ev.get('R_tag',0):.2f} Rb={ev.get('R_behavior',0):.2f}",
                          flush=True)
                    log.write(json.dumps({"event": "eval", "step": step, **ev}) + "\n")
                    log.flush()

                if (not args.dry_run) and step > 0 and step % args.save_every == 0:
                    save_path = CKPT_DIR / f"step_{step}"
                    model.save_pretrained(save_path)
                    print(f"  [save] -> {save_path}", flush=True)

                if step >= args.max_steps:
                    break

    print(f"[done] total={time.time()-t_start:.0f}s", flush=True)
    if not args.dry_run:
        save_path = CKPT_DIR / "final"
        model.save_pretrained(save_path)
        print(f"[final save] -> {save_path}", flush=True)
    log.close()


if __name__ == "__main__":
    sys.exit(main())
