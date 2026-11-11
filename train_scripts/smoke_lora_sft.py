"""6.1 LoRA SFT 冒烟 —— 验证 Intern-S1-mini + peft + bf16 在单卡上能跑通
forward + backward + LoRA 梯度更新, 不上 trl, 只测最底层 PyTorch 通路.

成功标志:
  - 模型加载不 OOM
  - forward 出 loss
  - loss.backward() 后 LoRA 层的 grad 不为 None
  - optimizer.step() 后某个 LoRA 权重发生变化
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = "/root/autodl-tmp/model/Intern-S1-mini"
TRAIN_JSONL = ROOT / "data_build" / "05_siuo_trace_lite.jsonl"


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "no-cuda"
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    return f"alloc={a:.1f}G reserved={r:.1f}G"


def main() -> int:
    print(f"[t=0.0s] start. {gpu_mem()}", flush=True)
    t0 = time.time()

    # 1) processor + base
    print(f"[load processor]", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[load model bf16] {gpu_mem()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    print(f"[loaded] dt={time.time()-t0:.1f}s {gpu_mem()}", flush=True)

    # 列一下顶层模块名，决定 LoRA target_modules
    print("\n=== top-level module names (前 25) ===")
    for i, (n, _) in enumerate(model.named_modules()):
        if i >= 25:
            break
        print(" ", n)

    # 2) 找出 LoRA 应该挂的线性层 —— 仅限语言塔，避免污染图像编码
    # 用 full path 列表传给 peft, peft 会做后缀匹配, 同名只挂指定路径
    target_modules = [
        n for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and n.startswith("model.language_model.")
        and n.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
    ]
    print(f"\n[lora target_modules] {len(target_modules)} 个 (例: {target_modules[0]})",
          flush=True)
    assert target_modules, "找不到 language_model 下的 q/k/v/o_proj"

    # 3) 包 LoRA
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[peft] trainable={n_train/1e6:.1f}M  total={n_total/1e9:.2f}B  "
          f"frac={n_train/n_total*100:.3f}% {gpu_mem()}", flush=True)

    # 4) 取一条 SIUO 训练样本（带图）做 forward
    sample = None
    with TRAIN_JSONL.open() as f:
        for l in f:
            r = json.loads(l)
            if r["source"] == "siuo":
                sample = r; break
    assert sample, "训练集为空?"
    img = Image.open(ROOT / sample["image"]).convert("RGB")
    print(f"\n[sample] qid={sample['question_id']} cat={sample['gt_category']}", flush=True)

    # 用 reference_answer 当 SFT target，构造 user/assistant 消息对
    user_text = f"Question: {sample['question']}\nAnswer:"
    target = sample.get("reference_answer") or "I cannot help with that."
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text},
        ]},
        {"role": "assistant", "content": target},
    ]
    rendered = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    inputs = processor(text=rendered, images=img, return_tensors="pt").to(model.device)
    inputs["labels"] = inputs["input_ids"].clone()
    print(f"[batch] seq_len={inputs['input_ids'].shape[1]} {gpu_mem()}", flush=True)

    # 5) forward + backward + step
    model.train()
    # 启用梯度检查点省显存
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        except Exception as e:
            print(f"[warn] gc not enabled: {e}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    print(f"\n[step 1] forward ...", flush=True)
    out = model(**inputs)
    loss = out.loss
    print(f"  loss={loss.item():.4f} {gpu_mem()}", flush=True)
    print(f"[step 1] backward ...", flush=True)
    loss.backward()
    # 检查 LoRA 是否拿到梯度
    grads_ok = 0; grads_zero = 0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.abs().sum().item() > 0:
                grads_ok += 1
            else:
                grads_zero += 1
    print(f"  LoRA params with grad: ok={grads_ok}  zero={grads_zero}", flush=True)
    # 记 1 个 LoRA 权重，看 step 是否真的改变它
    sample_param = next(p for n, p in model.named_parameters()
                        if p.requires_grad and "lora_A" in n)
    before = sample_param.detach().clone()
    opt.step(); opt.zero_grad()
    diff = (sample_param - before).abs().sum().item()
    print(f"  param updated by L1={diff:.6f}  {gpu_mem()}", flush=True)

    print(f"\n[done] total={time.time()-t0:.1f}s peak {gpu_mem()}", flush=True)
    return 0 if (grads_ok > 0 and diff > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
