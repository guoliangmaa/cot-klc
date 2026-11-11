"""6.6 训练后 dev 评测: 对比 base / stage0 / stage1.

读 dev 集 (73 条), 三个模型分别贪婪解码生成 DPCoT 输出, 用 compute_reward 算分.
输出每条样本的 5 分量奖励, 三模型并排对比.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train_scripts"))
from dpcot import (build_user_prompt, build_system_prompt, parse_dpcot,
                   compute_reward)

MODEL_PATH = "/root/autodl-tmp/model/Intern-S1-mini"
DEV = ROOT / "data_build" / "05_dev.jsonl"
OUT_DIR = ROOT / "train_logs" / "dev_eval"


@torch.no_grad()
def eval_model(model, processor, samples: list, tag: str) -> list:
    print(f"\n===== eval [{tag}] on {len(samples)} samples =====", flush=True)
    rows = []
    for i, s in enumerate(samples, 1):
        img = Image.open(ROOT / s["image"]).convert("RGB")
        msgs = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": build_user_prompt(s["question"])},
            ]},
        ]
        text = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=600, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
        gen = processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rs = compute_reward(gen, s)
        rows.append({
            "qid": s["question_id"], "source": s["source"],
            "gt_category": s["gt_category"], "gt_action": s["gt_action"],
            "R": rs["R"], "Rf": rs["R_format"], "Rt": rs["R_tag"],
            "Rb": rs["R_behavior"], "Rl": rs["R_logic"], "Rk": rs["R_knowledge"],
            "pred_category": rs["parsed"].get("category"),
            "pred_action": rs["parsed"].get("action"),
            "format_ok": rs["parsed"].get("format_ok"),
            "raw": gen[:1500],
        })
        if i % 10 == 0 or i == len(samples):
            done = sum(1 for r in rows if r["Rf"] == 1)
            cat_ok = sum(1 for r in rows if r["pred_category"] == r["gt_category"])
            print(f"  [{i}/{len(samples)}] avg_R={sum(r['R'] for r in rows)/len(rows):.3f} "
                  f"format_ok={done}/{len(rows)} cat_ok={cat_ok}/{len(rows)}",
                  flush=True)
    return rows


def summarize(name: str, rows: list) -> dict:
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    cat_ok = sum(1 for r in rows if r["pred_category"] == r["gt_category"]) / n
    act_ok = sum(1 for r in rows if r["pred_action"] == r["gt_action"]) / n
    # 分 SIUO/MOSSBench 看
    siuo = [r for r in rows if r["source"] == "siuo"]
    moss = [r for r in rows if r["source"] == "mossbench"]
    siuo_safety = sum(1 for r in siuo if r["pred_action"] == "refuse") / max(len(siuo), 1)
    moss_refuse = sum(1 for r in moss if r["pred_action"] == "refuse") / max(len(moss), 1)
    out = {
        "name": name, "n": n,
        "avg_R": avg("R"), "avg_Rf": avg("Rf"), "avg_Rt": avg("Rt"),
        "avg_Rb": avg("Rb"), "avg_Rl": avg("Rl"), "avg_Rk": avg("Rk"),
        "category_acc": cat_ok, "action_acc": act_ok,
        "siuo_safety_score": siuo_safety,
        "moss_over_refusal": moss_refuse,
    }
    print(f"\n=== summary {name} ===")
    for k in ["avg_R", "avg_Rf", "avg_Rt", "avg_Rb", "avg_Rl", "avg_Rk",
             "category_acc", "action_acc", "siuo_safety_score", "moss_over_refusal"]:
        print(f"  {k}: {out[k]:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="只评前 N 条样本, 0=全量")
    ap.add_argument("--ckpts", type=str,
                    default="base,train_ckpt/stage0/step_100,train_ckpt/stage1/final",
                    help="逗号分隔, 'base' 表示无 LoRA")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--data", type=str, default=str(DEV),
                    help="评测数据 jsonl 路径，默认 data_build/05_dev.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    samples = [json.loads(l) for l in data_path.open()]
    if args.limit:
        samples = samples[:args.limit]
    print(f"[data] {data_path} n={len(samples)}")

    print(f"[load] base + processor")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    base.eval()

    summaries = []
    for ckpt in args.ckpts.split(","):
        ckpt = ckpt.strip()
        tag = "base" if ckpt == "base" else Path(ckpt).name
        if ckpt == "base":
            model = base
        else:
            model = PeftModel.from_pretrained(base, str(ROOT / ckpt))
            model.eval()
        t0 = time.time()
        rows = eval_model(model, processor, samples, tag)
        print(f"  [{tag}] eval done in {time.time()-t0:.0f}s")
        with (out_dir / f"{tag}.jsonl").open("w") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        summaries.append(summarize(tag, rows))
        # 卸载 LoRA, 保留 base
        if ckpt != "base":
            model = model.unload()
        torch.cuda.empty_cache()

    print("\n\n========= 三方对比 =========")
    keys = ["avg_R", "avg_Rt", "avg_Rb", "avg_Rl", "category_acc",
            "action_acc", "siuo_safety_score", "moss_over_refusal"]
    name_w = max(len(s["name"]) for s in summaries)
    print(f"{'metric':25s} | " + " | ".join(f"{s['name']:>{name_w}}" for s in summaries))
    print("-" * (28 + (name_w + 3) * len(summaries)))
    for k in keys:
        row = " | ".join(f"{s[k]:>{name_w}.3f}" for s in summaries)
        print(f"{k:25s} | {row}")

    with (out_dir / "summary.json").open("w") as w:
        json.dump(summaries, w, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
