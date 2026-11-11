"""SIUO 生成：本地加载 Qwen3-VL-4B-Thinking，可选挂 LoRA，每条样本采样 n 次.

加载方式与 train_scripts/train_stage1_grpo.py 相同：
AutoModelForImageTextToText 加载基座，像素上限与训练一致，再用 peft 挂 LoRA（--lora-path 为空则只用基座）。
prompt 与训练一致：DPCoT system prompt + "User question: ..."（train_scripts/dpcot.py），基座和 LoRA 都用它。
这条 checkpoint 的 chat template 在生成前总会先打开 <think>（忽略 enable_thinking）；
LoRA 会直接在 <think> 里输出 DPCoT 标签，常常不写 </think>；基座会先长篇思考，需要更大的 --max-new-tokens。
写入 responses 的是 <answer> 标签内容（取不到再退回 </think> 之后的正文），完整输出存在 raw_responses；
结果是 <output-root>/<model_tag>/gen/output_*.json，供 mergejson.py 合并。

完整流程见同目录 batch.sh：

    bash eval_siuo/batch.sh                                        # 纯基座
    LORA_PATH=train_ckpt/stage1/step_200 bash eval_siuo/batch.sh   # 挂 LoRA
"""
import argparse
import json
import logging
import os
import sys
import unicodedata
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "train_scripts"))
from dpcot import build_system_prompt, build_user_prompt, parse_dpcot  # noqa: E402

MODEL_PATH = "/root/autodl-tmp/huggingface/Qwen3-VL-4B-Thinking"
# 与 stage0 / stage1 一致，否则 LoRA 看到的视觉 token 分布和训练时不同
MAX_PIXELS = 1024 * 32 * 32
MIN_PIXELS = 64 * 32 * 32

MAX_RETRY = 3
forbidden_list = []


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model(model_path, lora_path=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[backend] loading processor + base from {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True,
        max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    if lora_path:
        from peft import PeftModel
        print(f"[backend] loading LoRA adapter from {lora_path}", flush=True)
        model = PeftModel.from_pretrained(base, lora_path)
    else:
        print("[backend] no LoRA, using vanilla base", flush=True)
        model = base
    model.eval()
    return processor, model


def visible_answer(text):
    """取交给裁判的回答：优先 <answer> 标签，其次 </think> 之后的正文，最后是全文。

    先在最后一个 </think> 之后找 <answer>：基座的思考草稿、LoRA 重复输出的残缺标签都在它前面；
    之后没有标签（LoRA 先输出完整标签再关 think、接着乱写散文）时再在全文里找第一个。
    """
    tail = text.rsplit("</think>", 1)[-1]
    for segment in (tail, text):
        answer = parse_dpcot(segment)["answer"]
        if answer:
            return answer
    if "<answer>" in text:
        logging.warning("<answer> 未闭合，可能被 max_new_tokens 截断，取已写出的部分")
        return text.rsplit("<answer>", 1)[-1].strip()
    logging.warning("输出里没有 <answer> 标签，退回 </think> 之后的正文")
    return tail.strip()


def generate_n(processor, model, prompt, image_path, n, max_new_tokens,
               temperature, top_p, top_k):
    import torch

    img = Image.open(image_path).convert("RGB")
    msgs = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": build_user_prompt(prompt)},
        ]},
    ]
    text = processor.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=text, images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=n if do_sample else 1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    raws = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    # 贪心解码时 n 次结果相同，只算一次再复制
    if len(raws) < n:
        raws = raws * n
    return [visible_answer(raw) for raw in raws], raws


def process_data(data, start_idx, end_idx, processor, model, args, output_dir):
    for idx in range(start_idx, end_idx + 1):
        item = data[idx]
        if item["question_id"] in forbidden_list:
            continue
        question_id = item["question_id"]
        out_file = f"{output_dir}/output_{idx}-{question_id}.json"
        if args.skip_existing and os.path.exists(out_file):
            print(f"{question_id} exists, skip", flush=True)
            continue
        logging.info(question_id)
        print(f"[{idx + 1}/{end_idx + 1}] {question_id}", flush=True)
        try:
            image_path = os.path.join(args.image_dir, item["image"])
            prompt = item["question"]
            # 空回答在安全评测中会被直接判为安全，所以整组重试
            resp_list, raw_list = [], []
            for attempt in range(MAX_RETRY):
                resp_list, raw_list = generate_n(
                    processor, model, prompt, image_path, args.n,
                    args.max_new_tokens, args.temperature, args.top_p, args.top_k,
                )
                if all(resp_list):
                    break
                logging.warning(
                    f"{question_id}: empty content, retry {attempt + 1}/{MAX_RETRY}")
            else:
                raise RuntimeError(f"empty content after {MAX_RETRY} retries")

            item["responses"] = [
                unicodedata.normalize("NFKC", resp) for resp in resp_list
            ]
            item["raw_responses"] = raw_list
            with open(out_file, "w", encoding="utf-8") as outfile:
                json.dump(item, outfile, ensure_ascii=False, indent=4)
        except Exception as e:
            with open(f"{output_dir}/error_log.txt", "a") as log_file:
                log_file.write(f"Error processing item at index {idx}: {str(e)}\n")
            logging.info(e)
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH,
                        help="Qwen3-VL-4B-Thinking 基座路径")
    parser.add_argument("--lora-path", default="",
                        help="LoRA adapter 目录；不传或传空字符串则只用基座")
    parser.add_argument("--model-tag", default="Qwen3-VL-4B-Thinking",
                        help="结果目录名，需与 mergejson.py / gpt-eval.py 的 --model 一致")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "gpt-call"),
                        help="mergejson.py 从 ./gpt-call 读取，需在其所在目录运行合并")
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "siuo_gen.json"))
    parser.add_argument("--image-dir", default=str(ROOT / "dataset" / "images"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n", type=int, default=3, help="每条样本采样次数")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None,
                        help="含该下标；默认处理到数据集末尾")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="默认跳过已生成的 output_*.json，便于中断后续跑")
    args = parser.parse_args()
    if not args.lora_path:
        args.lora_path = None
    elif not os.path.isdir(args.lora_path):
        raise SystemExit(f"LoRA 目录不存在: {args.lora_path}")

    output_dir = Path(args.output_root) / args.model_tag / "gen"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)
    logging.basicConfig(
        filename=f"{output_dir}/log_file.log", level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"base={args.model_path} lora={args.lora_path}")

    processor, model = load_model(args.model_path, args.lora_path)
    data = load_data(args.dataset)
    end_idx = len(data) - 1 if args.end is None else min(args.end, len(data) - 1)
    process_data(data, args.start, end_idx, processor, model, args, output_dir)


if __name__ == "__main__":
    main()
