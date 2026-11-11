# CoT-KLC

用 Qwen3-VL-4B-Thinking 的 LoRA 做两段训练：先 SFT（`train_stage0_sft.py`），再 GRPO（`train_stage1_grpo.py`）。

## 1. 项目结构

```text
cot-klc/
├── data_build/                           # 自行标注的 jsonl
│   ├── annotated_train_20261105-0935.jsonl   # 当前训练集
│   └── annotated_test_20261105-0935.jsonl    # 当前验证集
├── dataset/                              # SIUO 和图片
│   ├── siuo_gen.json                     # SIUO 标注，评测读取
│   ├── siuo_mcqa.json                    # SIUO 选择题，当前流程不读
│   ├── images/                           # SIUO 原图
│   ├── train/                            # 训练集图片，jsonl 的 image 指向这里
│   └── test/                             # 验证集图片
├── eval_siuo/                            # SIUO 评测
│   ├── batch.sh                          # 生成 → 合并 → safe/effective 评分 → 汇总
│   ├── generation-Qwen3-VL-4B-Thinking-gen.py
│   ├── mergejson.py
│   ├── gpt-eval.py
│   ├── gen_score.py
│   ├── wait_and_eval.sh
│   ├── gpt-call/                         # 生成结果
│   └── gpt_eval/                         # 评分结果
├── train_scripts/
│   ├── dpcot.py                          # 提示词和 reward
│   ├── train_stage0_sft.py               # SFT
│   ├── train_stage1_grpo.py              # GRPO，默认从 stage0/final 接着训
│   ├── eval_dev_compare.py
│   ├── test_reward.py
│   └── smoke_lora_sft.py
├── train_ckpt/                           # LoRA 权重，不含基座
│   ├── stage0/                           # SFT：step_50 … step_200、final
│   └── stage1/                           # GRPO：step_25 … step_300、final
├── requirements.txt
└── readme.md
```

**`data_build/` 和 `dataset/` 是数据。在 `huggingface` 中获取 https://huggingface.co/datasets/liang0812/siuopp **

`data_build/` 是自行标注的 jsonl。样本来自 VL-Guard、SPA-VL、Beavertails、MSSBench、HoliSafe、MSTS。两个训练脚本默认读取：

- `data_build/annotated_train_20261105-0935.jsonl`：训练集
- `data_build/annotated_test_20261105-0935.jsonl`：验证集

`dataset/` 放 SIUO 和图片。`siuo_gen.json`、`images/` 是 SIUO 评测用的标注和原图；`train/`、`test/` 是上面 jsonl 里 `image` 字段指向的训练图和验证图。

**`eval_siuo/` 用来评 SIUO。** 生成回答后，用 GPT 打 safe / effective，结果写到 `eval_siuo/gpt_eval/`。

```bash
bash eval_siuo/batch.sh train_ckpt/stage1/final
```

**`train_scripts/` 是训练代码。** `train_stage0_sft.py` 做 SFT，`train_stage1_grpo.py` 做 GRPO，提示词和 reward 在 `dpcot.py`。

**`train_ckpt/` 存训练权重。** 只有 LoRA，不含基座。SFT 写到 `train_ckpt/stage0/`，GRPO 写到 `train_ckpt/stage1/`。GRPO 默认从 `train_ckpt/stage0/final` 接着训。

## 2. 如何运行

1. 从 Hugging Face 下载基座，把两个脚本里的 `MODEL_PATH` 改成本地目录。

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Thinking --local-dir /path/to/Qwen3-VL-4B-Thinking
```

需要改的位置：

- `train_scripts/train_stage0_sft.py` 的 `MODEL_PATH`
- `train_scripts/train_stage1_grpo.py` 的 `MODEL_PATH`

2. 建环境并安装依赖。

```bash
conda create -n safegrpo python=3.12 -y
conda activate safegrpo
pip install -r requirements.txt
```

3. 先跑 SFT，再跑 GRPO。

```bash
python train_scripts/train_stage0_sft.py
python train_scripts/train_stage1_grpo.py
```
