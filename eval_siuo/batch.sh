#!/bin/bash
# SIUO 完整评测流程：生成 -> 合并 -> GPT 评 safe / effective -> 汇总得分
#
# 用法（在任意目录执行均可）:
#   bash eval_siuo/batch.sh                                         # 纯基座
#   LORA_PATH=train_ckpt/stage1/step_200 bash eval_siuo/batch.sh    # 挂 LoRA（相对仓库根目录或绝对路径）
#   bash eval_siuo/batch.sh train_ckpt/stage1/20261101/final        # 也可以把 LoRA 作为第一个参数
# 其他可覆盖的环境变量: MODEL MODEL_PATH MAX_NEW_TOKENS N CUDA_VISIBLE_DEVICES SKIP_GEN SKIP_EVAL
set -e
cd "$(dirname "$0")"
REPO_ROOT=$(cd .. && pwd)

# 生成需要 torch/transformers/peft（safegrpo 环境）；GPT 评分需要 openai（vllm_qwen3 环境）
GEN_PYTHON=${GEN_PYTHON:-/root/miniconda3/bin/python}
EVAL_PYTHON=${EVAL_PYTHON:-/root/miniconda3/bin/python}

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/huggingface/Qwen3-VL-4B-Thinking}

LORA_PATH=${1:-${LORA_PATH:-}}
if [ -n "${LORA_PATH}" ]; then
    case "${LORA_PATH}" in
        /*) ;;
        *) LORA_PATH="${REPO_ROOT}/${LORA_PATH}" ;;
    esac
    LORA_PATH=${LORA_PATH%/}
    if [ ! -f "${LORA_PATH}/adapter_config.json" ]; then
        echo "LoRA 目录不存在或缺少 adapter_config.json: ${LORA_PATH}"
        echo "可用的 LoRA 例如:"
        find "${REPO_ROOT}/train_ckpt" -name adapter_config.json -printf '  %h\n' 2>/dev/null | sed "s#${REPO_ROOT}/##" | sort
        exit 1
    fi
fi

# 结果目录名，mergejson.py / gpt-eval.py / gen_score.py 都按它找文件
# 默认: 基座 -> Qwen3-VL-4B-Thinking-base；LoRA -> Qwen3-VL-4B-Thinking-lora-<train_ckpt 下的相对路径>
if [ -z "${MODEL}" ]; then
    if [ -z "${LORA_PATH}" ]; then
        MODEL=Qwen3-VL-4B-Thinking-base
    else
        LORA_TAG=${LORA_PATH#${REPO_ROOT}/train_ckpt/}
        LORA_TAG=${LORA_TAG#/}
        MODEL=Qwen3-VL-4B-Thinking-lora-${LORA_TAG//\//-}
    fi
fi

# 模板总会先打开 <think>：LoRA 直接在里面输出 DPCoT 标签（训练时上限 500），512 足够；基座要先长篇思考，给更大的预算
if [ -z "${MAX_NEW_TOKENS}" ]; then
    if [ -z "${LORA_PATH}" ]; then MAX_NEW_TOKENS=4096; else MAX_NEW_TOKENS=512; fi
fi
N=${N:-3}

DATASET=${REPO_ROOT}/dataset/siuo_gen.json
IMAGE_DIR=${REPO_ROOT}/dataset/images
GEN_DIR=./gpt-call/${MODEL}/gen
EVAL_DIR=./gpt_eval/${MODEL}

# gpt-eval.py 把图片拼在 ./SIUO-dataset/images/ 下，数据实际在仓库根目录 dataset/images
if [ ! -e ./SIUO-dataset ]; then
    ln -s ../dataset ./SIUO-dataset
fi

echo "MODEL=${MODEL}"
echo "base=${MODEL_PATH}"
echo "LoRA=${LORA_PATH:-<none>}"
echo "max_new_tokens=${MAX_NEW_TOKENS} n=${N}"

if [ "${SKIP_GEN}" != "1" ]; then
    echo "执行 SIUO 数据集 generation 任务（日志: ${GEN_DIR}/log_file.log）"
    # 已生成的 output_*.json 会被跳过，中断后重跑同一命令即可续跑
    $GEN_PYTHON generation-Qwen3-VL-4B-Thinking-gen.py \
        --model-path "${MODEL_PATH}" \
        --lora-path "${LORA_PATH}" \
        --model-tag "${MODEL}" \
        --output-root ./gpt-call \
        --dataset "${DATASET}" \
        --image-dir "${IMAGE_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --temperature 0.8 \
        --top-p 0.95 \
        --top-k 50 \
        --n "${N}"
    echo "SIUO 数据集 generation 任务完成"
fi

TOTAL=$($EVAL_PYTHON -c "import json;print(len(json.load(open('${DATASET}'))))")
DONE=$(find "${GEN_DIR}" -name 'output_*.json' 2>/dev/null | wc -l)
echo "已生成 ${DONE}/${TOTAL} 条"
if [ -f "${GEN_DIR}/error_log.txt" ]; then
    echo "生成阶段有报错，见 ${GEN_DIR}/error_log.txt"
fi
if [ "${DONE}" -eq 0 ]; then
    echo "没有任何生成结果，终止"
    exit 1
fi

echo "正在 merge json"
$EVAL_PYTHON mergejson.py --model "${MODEL}"

if [ "${SKIP_EVAL}" = "1" ]; then
    echo "SKIP_EVAL=1，跳过 GPT 评分；合并结果: ./gpt-call/${MODEL}/siuo_gen-${MODEL}.json"
    exit 0
fi

mkdir -p "${EVAL_DIR}/safe" "${EVAL_DIR}/effective"

echo "开始评分 safe 项（日志: ${EVAL_DIR}/safe/run.log）"
$EVAL_PYTHON gpt-eval.py --model "${MODEL}" --mode safe > "${EVAL_DIR}/safe/run.log" 2>&1 &
SAFE_PID=$!

echo "开始评分 effective 项（日志: ${EVAL_DIR}/effective/run.log）"
$EVAL_PYTHON gpt-eval.py --model "${MODEL}" --mode effective > "${EVAL_DIR}/effective/run.log" 2>&1 &
EFFECTIVE_PID=$!

FAILED=0
wait ${SAFE_PID} || { echo "safe 项评分失败，请查看 ${EVAL_DIR}/safe/run.log"; FAILED=1; }
wait ${EFFECTIVE_PID} || { echo "effective 项评分失败，请查看 ${EVAL_DIR}/effective/run.log"; FAILED=1; }
[ ${FAILED} -eq 0 ] || exit 1
for mode in safe effective; do
    if [ -s "${EVAL_DIR}/${mode}/error_log.txt" ]; then
        echo "${mode} 项有 $(wc -l < "${EVAL_DIR}/${mode}/error_log.txt") 条调用失败，见 ${EVAL_DIR}/${mode}/error_log.txt"
    fi
done
echo "safe 项和 effective 项评分完成"

echo "开始汇总安全性与有用性得分"
$EVAL_PYTHON gen_score.py --model "${MODEL}"
echo "评测结果已保存到 ${EVAL_DIR}/eval_results.json"
