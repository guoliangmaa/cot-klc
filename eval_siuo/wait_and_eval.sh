#!/bin/bash
# 等 stage1 训练进程退出后，在唯一一张 GPU 上评测本次 LoRA（SIUO）。
# 基座不跑：单卡串行太久，优先看训练后的模型。
#
# 用法（后台常驻，关掉终端也不受影响）:
#   setsid nohup bash eval_siuo/wait_and_eval.sh <训练PID> >> eval_siuo/logs/wait_and_eval.log 2>&1 &
# 可覆盖: LORA_PATH（默认 train_ckpt/stage1/final）POLL_SECONDS
cd "$(dirname "$0")"
REPO_ROOT=$(cd .. && pwd)

TRAIN_PID=${1:?用法: bash wait_and_eval.sh <训练PID>}
LORA_PATH=${LORA_PATH:-train_ckpt/stage1/final}
POLL_SECONDS=${POLL_SECONDS:-60}
LOG_DIR=./logs
mkdir -p "${LOG_DIR}"
START_TS=$(date +%s)

log() { echo "[$(date '+%F %T')] $*"; }

log "等待训练进程 ${TRAIN_PID} 结束: $(ps -o cmd= -p "${TRAIN_PID}" 2>/dev/null)"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
done
log "训练进程已退出，等待显存释放"
sleep 60

# 训练异常退出时不会写 final；此时退回本次训练最新的 step_* 存档。
# 旧的 final（mtime 早于本脚本启动）不算本次训练。
need_fallback=0
if [ ! -f "${REPO_ROOT}/${LORA_PATH}/adapter_config.json" ]; then
    need_fallback=1
else
    mtime=$(stat -c %Y "${REPO_ROOT}/${LORA_PATH}/adapter_model.safetensors" 2>/dev/null || echo 0)
    if [ "${mtime}" -lt "${START_TS}" ]; then
        need_fallback=1
    fi
fi
if [ "${need_fallback}" -eq 1 ]; then
    LATEST=$(ls -d "${REPO_ROOT}"/train_ckpt/stage1/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -z "${LATEST}" ]; then
        log "没有找到本次训练写出的 ${LORA_PATH}，也没有 step_* 存档，退出"
        exit 1
    fi
    log "没有找到本次训练写出的 ${LORA_PATH}，改用最新存档: ${LATEST#${REPO_ROOT}/}"
    LORA_PATH=${LATEST#${REPO_ROOT}/}
fi
log "LoRA=${LORA_PATH}（单卡，只评测 LoRA）"

# generation 默认跳过已有 output_*.json。同名目录是上一次评测，先挪走。
LORA_TAG=${LORA_PATH#train_ckpt/}
LORA_TAG=${LORA_TAG#/}
MODEL_TAG=Qwen3-VL-4B-Thinking-lora-${LORA_TAG//\//-}
for kind in gpt-call gpt_eval; do
    d="./${kind}/${MODEL_TAG}"
    if [ -d "${d}" ]; then
        dest="${d}.prev-${START_TS}"
        log "已有 ${d}，挪到 ${dest}"
        mv "${d}" "${dest}"
    fi
done

[ -e ./SIUO-dataset ] || ln -s ../dataset ./SIUO-dataset

log "启动 LoRA 评测（GPU 0，日志 ${LOG_DIR}/batch_lora.log）"
CUDA_VISIBLE_DEVICES=0 bash batch.sh "${LORA_PATH}" > "${LOG_DIR}/batch_lora.log" 2>&1
LORA_RC=$?
log "LoRA 评测结束，exit=${LORA_RC}"

f="./gpt_eval/${MODEL_TAG}/eval_results.json"
if [ -f "$f" ]; then
    log "$f"
    python3 -c "import json;d=json.load(open('$f'));print({k:d[k] for k in ('overall_avg_safetyness','overall_avg_helpfulness','overall_avg_combined')})"
fi
exit ${LORA_RC}
