"""6.4 reward 函数单元测试.

伪造 4 类 rollout 验证 5 分量奖励:
  case_perfect    - 全对, R 应接近 1.0
  case_no_format  - 缺 tag, R = 0 (format gate)
  case_wrong_cat  - category 错, R_tag/R_behavior=0
  case_partial    - cat 对但 path1 错, R_tag=0.75
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpcot import build_sft_target, compute_reward

# 用真实样本作为 ground truth: 找一条 path1 不是 none 的, 才能做 partial 测试
ROOT = Path(__file__).resolve().parent.parent
SAMPLE = None
for l in (ROOT / "data_build" / "05_siuo_trace_lite.jsonl").open():
    s = json.loads(l)
    if s["source"] == "siuo" and s["gt_path1_risk"] not in ("none", None):
        SAMPLE = s
        break
assert SAMPLE, "找不到 path1 非 none 的样本"
print(f"[ground truth] qid={SAMPLE['question_id']} cat={SAMPLE['gt_category']}")
print(f"  path1={SAMPLE['gt_path1_risk']}  path2={SAMPLE['gt_path2_intent']}  action={SAMPLE['gt_action']}")
print()


def show(name: str, text: str) -> None:
    rs = compute_reward(text, SAMPLE)
    print(f"=== {name} ===")
    print(f"  parsed: cat={rs['parsed'].get('category')} "
          f"path1={rs['parsed'].get('path1_risk')} action={rs['parsed'].get('action')} "
          f"missing={rs['parsed']['missing_tags']}")
    print(f"  R={rs['R']:.3f}  Rf={rs['R_format']:.2f}  Rt={rs['R_tag']:.2f}  "
          f"Rb={rs['R_behavior']:.2f}  Rl={rs['R_logic']:.2f}  Rk={rs['R_knowledge']:.2f}")
    print()
    return rs


# 1. perfect: 用 SFT target 当作模型输出, 应该满分
perfect = build_sft_target(SAMPLE)
r1 = show("PERFECT (= SFT target)", perfect)

# 2. no_format: 普通文本, 没有任何 tag
no_format = "I'm sorry, I cannot help with that request."
r2 = show("NO_FORMAT", no_format)

# 3. wrong_cat: 格式对, 但 category 错成 normal
wrong_cat = perfect.replace(
    f"<category>{SAMPLE['gt_category']}</category>",
    "<category>normal intention</category>",
).replace(
    f"<action>{SAMPLE['gt_action']}</action>",
    "<action>respond</action>",
).replace(
    "I'm sorry, I can't do that",
    "Sure, here is some encouragement",
)
r3 = show("WRONG_CATEGORY", wrong_cat)

# 4. partial: cat 对但 path1 错成 none
partial = perfect.replace(
    f"<path1_risk>{SAMPLE['gt_path1_risk']}</path1_risk>",
    "<path1_risk>none</path1_risk>",
)
# 让它 path1 错 -> 期待 R_tag=0.75 (cat对 0.5 + path1错0 + path2对0.25)
r4 = show("PARTIAL (path1 wrong)", partial)


# ---------- 断言 ----------
print("===== assertions =====")
checks = []

# 1. perfect: R_format=1, R_tag=1, R_behavior=1
checks.append(("perfect R_format", r1["R_format"] == 1.0))
checks.append(("perfect R_tag", r1["R_tag"] == 1.0))
checks.append(("perfect R_behavior", r1["R_behavior"] == 1.0))
checks.append(("perfect R >= 0.6", r1["R"] >= 0.6))

# 2. no_format: 全 0
checks.append(("no_format R_format=0", r2["R_format"] == 0.0))
checks.append(("no_format R=0", r2["R"] == 0.0))

# 3. wrong_cat: R_format=1, R_tag=0, R_behavior=0
checks.append(("wrong_cat R_format=1", r3["R_format"] == 1.0))
checks.append(("wrong_cat R_tag=0", r3["R_tag"] == 0.0))
checks.append(("wrong_cat R_behavior=0", r3["R_behavior"] == 0.0))

# 4. partial: R_tag=0.75 (path1 错)
checks.append(("partial R_tag = 0.75", abs(r4["R_tag"] - 0.75) < 1e-6))
checks.append(("partial R_behavior=1", r4["R_behavior"] == 1.0))

ok = all(v for _, v in checks)
for name, v in checks:
    print(f"  {'✓' if v else '✗'} {name}")
print(f"\n{'PASSED' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
