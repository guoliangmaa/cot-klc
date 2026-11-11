import os
import json
from pathlib import Path


def merge_json_files(
        input_dir: str,
        output_file: str,
        encoding: str = "utf-8"
):
    """
    读取文件夹中的所有 .json 文件，将内容拼接为一个 list 后保存。

    规则：
    - 如果 json 文件内容是 list，则 extend
    - 如果 json 文件内容是 dict，则 append
    """

    input_dir = Path(input_dir)
    merged_data = []

    json_paths = sorted(input_dir.glob("output_*.json"))
    for json_path in json_paths:
        with open(json_path, "r", encoding=encoding) as f:
            data = json.load(f)

        if isinstance(data, list):
            merged_data.extend(data)
        elif isinstance(data, dict):
            merged_data.append(data)
        else:
            raise ValueError(f"Unsupported JSON type in {json_path}: {type(data)}")

    with open(output_file, "w", encoding=encoding) as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(json_paths)} files "
          f"-> {len(merged_data)} items saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Intern-S2-Preview")
    args = parser.parse_args()

    # input_dir="./gpt-call/gpt4o"
    # output_file="./test_results/siuo_gen-gpt4o.json"

    input_dir = f"./gpt-call/{args.model}/gen"
    output_file = f"./gpt-call/{args.model}/siuo_gen-{args.model}.json"
    merge_json_files(
        input_dir,
        output_file
    )