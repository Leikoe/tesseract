#!/usr/bin/env python3
"""Emit pinned Transformers BF16 next-token logits for Tesseract validation."""

import argparse
import json
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tesseract-bin")
    # BF16 output logits are quantized in 0.0625/0.125-sized steps in this
    # range, and cuTile/cublas reduction order differs from PyTorch eager.
    parser.add_argument("--max-abs-diff", type=float, default=0.5)
    args = parser.parse_args()

    prompts = args.prompt or [
        "The capital of France is",
        "2 + 2 =",
        "Rust is a programming",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    model.eval()
    results = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to("cuda:0")
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
        values, indices = torch.topk(logits, args.top_k)
        result = {
            "prompt": prompt,
            "input_ids": input_ids[0].cpu().tolist(),
            "top_logits": [
                {"token_id": int(token), "logit": float(logit)}
                for token, logit in zip(indices.cpu(), values.cpu())
            ],
        }
        if args.tesseract_bin:
            completed = subprocess.run(
                [
                    args.tesseract_bin,
                    "--model-path",
                    args.model_path,
                    "--prompt",
                    prompt,
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            actual = json.loads(completed.stdout)
            if actual["prompt_tokens"] != len(result["input_ids"]):
                raise AssertionError("Tesseract and reference token counts differ")
            expected_by_id = {
                entry["token_id"]: entry["logit"] for entry in result["top_logits"]
            }
            if actual["next_token_id"] != result["top_logits"][0]["token_id"]:
                raise AssertionError("Tesseract and reference top-1 tokens differ")
            actual_ids = {entry["token_id"] for entry in actual["top_logits"]}
            reference_top_ten = {
                entry["token_id"] for entry in result["top_logits"][:10]
            }
            if not reference_top_ten.issubset(actual_ids):
                missing = sorted(reference_top_ten - actual_ids)
                raise AssertionError(
                    f"Tesseract top-k misses reference top-ten tokens {missing} for {prompt!r}"
                )
            common_ids = set(expected_by_id) & actual_ids
            max_abs_diff = max(
                abs(entry["logit"] - expected_by_id[entry["token_id"]])
                for entry in actual["top_logits"]
                if entry["token_id"] in common_ids
            )
            if max_abs_diff > args.max_abs_diff:
                raise AssertionError(
                    f"maximum logit difference {max_abs_diff} exceeds {args.max_abs_diff}"
                )
            result["tesseract_next_token_id"] = actual["next_token_id"]
            result["max_abs_logit_diff"] = max_abs_diff
            result["top_k_overlap"] = len(common_ids)
        results.append(result)

    report = {
        "implementation": "transformers-4.55.0/torch-2.8.0",
        "attention": "eager",
        "dtype": "bfloat16",
        "max_abs_diff_tolerance": args.max_abs_diff,
        "results": results,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
