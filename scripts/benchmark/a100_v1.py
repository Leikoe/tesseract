#!/usr/bin/env python3
"""Reproducible streaming benchmark for the Tesseract v1 A100 gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import http.client
import json
import pathlib
import statistics
import subprocess
import time
import urllib.parse


PROMPTS = [
    "What is the capital of France? Answer briefly.",
    "Count from one to ten, one number per line.",
    "In two sentences, explain why the sky appears blue.",
    "Write a short Rust function that adds two i32 values.",
    "Name three practical uses for a priority queue.",
    "Summarize the difference between TCP and UDP in one paragraph.",
    "Give four tips for debugging a production service.",
    "What is 17 multiplied by 23? Show the result only.",
]


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> dict[str, object]:
    url = urllib.parse.urlsplit(base_url)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=300)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    started = time.perf_counter()
    connection.request(
        "POST",
        f"{url.path.rstrip('/')}/v1/chat/completions",
        body=body,
        headers={"content-type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        payload = response.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {payload}")

    token_event_times: list[float] = []
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        content = (choice.get("delta") or {}).get("content")
        if content:
            token_event_times.append(time.perf_counter())
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
    finished = time.perf_counter()
    connection.close()

    ttft = token_event_times[0] - started if token_event_times else finished - started
    intervals = [
        right - left for left, right in zip(token_event_times, token_event_times[1:])
    ]
    return {
        "prompt": prompt,
        "latency_seconds": finished - started,
        "ttft_seconds": ttft,
        "mean_inter_token_seconds": statistics.fmean(intervals) if intervals else 0.0,
        "token_events": len(token_event_times),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "finish_reason": finish_reason,
    }


def summarize(results: list[dict[str, object]], wall_seconds: float) -> dict[str, object]:
    latencies = [float(result["latency_seconds"]) for result in results]
    ttfts = [float(result["ttft_seconds"]) for result in results]
    tpots = [
        float(result["mean_inter_token_seconds"])
        for result in results
        if int(result["token_events"]) > 1
    ]
    completion_tokens = sum(int(result["completion_tokens"]) for result in results)

    def distribution(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values) if values else 0.0,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p99": percentile(values, 0.99),
        }

    return {
        "requests": len(results),
        "completion_tokens": completion_tokens,
        "wall_seconds": wall_seconds,
        "completion_tokens_per_second": completion_tokens / wall_seconds,
        "request_latency_seconds": distribution(latencies),
        "ttft_seconds": distribution(ttfts),
        "inter_token_seconds": distribution(tpots),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--source-revision",
        help="explicit source label for Git-less snapshots (defaults to git rev-parse HEAD)",
    )
    parser.add_argument("--server-args", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument(
        "--fixed-prompt",
        help="use one prompt for every warmup and measured request",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.requests <= 0 or args.max_tokens <= 0:
        parser.error("concurrency, requests, and max-tokens must be positive")

    prompts = [args.fixed_prompt] if args.fixed_prompt is not None else PROMPTS
    for index in range(args.warmup_requests):
        request_once(args.base_url, args.model, prompts[index % len(prompts)], 4, index)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                args.base_url,
                args.model,
                prompts[index % len(prompts)],
                args.max_tokens,
                index,
            )
            for index in range(args.requests)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started

    config_path = args.model_path / "config.json"
    report = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware": command(
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "cuda": command("nvcc", "--version").splitlines()[-1],
        "rust": command("rustc", "--version"),
        "git_revision": args.source_revision or command("git", "rev-parse", "HEAD"),
        "model": args.model,
        "model_revision": args.model_revision,
        "model_config_sha256": sha256(config_path),
        "server_args": args.server_args,
        "workload": {
            "concurrency": args.concurrency,
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "warmup_requests": args.warmup_requests,
            "prompts": prompts,
        },
        "summary": summarize(results, wall_seconds),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
