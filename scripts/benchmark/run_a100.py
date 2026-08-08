#!/usr/bin/env python3
"""Build, run, benchmark, and stop a production Tesseract A100 server."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"


def command(*args: str, cwd: pathlib.Path) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run the reproducible Tesseract serving suite on an A100"
    )
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get(
                "TESSERACT_MODEL_PATH",
                "/home/ubuntu/models/Llama-3.2-1B-Instruct",
            )
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("TESSERACT_MODEL_REVISION", MODEL_REVISION),
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--listen", default="127.0.0.1:8000")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("target/benchmarks") / timestamp,
    )
    parser.add_argument("--batch1-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--concurrent-requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--ready-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="extra server argument; repeat and use --server-arg=--flag=value",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record results even when tracked files differ from HEAD",
    )
    args = parser.parse_args()
    positive = {
        "batch1-requests": args.batch1_requests,
        "concurrency": args.concurrency,
        "concurrent-requests": args.concurrent_requests,
        "max-tokens": args.max_tokens,
        "ready-timeout-seconds": args.ready_timeout_seconds,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.warmup_requests < 0:
        parser.error("warmup-requests must be non-negative")
    return args


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def base_url(listen: str) -> str:
    host, separator, port = listen.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError("--listen must be HOST:PORT")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def http_get(url: str, timeout: float = 2.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return response.read()


def wait_until_ready(
    url: str,
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
    server_log: pathlib.Path,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"server exited with status {exit_code} before readiness\n{tail}"
            )
        try:
            http_get(f"{url}/health/ready")
            return
        except (OSError, RuntimeError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(f"server did not become ready within {timeout_seconds:g}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_workload(
    *,
    root: pathlib.Path,
    environment: dict[str, str],
    args: argparse.Namespace,
    url: str,
    server_args: str,
    concurrency: int,
    requests: int,
    output: pathlib.Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/benchmark/a100_v1.py"),
            "--base-url",
            url,
            "--model",
            args.model,
            "--model-path",
            str(args.model_path),
            "--model-revision",
            args.model_revision,
            "--server-args",
            server_args,
            "--concurrency",
            str(concurrency),
            "--requests",
            str(requests),
            "--max-tokens",
            str(args.max_tokens),
            "--warmup-requests",
            str(args.warmup_requests),
            "--output",
            str(output),
        ],
        cwd=root,
        env=environment,
        check=True,
    )


def milliseconds(value: float) -> str:
    return f"{value * 1000.0:.2f} ms"


def write_summary(output_dir: pathlib.Path) -> None:
    workloads = [
        ("Batch 1", output_dir / "batch1.json"),
        ("Concurrency, first shape pass", output_dir / "concurrent-first.json"),
        ("Concurrency, warm shapes", output_dir / "concurrent-warm.json"),
    ]
    reports = [(name, json.loads(path.read_text())) for name, path in workloads]
    first = reports[0][1]
    lines = [
        "# Tesseract A100 serving benchmark",
        "",
        "These are measured results, not performance targets.",
        "",
        "## Environment",
        "",
        f"- Git: `{first['git_revision']}`",
        f"- GPU: {first['hardware']}",
        f"- CUDA: {first['cuda']}",
        f"- Rust: {first['rust']}",
        f"- Model: `{first['model']}`",
        f"- Model revision: `{first['model_revision']}`",
        f"- Server arguments: `{first['server_args']}`",
        "",
        "## Results",
        "",
        "| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in reports:
        summary = report["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{summary['completion_tokens_per_second']:.2f}",
                    milliseconds(summary["ttft_seconds"]["mean"]),
                    milliseconds(summary["ttft_seconds"]["p99"]),
                    milliseconds(summary["inter_token_seconds"]["mean"]),
                    milliseconds(summary["request_latency_seconds"]["p99"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The first concurrent pass includes lazy capture of exact batch/context",
            "shapes. The identical warm pass reuses those graphs.",
            "",
            "Raw request results, final Prometheus metrics, the server log, and the",
            "suite manifest are retained beside this file.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = repo_root()
    os.chdir(root)
    tracked_changes = command(
        "git", "status", "--porcelain", "--untracked-files=no", cwd=root
    )
    if tracked_changes and not args.allow_dirty:
        print(
            "benchmark refused: tracked files differ from HEAD "
            "(use --allow-dirty to override)",
            file=sys.stderr,
        )
        return 2
    if not (args.model_path / "config.json").is_file():
        print(f"benchmark refused: model not found at {args.model_path}", file=sys.stderr)
        return 2

    output_dir = args.output if args.output.is_absolute() else root / args.output
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"benchmark refused: output already exists: {output_dir}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    cuda_path = environment.get("TESSERACT_CUDA_PATH", "/usr/local/cuda-13.3")
    cargo_bin = str(pathlib.Path.home() / ".cargo/bin")
    environment["PATH"] = (
        f"{cuda_path}/bin:{cargo_bin}:{environment.get('PATH', '')}"
    )
    toolchain = environment.get("TESSERACT_RUST_TOOLCHAIN", "1.89.0")
    cargo = str(pathlib.Path(cargo_bin) / "cargo")
    if not pathlib.Path(cargo).is_file():
        cargo = "cargo"

    server_cli = ["--listen", args.listen, "--model", args.model]
    server_cli.extend(args.server_arg)
    server_args = shlex.join(server_cli)
    revision = command("git", "rev-parse", "HEAD", cwd=root)
    manifest_path = output_dir / "run.json"
    manifest = {
        "schema_version": 1,
        "status": "building",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_revision": revision,
        "git_dirty": bool(tracked_changes),
        "model": args.model,
        "model_revision": args.model_revision,
        "model_path": str(args.model_path),
        "server_args": server_args,
        "suite": {
            "batch1_requests": args.batch1_requests,
            "concurrency": args.concurrency,
            "concurrent_requests": args.concurrent_requests,
            "max_tokens": args.max_tokens,
            "warmup_requests": args.warmup_requests,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"benchmark revision: {revision}")
    print(f"benchmark output: {output_dir}")
    subprocess.run(
        [
            cargo,
            f"+{toolchain}",
            "build",
            "--release",
            "--features",
            "cuda",
            "--bin",
            "tesseract",
        ],
        cwd=root,
        env=environment,
        check=True,
    )

    server_log_path = output_dir / "server.log"
    server_process: subprocess.Popen[bytes] | None = None
    try:
        with server_log_path.open("wb") as server_log:
            server_process = subprocess.Popen(
                [str(root / "target/release/tesseract"), *server_cli],
                cwd=root,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
            manifest["status"] = "warming"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            url = base_url(args.listen)
            wait_until_ready(
                url,
                server_process,
                args.ready_timeout_seconds,
                server_log_path,
            )
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print("server ready; running batch-1 workload")
            run_workload(
                root=root,
                environment=environment,
                args=args,
                url=url,
                server_args=server_args,
                concurrency=1,
                requests=args.batch1_requests,
                output=output_dir / "batch1.json",
            )
            print("running concurrent first-shape workload")
            run_workload(
                root=root,
                environment=environment,
                args=args,
                url=url,
                server_args=server_args,
                concurrency=args.concurrency,
                requests=args.concurrent_requests,
                output=output_dir / "concurrent-first.json",
            )
            print("running identical warm-shape workload")
            run_workload(
                root=root,
                environment=environment,
                args=args,
                url=url,
                server_args=server_args,
                concurrency=args.concurrency,
                requests=args.concurrent_requests,
                output=output_dir / "concurrent-warm.json",
            )
            (output_dir / "metrics.prom").write_bytes(http_get(f"{url}/metrics"))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    finally:
        if server_process is not None:
            stop_server(server_process)

    manifest["status"] = "complete"
    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_summary(output_dir)
    print(f"benchmark complete: {output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
