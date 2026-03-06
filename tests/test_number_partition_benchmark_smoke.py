from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_number_partition_benchmark_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "python" / "number_partition_benchmark.py"

    assert script.exists(), f"Missing benchmark script: {script}"

    out_base = Path(tempfile.gettempdir()) / "qanneal_np_bench_smoke"
    out_json = out_base.with_suffix(".json")

    if out_json.exists():
        out_json.unlink()

    cmd = [
        sys.executable,
        str(script),
        "--n",
        "14",
        "--reads",
        "2",
        "--steps",
        "8",
        "--sweeps",
        "4",
        "--worldline",
        "1",
        "--slices",
        "6",
        "--replicas",
        "3",
        "--cluster",
        "1",
        "--schedule-mode",
        "balanced",
        "--with-sqapt",
        "--pt-steps",
        "6",
        "--swap-interval",
        "1",
        "--jobs",
        "1",
        "--out",
        str(out_base),
    ]

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "Benchmark script failed\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    assert out_json.exists(), f"Missing output JSON: {out_json}"
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert "results" in payload and isinstance(payload["results"], list)
    method_names = {row["name"] for row in payload["results"]}

    assert "qanneal-SA" in method_names
    assert "qanneal-SQA" in method_names
    assert "qanneal-SQAPT" in method_names
