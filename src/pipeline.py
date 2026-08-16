"""One-command entry point: embed -> validate -> report -> sensitivity.

    python src/pipeline.py --tag myrun

Stops at the validation gate on failure. That gate exists to prevent
publishing a number the embedding cannot actually support, so it is not
skippable from here by design -- rerun the individual steps if you want to
inspect a failing run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(script: str, *args: str) -> int:
    cmd = [PYTHON, str(SRC / script), *args]
    print(f"\n{'=' * 62}\n$ {' '.join(cmd[1:])}\n{'=' * 62}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full EpiScore pipeline.")
    parser.add_argument("--tag", default="run1")
    parser.add_argument("--source", choices=("subset", "gated"), default="subset")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--factories", type=int, default=10)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--blur-percentile", type=float, default=20.0)
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="reuse existing embeddings for this tag",
    )
    args = parser.parse_args()

    if not args.skip_embed:
        embed_args = [
            "--tag", args.tag,
            "--source", args.source,
            "--workers", str(args.workers),
            "--factories", str(args.factories),
        ]
        if args.max_clips:
            embed_args += ["--max-clips", str(args.max_clips)]
        if run("run_embed.py", *embed_args) != 0:
            print("\nembedding failed", file=sys.stderr)
            return 1

    gate = run("validate.py", "--tag", args.tag)
    if gate == 2:
        print(
            "\nVALIDATION FAILED — the embedding does not discriminate on this "
            "footage. Any redundancy figure would be measuring that failure "
            "rather than the corpus. See README for remedies.",
            file=sys.stderr,
        )
        return 2
    if gate != 0:
        return gate

    if run("run_report.py", "--tag", args.tag,
           "--blur-percentile", str(args.blur_percentile)) != 0:
        return 1

    run("sensitivity.py", "--tag", args.tag)
    print("\ndone. report + curves in out/, example pairs via extract_pairs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
