#!/usr/bin/env python3
"""Read-only planner for syncing a fork branch onto upstream."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan fork branch sync against upstream.")
    parser.add_argument("--repo", default=".", help="Repository path. Default: current directory.")
    parser.add_argument("--upstream", required=True, help="Upstream ref, for example upstream/main.")
    parser.add_argument("--branch", default="HEAD", help="Branch/ref to rebuild. Default: HEAD.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    branch_name = git(["rev-parse", "--abbrev-ref", args.branch], repo)
    head_sha = git(["rev-parse", args.branch], repo)
    upstream_sha = git(["rev-parse", args.upstream], repo)
    counts = git(["rev-list", "--left-right", "--count", f"{args.upstream}...{args.branch}"], repo)
    merge_base = git(["merge-base", args.upstream, args.branch], repo)
    cherry = git(["cherry", "-v", args.upstream, args.branch], repo)

    print(f"repo: {repo}")
    print(f"branch: {branch_name} ({head_sha})")
    print(f"upstream: {args.upstream} ({upstream_sha})")
    print(f"merge-base: {merge_base}")
    print(f"left-right-count upstream...branch: {counts}")
    print()
    print("git cherry -v classification:")
    print(cherry if cherry else "(no local commits)")
    print()
    print("Legend: '+' replay this local commit; '-' drop as patch-equivalent to upstream.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
