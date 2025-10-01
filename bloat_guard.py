#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repo-bloat-guard — single-file CI watchdog for repository size creep.

Compares the tracked files at a base ref (e.g., origin/main) vs the current ref (HEAD)
and evaluates absolute/relative growth against configurable thresholds.

Exit codes:
  0 = within thresholds
  1 = threshold exceeded
  2 = configuration/runtime error

Author: You (MIT)
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from typing import Dict, Tuple, List, Optional

# -------------------------------
# Utility: run a git command
# -------------------------------
def run(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

# -------------------------------
# Parse thresholds
# e.g. "+1.5MB or +12%"
# -------------------------------
_SIZE_RE = re.compile(r'^\s*\+?\s*([0-9]*\.?[0-9]+)\s*(B|KB|MB)?\s*$', re.IGNORECASE)
_PCT_RE  = re.compile(r'^\s*\+?\s*([0-9]*\.?[0-9]+)\s*%\s*$', re.IGNORECASE)

DEC_KB = 1000
DEC_MB = 1000 * 1000

def parse_size_to_bytes(s: str) -> int:
    m = _SIZE_RE.match(s)
    if not m:
        raise ValueError(f"Invalid size threshold: {s!r}")
    val = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    if unit == "B":
        mult = 1
    elif unit == "KB":
        mult = DEC_KB
    elif unit == "MB":
        mult = DEC_MB
    else:
        raise ValueError(f"Unsupported unit in size: {unit}")
    return int(val * mult)

def parse_fail_on(expr: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (abs_bytes, rel_percent).
    Accepts formats like:
      "+1MB or +10%"
      "+750KB or +0%"
      "+0 or +5%"
    Both parts are optional, but at least one must exist.
    """
    if not expr:
        return None, None
    parts = [p.strip() for p in re.split(r'\bor\b', expr, flags=re.IGNORECASE) if p.strip()]
    abs_bytes = None
    rel_pct = None
    for p in parts:
        if _PCT_RE.match(p):
            rel_pct = float(_PCT_RE.match(p).group(1))
        elif _SIZE_RE.match(p):
            abs_bytes = parse_size_to_bytes(p)
        else:
            raise ValueError(f"Could not parse threshold piece: {p!r}")
    if abs_bytes is None and rel_pct is None:
        raise ValueError(f"No thresholds parsed from: {expr!r}")
    return abs_bytes, rel_pct

# -------------------------------
# Parse one ls-tree line (robust)
# format: "<mode> <type> <object> <size>\t<path>"
# -------------------------------
def parse_ls_tree_entry(entry: str) -> Optional[Tuple[str, int]]:
    """
    Returns (path, size) or None if unparsable.
    Works with lines from: git ls-tree -r -z --long <ref>
    """
    if not entry:
        return None
    tab_idx = entry.find('\t')
    if tab_idx < 0:
        return None
    meta = entry[:tab_idx].strip()
    path = entry[tab_idx + 1 :].strip()
    # meta = "<mode> <type> <object> <size>"
    parts = meta.split()
    if len(parts) < 4:
        return None
    size_str = parts[-1]
    try:
        size = int(size_str)
    except Exception:
        return None
    if not path:
        return None
    return (path, size)

# -------------------------------
# Git size scan via ls-tree
# -------------------------------
def ls_tree_sizes(ref: str, include: List[str], exclude: List[str]) -> Dict[str, int]:
    """
    Returns {path: size_in_bytes} for tracked files at a given ref.
    Uses: git ls-tree -r -z --long <ref>
    """
    code, out, err = run(["git", "ls-tree", "-r", "-z", "--long", ref])
    if code != 0:
        raise RuntimeError(f"git ls-tree failed for {ref!r}: {err.strip() or out.strip()}")
    sizes: Dict[str, int] = {}
    for e in out.split('\x00'):
        if not e:
            continue
        parsed = parse_ls_tree_entry(e)
        if not parsed:
            continue
        path, size = parsed
        if include and not any(fnmatch.fnmatch(path, pat) for pat in include):
            continue
        if exclude and any(fnmatch.fnmatch(path, pat) for pat in exclude):
            continue
        sizes[path] = size
    return sizes

# -------------------------------
# Delta computation & reporting
# -------------------------------
def human_bytes(n: int) -> str:
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    if n_abs >= DEC_MB:
        return f"{sign}{n_abs/DEC_MB:.2f} MB"
    if n_abs >= DEC_KB:
        return f"{sign}{n_abs/DEC_KB:.2f} KB"
    return f"{sign}{n_abs} B"

def compute_totals(base: Dict[str, int], head: Dict[str, int]) -> Tuple[int, int, int]:
    base_total = sum(base.values())
    head_total = sum(head.values())
    delta = head_total - base_total
    return base_total, head_total, delta

def top_deltas(base: Dict[str, int], head: Dict[str, int], limit: int = 20) -> List[Tuple[str, int]]:
    # Positive size changes only; new or grown files
    deltas: List[Tuple[str, int]] = []
    all_paths = set(base.keys()) | set(head.keys())
    for p in all_paths:
        b = base.get(p, 0)
        h = head.get(p, 0)
        d = h - b
        if d > 0:
            deltas.append((p, d))
    deltas.sort(key=lambda x: x[1], reverse=True)
    return deltas[:limit]

# -------------------------------
# Threshold evaluation
# -------------------------------
def exceeds(delta: int, base_total: int, abs_bytes: Optional[int], rel_pct: Optional[float]) -> Tuple[bool, str]:
    messages = []
    hit = False
    if abs_bytes is not None:
        if delta > abs_bytes:
            hit = True
            messages.append(f"ABS threshold exceeded: {human_bytes(delta)} > {human_bytes(abs_bytes)}")
        else:
            messages.append(f"ABS threshold OK: {human_bytes(delta)} ≤ {human_bytes(abs_bytes)}")
    if rel_pct is not None:
        pct = (delta / base_total * 100.0) if base_total > 0 else (100.0 if delta > 0 else 0.0)
        if pct > rel_pct:
            hit = True
            messages.append(f"REL threshold exceeded: {pct:.2f}% > {rel_pct:.2f}%")
        else:
            messages.append(f"REL threshold OK: {pct:.2f}% ≤ {rel_pct:.2f}%")
    if not messages:
        messages.append("No thresholds configured; nothing to enforce.")
    return hit, " | ".join(messages)

# -------------------------------
# Argument parsing
# -------------------------------
def parse_glob_csv(val: Optional[str]) -> List[str]:
    if not val:
        return []
    out = []
    for part in val.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="repo-bloat-guard — single-file CI watchdog for repo size creep")
    p.add_argument("--ref-base", required=False, default=None,
                   help="Base ref/commit to compare against (e.g., origin/main, <sha>). If omitted, tries env GITHUB_BASE_REF, then 'origin/main'.")
    p.add_argument("--ref-head", required=False, default="HEAD", help="Head ref/commit (default: HEAD)")
    p.add_argument("--include", required=False, default=os.getenv("BG_INCLUDE", ""),
                   help="Comma-separated globs to include (default: include all).")
    p.add_argument("--exclude", required=False, default=os.getenv("BG_EXCLUDE", ""),
                   help="Comma-separated globs to exclude.")
    p.add_argument("--fail-on", required=False, default=None,
                   help='Threshold expression, e.g. "+1.5MB or +12%%". Overrides BG_FAIL_ABS/BG_FAIL_REL if provided.')
    p.add_argument("--report", action="store_true", help="Report only; never fail (exit 0).")
    p.add_argument("--json", action="store_true", help="Output JSON summary in addition to text.")
    return p

# -------------------------------
# Main
# -------------------------------
def main() -> int:
    try:
        parser = build_parser()
        args = parser.parse_args()

        ref_base = args.ref_base or os.getenv("GITHUB_BASE_REF") or "origin/main"
        ref_head = args.ref_head

        include = parse_glob_csv(args.include)
        exclude = parse_glob_csv(args.exclude)

        # Thresholds precedence:
        # 1) --fail-on
        # 2) BG_FAIL_ABS (bytes) / BG_FAIL_REL (percent)
        # 3) None (report only or manual check)
        abs_bytes: Optional[int] = None
        rel_pct: Optional[float] = None

        if args.fail_on:
            abs_bytes, rel_pct = parse_fail_on(args.fail_on)
        else:
            env_abs = os.getenv("BG_FAIL_ABS")
            env_rel = os.getenv("BG_FAIL_REL")
            if env_abs:
                try:
                    abs_bytes = int(env_abs)
                except Exception:
                    raise ValueError(f"BG_FAIL_ABS must be integer bytes, got {env_abs!r}")
            if env_rel:
                try:
                    rel_pct = float(env_rel)
                except Exception:
                    raise ValueError(f"BG_FAIL_REL must be a number (percent), got {env_rel!r}")

        # Ensure git available
        code, out, err = run(["git", "--version"])
        if code != 0:
            raise RuntimeError(f"git not available: {err.strip() or out.strip()}")

        # Ensure base ref exists (try to fetch if not)
        code, _, _ = run(["git", "rev-parse", "--verify", ref_base])
        if code != 0:
            if "/" in ref_base:
                run(["git", "fetch", "--all", "--tags", "--prune"])
                code2, _, err2 = run(["git", "rev-parse", "--verify", ref_base])
                if code2 != 0:
                    raise RuntimeError(f"Cannot resolve base ref {ref_base!r}: {err2.strip()}")
            else:
                raise RuntimeError(f"Cannot resolve base ref {ref_base!r}. Ensure history is available (fetch-depth: 0).")

        # Resolve head too
        code, _, err = run(["git", "rev-parse", "--verify", ref_head])
        if code != 0:
            raise RuntimeError(f"Cannot resolve head ref {ref_head!r}: {err.strip()}")

        base_sizes = ls_tree_sizes(ref_base, include, exclude)
        head_sizes = ls_tree_sizes(ref_head, include, exclude)

        base_total, head_total, delta = compute_totals(base_sizes, head_sizes)
        rel_delta_pct = (delta / base_total * 100.0) if base_total > 0 else (100.0 if delta > 0 else 0.0)

        # Compose report
        lines = []
        lines.append("repo-bloat-guard report")
        lines.append(f"  base: {ref_base}")
        lines.append(f"  head: {ref_head}")
        lines.append(f"  include: {include or ['<all>']}")
        lines.append(f"  exclude: {exclude or ['<none>']}")
        lines.append(f"  base total: {human_bytes(base_total)} ({base_total} B)")
        lines.append(f"  head total: {human_bytes(head_total)} ({head_total} B)")
        lines.append(f"  delta:      {human_bytes(delta)} ({delta} B)  [{rel_delta_pct:.2f}%]")

        top = top_deltas(base_sizes, head_sizes, limit=20)
        if top:
            lines.append("\nTop increases:")
            for p, d in top:
                lines.append(f"  + {human_bytes(d):>10}  {p}")
        else:
            lines.append("\nTop increases: (none)")

        # Evaluate thresholds
        hit = False
        thresh_msg = "No thresholds configured."
        if abs_bytes is not None or rel_pct is not None:
            hit, thresh_msg = exceeds(delta, base_total, abs_bytes, rel_pct)
        lines.append(f"\nThresholds: {thresh_msg}")

        # Print text
        print("\n".join(lines))

        # Optional JSON
        if args.json:
            payload = {
                "base_ref": ref_base,
                "head_ref": ref_head,
                "include": include,
                "exclude": exclude,
                "base_total_bytes": base_total,
                "head_total_bytes": head_total,
                "delta_bytes": delta,
                "delta_percent": rel_delta_pct,
                "threshold_abs_bytes": abs_bytes,
                "threshold_rel_percent": rel_pct,
                "exceeded": bool(hit),
                "top_increases": [{"path": p, "delta_bytes": d} for p, d in top],
            }
            print("\n===JSON===")
            print(json.dumps(payload, indent=2))

        if args.report:
            return 0
        return 1 if hit else 0

    except Exception as e:
        print(f"[repo-bloat-guard] ERROR: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
