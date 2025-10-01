#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repo-bloat-guard — single-file CI watchdog for repository size creep.

Features:
- Compares tracked file sizes at a base ref (e.g., origin/main) vs HEAD
- Absolute/relative thresholds (flags/env or --fail-on "+1.5MB or +12%")
- Per-folder budgets via .bloatbudgets.json (max_total_bytes/max_delta_bytes)
- Suspicious file hints (media/zips/binaries/CAD/graphics)
- Biggest files at HEAD (top-N)
- "What changed" section (new vs. deleted files)
- Git LFS-aware mode (resolve true HEAD sizes where possible)
- Largest files over time: compare against a baseline JSON and write current snapshot
- Plain text, Markdown, and JSON outputs
- Zero external Python deps

Exit codes:
  0 = within thresholds
  1 = threshold exceeded (global thresholds or budgets)
  2 = configuration/runtime error

MIT © You
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
# Constants / suspicious types
# -------------------------------
DEC_KB = 1000
DEC_MB = 1000 * 1000

SUSPICIOUS_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi",
    ".zip", ".7z", ".rar", ".gz", ".tar",
    ".psd", ".ai", ".tiff", ".tif",
    ".pdf",
    ".step", ".stp", ".iges", ".igs",
    ".iso",
    ".exe", ".dll", ".dmg", ".apk",
    ".pkg",
}

# -------------------------------
# Process helpers
# -------------------------------
def run(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

# -------------------------------
# Threshold parsing
# -------------------------------
_SIZE_RE = re.compile(r'^\s*\+?\s*([0-9]*\.?[0-9]+)\s*(B|KB|MB)?\s*$', re.IGNORECASE)
_PCT_RE  = re.compile(r'^\s*\+?\s*([0-9]*\.?[0-9]+)\s*%\s*$', re.IGNORECASE)

def parse_size_to_bytes(s: str) -> int:
    m = _SIZE_RE.match(s)
    if not m:
        raise ValueError(f"Invalid size threshold: {s!r}")
    val = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    if unit == "B": mult = 1
    elif unit == "KB": mult = DEC_KB
    elif unit == "MB": mult = DEC_MB
    else: raise ValueError(f"Unsupported unit in size: {unit}")
    return int(val * mult)

def parse_fail_on(expr: Optional[str]) -> Tuple[Optional[int], Optional[float]]:
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
# git ls-tree parsing (robust)
# format: "<mode> <type> <object> <size>\t<path>"
# -------------------------------
def parse_ls_tree_entry(entry: str) -> Optional[Tuple[str, int]]:
    if not entry:
        return None
    tab_idx = entry.find('\t')
    if tab_idx < 0:
        return None
    meta = entry[:tab_idx].strip()
    path = entry[tab_idx + 1:].strip()
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

def ls_tree_sizes(ref: str, include: List[str], exclude: List[str]) -> Dict[str, int]:
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
# Totals / deltas
# -------------------------------
def human_bytes(n: int) -> str:
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    if n_abs >= DEC_MB: return f"{sign}{n_abs/DEC_MB:.2f} MB"
    if n_abs >= DEC_KB: return f"{sign}{n_abs/DEC_KB:.2f} KB"
    return f"{sign}{n_abs} B"

def compute_totals(base: Dict[str, int], head: Dict[str, int]) -> Tuple[int, int, int]:
    bt = sum(base.values())
    ht = sum(head.values())
    return bt, ht, ht - bt

def per_path_deltas(base: Dict[str, int], head: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    all_paths = set(base) | set(head)
    for p in all_paths:
        out[p] = head.get(p, 0) - base.get(p, 0)
    return out

def top_positive_deltas(deltas: Dict[str, int], limit: int = 20) -> List[Tuple[str, int]]:
    inc = [(p, d) for p, d in deltas.items() if d > 0]
    inc.sort(key=lambda x: x[1], reverse=True)
    return inc[:limit]

def biggest_files(sizes: Dict[str, int], limit: int = 20) -> List[Tuple[str, int]]:
    items = list(sizes.items())
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:limit]

def new_and_deleted(base: Dict[str, int], head: Dict[str, int]) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]:
    new_files = [(p, head[p]) for p in head.keys() - base.keys()]
    deleted_files = [(p, base[p]) for p in base.keys() - head.keys()]
    new_files.sort(key=lambda x: x[1], reverse=True)
    deleted_files.sort(key=lambda x: x[1], reverse=True)
    return new_files, deleted_files

# -------------------------------
# Threshold evaluation
# -------------------------------
def exceeds_global(delta: int, base_total: int, abs_bytes: Optional[int], rel_pct: Optional[float]) -> Tuple[bool, str]:
    msgs = []
    hit = False
    if abs_bytes is not None:
        if delta > abs_bytes: hit = True; msgs.append(f"ABS exceeded: {human_bytes(delta)} > {human_bytes(abs_bytes)}")
        else: msgs.append(f"ABS OK: {human_bytes(delta)} ≤ {human_bytes(abs_bytes)}")
    if rel_pct is not None:
        pct = (delta / base_total * 100.0) if base_total > 0 else (100.0 if delta > 0 else 0.0)
        if pct > rel_pct: hit = True; msgs.append(f"REL exceeded: {pct:.2f}% > {rel_pct:.2f}%")
        else: msgs.append(f"REL OK: {pct:.2f}% ≤ {rel_pct:.2f}%")
    if not msgs:
        msgs.append("No global thresholds configured.")
    return hit, " | ".join(msgs)

# -------------------------------
# Budgets
# -------------------------------
class Budget:
    def __init__(self, pattern: str, max_total_bytes: Optional[int], max_delta_bytes: Optional[int]):
        self.pattern = pattern
        self.max_total_bytes = max_total_bytes
        self.max_delta_bytes = max_delta_bytes

def load_budgets(path: Optional[str]) -> List[Budget]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    budgets: List[Budget] = []
    for item in raw:
        pat = item.get("pattern") or ""
        if not pat: continue
        mt = item.get("max_total_bytes")
        md = item.get("max_delta_bytes")
        budgets.append(Budget(pat, mt if isinstance(mt, int) else None, md if isinstance(md, int) else None))
    return budgets

def check_budgets(budgets: List[Budget], head_sizes: Dict[str, int], deltas: Dict[str, int]) -> Tuple[bool, List[str]]:
    hit = False
    msgs: List[str] = []
    for b in budgets:
        total = 0
        delta_pos = 0
        for p, size in head_sizes.items():
            if fnmatch.fnmatch(p, b.pattern):
                total += size
                d = deltas.get(p, 0)
                if d > 0:
                    delta_pos += d
        if b.max_total_bytes is not None and total > b.max_total_bytes:
            hit = True
            msgs.append(f"[Budget] pattern={b.pattern!r} total {human_bytes(total)} > max_total {human_bytes(b.max_total_bytes)}")
        if b.max_delta_bytes is not None and delta_pos > b.max_delta_bytes:
            hit = True
            msgs.append(f"[Budget] pattern={b.pattern!r} delta +{human_bytes(delta_pos)} > max_delta +{human_bytes(b.max_delta_bytes)}")
        if b.max_total_bytes is None and b.max_delta_bytes is None:
            msgs.append(f"[Budget] pattern={b.pattern!r} has no limits (ignored).")
    return hit, msgs

# -------------------------------
# Suspicious hints
# -------------------------------
def suspicious_increases(deltas: Dict[str, int], exts: set, limit: int = 15) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for p, d in deltas.items():
        if d <= 0: continue
        _, ext = os.path.splitext(p.lower())
        if ext in exts:
            out.append((p, d))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:limit]

# -------------------------------
# Git LFS true-size resolution (HEAD only)
# -------------------------------
def lfs_resolve_head_sizes(head_sizes: Dict[str, int]) -> Tuple[Dict[str, int], str]:
    """
    Attempts to resolve true sizes for HEAD LFS files.
    Returns (possibly adjusted head_sizes, note_message).
    If git-lfs not available or parsing fails, returns original sizes with a note.
    """
    code, _, _ = run(["git", "lfs", "version"])
    if code != 0:
        return head_sizes, "LFS: git-lfs not available; using pointer sizes."

    code, out, err = run(["git", "lfs", "ls-files", "-l"])  # long format
    if code != 0:
        return head_sizes, f"LFS: ls-files failed; using pointer sizes. ({err.strip()})"

    sizes_by_path: Dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # Parse "... size <BYTES> <path>"
        m = re.search(r'\bsize\s+([0-9]+)\b', line)
        if not m:
            continue
        try:
            size = int(m.group(1))
        except Exception:
            continue
        parts = re.split(r'\bsize\s+[0-9]+\b\s+', line, maxsplit=1)
        if len(parts) != 2:
            continue
        path = parts[1].strip()
        if path:
            sizes_by_path[path] = size

    if not sizes_by_path:
        return head_sizes, "LFS: could not parse sizes; using pointer sizes."

    adjusted = dict(head_sizes)
    for p in list(adjusted.keys()):
        if p in sizes_by_path:
            adjusted[p] = sizes_by_path[p]
    return adjusted, "LFS: resolved true sizes for HEAD where available."

# -------------------------------
# Baseline history (largest files over time)
# -------------------------------
def to_biglist_dict(items: List[Tuple[str,int]]) -> Dict[str, int]:
    return {p: s for (p, s) in items}

def from_biglist_dict(d: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)

def load_baseline(path: Optional[str]) -> Optional[Dict[str,int]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_snapshot(path: Optional[str], head_big: List[Tuple[str,int]]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_biglist_dict(head_big), f, indent=2)

def compare_biglists(prev: Dict[str,int], curr: Dict[str,int], limit:int=20) -> List[Tuple[str,int,int]]:
    """
    Returns list of (path, prev_size, curr_size) sorted by curr_size desc.
    """
    keys = set(prev.keys()) | set(curr.keys())
    rows = []
    for k in keys:
        rows.append((k, prev.get(k, 0), curr.get(k, 0)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:limit]

# -------------------------------
# CLI
# -------------------------------
def parse_glob_csv(val: Optional[str]) -> List[str]:
    if not val: return []
    return [p.strip() for p in val.split(",") if p.strip()]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="repo-bloat-guard — single-file CI watchdog for repo size creep")
    p.add_argument("--ref-base", default=None, help="Base ref/commit to compare (tries env GITHUB_BASE_REF, then 'origin/main').")
    p.add_argument("--ref-head", default="HEAD", help="Head ref/commit (default: HEAD)")
    p.add_argument("--include", default=os.getenv("BG_INCLUDE", ""), help="Comma-separated globs to include (default: all).")
    p.add_argument("--exclude", default=os.getenv("BG_EXCLUDE", ""), help="Comma-separated globs to exclude.")
    p.add_argument("--fail-on", default=None, help='Threshold expression, e.g. "+1.5MB or +12%%". Overrides BG_FAIL_ABS/BG_FAIL_REL.')
    p.add_argument("--report", action="store_true", help="Report only; never fail (exit 0).")
    p.add_argument("--json", action="store_true", help="Also print JSON payload.")
    p.add_argument("--markdown", action="store_true", help="Print Markdown summary instead of plain text.")
    p.add_argument("--budgets-file", default=os.getenv("BG_BUDGETS_FILE", ".bloatbudgets.json"),
                   help="Path to budgets json (default .bloatbudgets.json if present).")
    p.add_argument("--lfs", action="store_true", help="Resolve true HEAD sizes via Git LFS if available.")
    p.add_argument("--top-head", type=int, default=20, help="How many biggest HEAD files to list (default 20).")
    p.add_argument("--top-increases", type=int, default=20, help="How many top increases to list (default 20).")
    p.add_argument("--top-new", type=int, default=20, help="How many new files to list (default 20).")
    p.add_argument("--top-deleted", type=int, default=20, help="How many deleted files to list (default 20).")
    p.add_argument("--baseline-json", default=os.getenv("BG_BASELINE_JSON", ""),
                   help="Path to a baseline JSON (previous 'biggest files at HEAD') for historical comparison.")
    p.add_argument("--snapshot-json", default=os.getenv("BG_SNAPSHOT_JSON", ""),
                   help="Path to write current 'biggest files at HEAD' snapshot JSON.")
    return p

# -------------------------------
# Markdown report
# -------------------------------
def render_markdown(ref_base: str,
                    ref_head: str,
                    include: List[str],
                    exclude: List[str],
                    base_total: int,
                    head_total: int,
                    delta: int,
                    rel_delta_pct: float,
                    top_inc: List[Tuple[str,int]],
                    global_msg: str,
                    budgets_msgs: List[str],
                    sus: List[Tuple[str,int]],
                    head_big: List[Tuple[str,int]],
                    new_files: List[Tuple[str,int]],
                    del_files: List[Tuple[str,int]],
                    lfs_note: Optional[str],
                    history_rows: List[Tuple[str,int,int]],
                    baseline_note: Optional[str]) -> str:
    lines: List[str] = []
    lines.append(f"## 📦 Repo Bloat Guard")
    lines.append("")
    lines.append(f"**Base:** `{ref_base}` &nbsp;&nbsp; **Head:** `{ref_head}`")
    if lfs_note:
        lines.append(f"<sub>{lfs_note}</sub>")
    if baseline_note:
        lines.append(f"<sub>{baseline_note}</sub>")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Base total | {human_bytes(base_total)} ({base_total} B) |")
    lines.append(f"| Head total | {human_bytes(head_total)} ({head_total} B) |")
    lines.append(f"| Delta | {human_bytes(delta)} ({delta} B) [{rel_delta_pct:.2f}%] |")
    lines.append("")
    lines.append(f"> **Global thresholds:** {global_msg}")
    if budgets_msgs:
        lines.append("")
        lines.append("**Budgets:**")
        for m in budgets_msgs:
            lines.append(f"- {m}")
    if top_inc:
        lines.append("")
        lines.append("**Top increases**")
        lines.append("")
        lines.append("| File | Δ Size |")
        lines.append("|---|---:|")
        for p, d in top_inc:
            lines.append(f"| `{p}` | +{human_bytes(d)} |")
    if sus:
        lines.append("")
        lines.append("**Suspicious increases** (by extension)")
        lines.append("")
        lines.append("| File | Δ Size |")
        lines.append("|---|---:|")
        for p, d in sus:
            lines.append(f"| `{p}` | +{human_bytes(d)} |")
    if head_big:
        lines.append("")
        lines.append("**Biggest files at HEAD**")
        lines.append("")
        lines.append("| File | Size |")
        lines.append("|---|---:|")
        for p, s in head_big:
            lines.append(f"| `{p}` | {human_bytes(s)} |")
    # What changed
    if new_files or del_files:
        lines.append("")
        lines.append("**What changed**")
        lines.append("")
        if new_files:
            lines.append("<details><summary><b>New files</b></summary>")
            lines.append("")
            lines.append("| File | Size |")
            lines.append("|---|---:|")
            for p, s in new_files:
                lines.append(f"| `{p}` | {human_bytes(s)} |")
            lines.append("</details>")
        if del_files:
            lines.append("")
            lines.append("<details><summary><b>Deleted files</b></summary>")
            lines.append("")
            lines.append("| File | Size (at base) |")
            lines.append("|---|---:|")
            for p, s in del_files:
                lines.append(f"| `{p}` | {human_bytes(s)} |")
            lines.append("</details>")
    # History
    if history_rows:
        lines.append("")
        lines.append("**Largest files over time** (vs baseline)")
        lines.append("")
        lines.append("| File | Prev | Now | Δ |")
        lines.append("|---|---:|---:|---:|")
        for p, prev_sz, now_sz in history_rows:
            diff = now_sz - prev_sz
            sign = "+" if diff >= 0 else ""
            lines.append(f"| `{p}` | {human_bytes(prev_sz)} | {human_bytes(now_sz)} | {sign}{human_bytes(diff)} |")
    if include or exclude:
        lines.append("")
        lines.append("<sub>")
        lines.append(f"include: `{include or ['<all>']}` &nbsp;&nbsp; exclude: `{exclude or ['<none>']}`")
        lines.append("</sub>")
    return "\n".join(lines)

# -------------------------------
# Main
# -------------------------------
def main() -> int:
    try:
        args = build_parser().parse_args()

        ref_base = args.ref_base or os.getenv("GITHUB_BASE_REF") or "origin/main"
        ref_head = args.ref_head

        include = parse_glob_csv(args.include)
        exclude = parse_glob_csv(args.exclude)

        # thresholds
        abs_bytes: Optional[int] = None
        rel_pct: Optional[float] = None
        if args.fail_on:
            abs_bytes, rel_pct = parse_fail_on(args.fail_on)
        else:
            env_abs = os.getenv("BG_FAIL_ABS")
            env_rel = os.getenv("BG_FAIL_REL")
            if env_abs:
                try: abs_bytes = int(env_abs)
                except Exception: raise ValueError(f"BG_FAIL_ABS must be integer bytes, got {env_abs!r}")
            if env_rel:
                try: rel_pct = float(env_rel)
                except Exception: raise ValueError(f"BG_FAIL_REL must be a number (percent), got {env_rel!r}")

        # git ok?
        code, out, err = run(["git", "--version"])
        if code != 0:
            raise RuntimeError(f"git not available: {err.strip() or out.strip()}")

        # resolve refs
        code, _, _ = run(["git", "rev-parse", "--verify", ref_base])
        if code != 0:
            if "/" in ref_base:
                run(["git", "fetch", "--all", "--tags", "--prune"])
                code2, _, err2 = run(["git", "rev-parse", "--verify", ref_base])
                if code2 != 0:
                    raise RuntimeError(f"Cannot resolve base ref {ref_base!r}: {err2.strip()}")
            else:
                raise RuntimeError(f"Cannot resolve base ref {ref_base!r}. Ensure checkout uses fetch-depth: 0.")
        code, _, err = run(["git", "rev-parse", "--verify", ref_head])
        if code != 0:
            raise RuntimeError(f"Cannot resolve head ref {ref_head!r}: {err.strip()}")

        # sizes
        base_sizes = ls_tree_sizes(ref_base, include, exclude)
        head_sizes = ls_tree_sizes(ref_head, include, exclude)

        # Optional: LFS adjust true sizes for HEAD (only)
        lfs_note = None
        if args.lfs and ref_head == "HEAD":
            head_sizes, lfs_note = lfs_resolve_head_sizes(head_sizes)

        deltas = per_path_deltas(base_sizes, head_sizes)
        base_total, head_total, delta = compute_totals(base_sizes, head_sizes)
        rel_delta_pct = (delta / base_total * 100.0) if base_total > 0 else (100.0 if delta > 0 else 0.0)

        # global thresholds
        global_hit, global_msg = False, "No global thresholds configured."
        if abs_bytes is not None or rel_pct is not None:
            global_hit, global_msg = exceeds_global(delta, base_total, abs_bytes, rel_pct)

        # budgets
        budgets = load_budgets(args.budgets_file)
        budget_hit, budget_msgs = False, []
        if budgets:
            budget_hit, budget_msgs = check_budgets(budgets, head_sizes, deltas)

        # suspicious / biggest / changed
        top_inc = top_positive_deltas(deltas, limit=args.top_increases)
        sus = suspicious_increases(deltas, SUSPICIOUS_EXTS, limit=15)
        head_big = biggest_files(head_sizes, limit=args.top_head)
        new_files, del_files = new_and_deleted(base_sizes, head_sizes)
        new_files = new_files[:args.top_new]
        del_files = del_files[:args.top_deleted]

        # Save snapshot (largest files at HEAD)
        save_snapshot(args.snapshot_json, head_big)

        # Baseline compare (largest files over time)
        history_rows: List[Tuple[str,int,int]] = []
        baseline_note = None
        base_dict = load_baseline(args.baseline_json)
        if base_dict is not None:
            history_rows = compare_biglists(base_dict, to_biglist_dict(head_big), limit=args.top_head)
            baseline_note = f"Comparing against baseline: {args.baseline_json}"
        else:
            if args.baseline_json:
                baseline_note = f"Baseline file not found: {args.baseline_json}"
            else:
                baseline_note = "No baseline file provided."

        # output
        if args.markdown:
            md = render_markdown(ref_base, ref_head, include, exclude,
                                 base_total, head_total, delta, rel_delta_pct,
                                 top_inc, global_msg, budget_msgs, sus,
                                 head_big, new_files, del_files,
                                 lfs_note, history_rows, baseline_note)
            print(md)
        else:
            lines: List[str] = []
            lines.append("repo-bloat-guard report")
            lines.append(f"  base: {ref_base}")
            lines.append(f"  head: {ref_head}")
            lines.append(f"  include: {include or ['<all>']}")
            lines.append(f"  exclude: {exclude or ['<none>']}")
            lines.append(f"  base total: {human_bytes(base_total)} ({base_total} B)")
            lines.append(f"  head total: {human_bytes(head_total)} ({head_total} B)")
            lines.append(f"  delta:      {human_bytes(delta)} ({delta} B)  [{rel_delta_pct:.2f}%]")
            if lfs_note:
                lines.append(f"  note: {lfs_note}")
            if top_inc:
                lines.append("\nTop increases:")
                for p, d in top_inc:
                    lines.append(f"  + {human_bytes(d):>10}  {p}")
            else:
                lines.append("\nTop increases: (none)")
            if sus:
                lines.append("\nSuspicious increases:")
                for p, d in sus:
                    lines.append(f"  + {human_bytes(d):>10}  {p}")
            if head_big:
                lines.append("\nBiggest files at HEAD:")
                for p, s in head_big:
                    lines.append(f"    {human_bytes(s):>10}  {p}")
            if new_files or del_files:
                lines.append("\nWhat changed:")
                if new_files:
                    lines.append("  New files:")
                    for p, s in new_files:
                        lines.append(f"    + {human_bytes(s):>9}  {p}")
                if del_files:
                    lines.append("  Deleted files (at base sizes):")
                    for p, s in del_files:
                        lines.append(f"    - {human_bytes(s):>9}  {p}")
            if history_rows:
                lines.append("\nLargest files over time (vs baseline):")
                for p, prev_sz, now_sz in history_rows:
                    diff = now_sz - prev_sz
                    sign = "+" if diff >= 0 else ""
                    lines.append(f"    {p}\n      prev: {human_bytes(prev_sz)}  now: {human_bytes(now_sz)}  Δ: {sign}{human_bytes(diff)}")
            lines.append(f"\nGlobal thresholds: {global_msg}")
            if budgets:
                if budget_msgs:
                    lines.append("Budgets:")
                    for m in budget_msgs:
                        lines.append(f"  - {m}")
                else:
                    lines.append("Budgets: OK (no violations)")
            if baseline_note:
                lines.append(f"Baseline: {baseline_note}")
            print("\n".join(lines))

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
                "exceeded": bool((abs_bytes is not None or rel_pct is not None) and (global_hit)),
                "budgets_exceeded": bool(budget_hit),
                "top_increases": [{"path": p, "delta_bytes": d} for p, d in top_inc],
                "suspicious_increases": [{"path": p, "delta_bytes": d} for p, d in sus],
                "head_biggest": [{"path": p, "bytes": s} for p, s in head_big],
                "new_files": [{"path": p, "bytes": s} for p, s in new_files],
                "deleted_files": [{"path": p, "bytes": s} for p, s in del_files],
                "lfs_note": lfs_note,
                "baseline_note": baseline_note,
                "history_rows": [{"path": p, "prev_bytes": a, "now_bytes": b, "delta_bytes": b - a} for p, a, b in history_rows],
            }
            print("\n===JSON===")
            print(json.dumps(payload, indent=2))

        if args.report:
            return 0
        return 1 if (global_hit or budget_hit) else 0

    except Exception as e:
        print(f"[repo-bloat-guard] ERROR: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
