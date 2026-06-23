#!/usr/bin/env python3
"""Fable/Mythos distillation datasets structure analyzer."""
import json
import os
import sys
from pathlib import Path

BASE = Path("/home/work/.projects/LLM-OS-Models/fable_distillation/datasets")


def head_jsonl(path, n=2, max_chars=800):
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                try:
                    obj = json.loads(line)
                    keys = list(obj.keys()) if isinstance(obj, dict) else f"<{type(obj).__name__}>"
                    preview = json.dumps(obj, ensure_ascii=False)[:max_chars]
                    out.append({"idx": i, "keys": keys, "preview": preview})
                except json.JSONDecodeError:
                    out.append({"idx": i, "raw": line[:max_chars]})
    except Exception as e:
        out.append({"error": str(e)})
    return out


def count_jsonl(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return -1


def analyze_jsonl(path, label):
    print(f"\n{'='*70}\n[{label}] {path}\n{'='*70}")
    if not os.path.exists(path):
        print("  FILE NOT FOUND")
        return
    size = os.path.getsize(path)
    print(f"  size: {size/1024/1024:.2f} MB")
    n = count_jsonl(path)
    print(f"  rows: {n:,}")
    samples = head_jsonl(path, n=2, max_chars=1200)
    for s in samples:
        print(f"  sample[{s.get('idx')}]:")
        if "keys" in s:
            print(f"    keys: {s['keys']}")
        print(f"    preview: {s.get('preview', s.get('raw',''))[:600]}")


def analyze_parquet(path, label):
    print(f"\n{'='*70}\n[{label}] {path}\n{'='*70}")
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        n = pf.metadata.num_rows
        print(f"  rows: {n:,}")
        print(f"  schema ({len(schema)} cols):")
        for field in schema:
            print(f"    {field.name}: {field.type}")
        batch = next(pf.iter_batches(batch_size=2))
        rows = batch.to_pylist()
        for i, r in enumerate(rows):
            print(f"  sample[{i}]:")
            for k, v in r.items():
                vs = str(v)
                if len(vs) > 400:
                    vs = vs[:400] + "...[truncated]"
                print(f"    {k}: {vs}")
    except Exception as e:
        print(f"  ERROR: {e}")


print("#" * 70)
print("# FABLE / MYTHOS DISTILLATION DATASETS — STRUCTURE ANALYSIS")
print("#" * 70)

# 1. Complete-FABLE.5-traces-2M
analyze_parquet(BASE / "Complete-FABLE.5-traces-2M/data/train.parquet", "1. Complete-FABLE.5-traces-2M (parquet)")
# raw jsonl is .gz — skip full analysis, just note
rawgz = BASE / "Complete-FABLE.5-traces-2M/raw/fable5_mythos_dedup.jsonl.gz"
print(f"\n  raw gz: {rawgz} ({rawgz.stat().st_size/1024/1024:.1f} MB compressed)")

# 2. Fable-5-traces
analyze_jsonl(BASE / "Fable-5-traces/fable5_cot_merged.jsonl", "2. Fable-5-traces (merged jsonl)")

# 3. claude-fable-5-claude-code (pick biggest + a small one)
armand_dir = BASE / "claude-fable-5-claude-code"
jsonls = sorted(armand_dir.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
print(f"\n{'='*70}\n[3. armand0e/claude-fable-5-claude-code] {armand_dir}\n{'='*70}")
print(f"  total jsonl files: {len(jsonls)}")
if jsonls:
    analyze_jsonl(jsonls[0], "3a. biggest file")

# 4. WithinUs
analyze_jsonl(BASE / "claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl", "4. WithinUs 25k")

# 5. lordx64
analyze_parquet(BASE / "agentic-distill-fable-5-sft/data/train-00000-of-00001.parquet", "5. lordx64 SFT")

# 6. HelioAI
analyze_jsonl(BASE / "Fable-5-Distill-Reasoning-462x/Claude-Opus-4.7-4.8-DeepReason-462x-105M.jsonl", "6. HelioAI 462x")

print("\n\nDONE.")
