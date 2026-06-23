#!/usr/bin/env python3
"""Collect distribution / length stats for each dataset."""
import json
import gzip
from pathlib import Path
from collections import Counter

BASE = Path("/home/work/.projects/LLM-OS-Models/fable_distillation/datasets")


def stat_jsonl(path, key=None, limit=None, gz=False):
    counter = Counter()
    total = 0
    lengths = []
    opener = gzip.open if gz else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            if limit and total > limit:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if key and isinstance(obj, dict):
                counter[obj.get(key)] += 1
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        lengths.append(len(v))
                        break
    return total, counter, lengths


print("=" * 70)
print("STATS COLLECTION")
print("=" * 70)

# --- 1. Complete-FABLE.5-traces-2M (sample first 5000 rows of parquet) ---
print("\n## 1. Complete-FABLE.5-traces-2M (parquet, sample 50k rows)")
import pyarrow.parquet as pq
pf = pq.ParquetFile(BASE / "Complete-FABLE.5-traces-2M/data/train.parquet")
total = pf.metadata.num_rows
ds_counter = Counter()
split_counter = Counter()
seen_counter = Counter()
sampled = 0
for batch in pf.iter_batches(batch_size=20000):
    d = batch.column("first_source_dataset").to_pylist()
    s = batch.column("first_source_split").to_pylist()
    sc = batch.column("seen_count").to_pylist()
    ds_counter.update(d)
    split_counter.update(s)
    seen_counter.update(sc)
    sampled += len(d)
    if sampled >= 50000:
        break
print(f"  total rows: {total:,}  (sampled {sampled:,} for distribution)")
print(f"  top source_dataset:")
for k, v in ds_counter.most_common(10):
    print(f"    {v:>6}  {k}")
print(f"  seen_count distribution: min={min(seen_counter)} max={max(seen_counter)}")
print(f"    top seen_count values: {seen_counter.most_common(5)}")

# --- 2. Fable-5-traces ---
print("\n## 2. Fable-5-traces (fable5_cot_merged.jsonl)")
total, ot_counter, _ = stat_jsonl(BASE / "Fable-5-traces/fable5_cot_merged.jsonl", key="output_type")
print(f"  rows: {total:,}")
print(f"  output_type: {dict(ot_counter)}")
# tool names
tool_counter = Counter()
with open(BASE / "Fable-5-traces/fable5_cot_merged.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("output_type") == "tool_use":
            out = obj.get("output")
            if isinstance(out, dict):
                tool_counter[out.get("name", "?")] += 1
print(f"  top tools:")
for k, v in tool_counter.most_common(8):
    print(f"    {v:>5}  {k}")

# --- 3. armand0e ---
print("\n## 3. armand0e/claude-fable-5-claude-code")
armand_dir = BASE / "claude-fable-5-claude-code"
jsonls = sorted(armand_dir.glob("*.jsonl"))
total_rows = 0
type_counter = Counter()
for jp in jsonls:
    with open(jp) as f:
        for line in f:
            total_rows += 1
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    type_counter[obj.get("type", "?")] += 1
            except Exception:
                pass
print(f"  files: {len(jsonls)}, total rows: {total_rows:,}")
print(f"  type distribution (top 15):")
for k, v in type_counter.most_common(15):
    print(f"    {v:>6}  {k}")

# --- 4. WithinUs ---
print("\n## 4. WithinUs 25k")
total, cat_counter, _ = stat_jsonl(BASE / "claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl", key="category")
print(f"  rows: {total:,}")
print(f"  category:")
for k, v in cat_counter.most_common():
    print(f"    {v:>6}  {k}")
# message length stats
assistant_lens = []
with open(BASE / "claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        for m in obj.get("messages", []):
            if m.get("role") == "assistant":
                assistant_lens.append(len(m.get("content", "")))
                break
if assistant_lens:
    assistant_lens.sort()
    n = len(assistant_lens)
    print(f"  assistant content length: min={assistant_lens[0]} median={assistant_lens[n//2]} max={assistant_lens[-1]} mean={sum(assistant_lens)//n}")

# --- 5. lordx64 ---
print("\n## 5. lordx64 SFT (parquet)")
pf = pq.ParquetFile(BASE / "agentic-distill-fable-5-sft/data/train-00000-of-00001.parquet")
texts = pf.read(columns=["text"]).column("text").to_pylist()
tool_count = sum(1 for t in texts if "<tool_use" in t)
think_count = sum(1 for t in texts if "<think>" in t)
lens = sorted(len(t) for t in texts)
n = len(lens)
print(f"  rows: {n}")
print(f"  contains <tool_use>: {tool_count} ({100*tool_count//n}%)")
print(f"  contains <think>: {think_count} ({100*think_count//n}%)")
print(f"  text length: min={lens[0]} median={lens[n//2]} max={lens[-1]} mean={sum(lens)//n}")

# --- 6. Helio ---
print("\n## 6. HelioAI 462x")
think_lens = []
query_langs = Counter()
with open(BASE / "Fable-5-Distill-Reasoning-462x/Claude-Opus-4.7-4.8-DeepReason-462x-105M.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        think_lens.append(len(obj.get("thinking", "")))
        q = obj.get("query", "")
        cyr = sum(1 for c in q[:100] if 'Ѐ' <= c <= 'ӿ')
        if cyr > 5:
            query_langs["cyrillic"] += 1
        else:
            query_langs["latin/other"] += 1
think_lens.sort()
n = len(think_lens)
print(f"  rows: {n}")
print(f"  thinking length: min={think_lens[0]:,} median={think_lens[n//2]:,} max={think_lens[-1]:,} mean={sum(think_lens)//n:,}")
print(f"  query script: {dict(query_langs)}")
print(f"  traces > 300k chars: {sum(1 for l in think_lens if l > 300000)}")

print("\nDONE.")
