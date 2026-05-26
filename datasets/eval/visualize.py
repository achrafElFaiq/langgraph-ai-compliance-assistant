import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/eval/results.json")
data = json.loads(path.read_text())

metrics = ["faithfulness", "factual_correctness", "context_recall", "llm_context_precision_with_reference"]
bar_width = 30

def bar(score):
    if score is None:
        return "N/A"
    filled = int(score * bar_width)
    color = "\033[92m" if score >= 0.7 else "\033[93m" if score >= 0.4 else "\033[91m"
    reset = "\033[0m"
    return f"{color}{'█' * filled}{'░' * (bar_width - filled)}{reset} {score:.2f}"

print(f"\n{'─' * 70}")
print(f"  Eval results — {len(data)} questions")
print(f"{'─' * 70}\n")

for item in sorted(data, key=lambda x: x["id"]):
    print(f"  [{item['id']}]")
    for m in metrics:
        label = m.replace("llm_context_precision_with_reference", "ctx_precision").replace("_", " ")
        score = item.get(m)
        print(f"    {label:<20} {bar(score)}")
    print()

# Averages
print(f"{'─' * 70}")
print("  Averages")
for m in metrics:
    scores = [item[m] for item in data if item.get(m) is not None]
    if scores:
        avg = sum(scores) / len(scores)
        label = m.replace("llm_context_precision_with_reference", "ctx_precision").replace("_", " ")
        print(f"    {label:<20} {bar(avg)}")
print(f"{'─' * 70}\n")

# Distribution
def distribution(scores):
    buckets = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8–1.0": 0}
    for s in scores:
        if s < 0.2:   buckets["0.0–0.2"] += 1
        elif s < 0.4: buckets["0.2–0.4"] += 1
        elif s < 0.6: buckets["0.4–0.6"] += 1
        elif s < 0.8: buckets["0.6–0.8"] += 1
        else:         buckets["0.8–1.0"] += 1
    return buckets

print(f"  Distribution")
print(f"{'─' * 70}")
for m in metrics:
    scores = [item[m] for item in data if item.get(m) is not None]
    if not scores:
        continue
    label = m.replace("llm_context_precision_with_reference", "ctx_precision").replace("_", " ")
    print(f"\n  {label}")
    buckets = distribution(scores)
    max_count = max(buckets.values()) or 1
    for bucket, count in buckets.items():
        filled = int((count / max_count) * 20)
        color = "\033[92m" if bucket in ("0.6–0.8", "0.8–1.0") else "\033[93m" if bucket == "0.4–0.6" else "\033[91m"
        reset = "\033[0m"
        print(f"    {bucket}  {color}{'█' * filled}{'░' * (20 - filled)}{reset}  {count}/{len(scores)}")
print(f"\n{'─' * 70}\n")
