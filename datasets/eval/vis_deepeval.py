import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/eval/results_deepeval.json")
data = json.loads(path.read_text())

metrics = [
    ("FaithfulnessMetric",      "faithfulness"),
    ("ContextualRecallMetric",  "contextual recall"),
    ("ContextualPrecisionMetric", "ctx precision"),
    ("AnswerRelevancyMetric",   "answer relevancy"),
]

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
    for key, label in metrics:
        score = item.get(key)
        print(f"    {label:<22} {bar(score)}")
    print()

# Averages
print(f"{'─' * 70}")
print("  Averages")
for key, label in metrics:
    scores = [item[key] for item in data if item.get(key) is not None]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"    {label:<22} {bar(avg)}")
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

print("  Distribution")
print(f"{'─' * 70}")
for key, label in metrics:
    scores = [item[key] for item in data if item.get(key) is not None]
    if not scores:
        continue
    print(f"\n  {label}")
    buckets = distribution(scores)
    max_count = max(buckets.values()) or 1
    for bucket, count in buckets.items():
        filled = int((count / max_count) * 20)
        color = "\033[92m" if bucket in ("0.6–0.8", "0.8–1.0") else "\033[93m" if bucket == "0.4–0.6" else "\033[91m"
        reset = "\033[0m"
        print(f"    {bucket}  {color}{'█' * filled}{'░' * (20 - filled)}{reset}  {count}/{len(scores)}")

print(f"\n{'─' * 70}\n")