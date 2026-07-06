BAR_WIDTH = 30
METRICS = [
    ("FaithfulnessMetric",        "Faithfulness"),
    ("ContextualRecallMetric",     "Ctx Recall"),
    ("ContextualPrecisionMetric",  "Ctx Precision"),
    ("AnswerRelevancyMetric",      "Answer Relevancy"),
]


def _bar(score: float | None) -> str:
    if score is None:
        return "N/A"
    filled = int(score * BAR_WIDTH)
    color = "\033[92m" if score >= 0.7 else "\033[93m" if score >= 0.4 else "\033[91m"
    return f"{color}{'█' * filled}{'░' * (BAR_WIDTH - filled)}\033[0m {score:.2f}"


def _distribution(scores: list[float]) -> dict[str, int]:
    buckets = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8–1.0": 0}
    for s in scores:
        if s < 0.2:
            buckets["0.0–0.2"] += 1
        elif s < 0.4:
            buckets["0.2–0.4"] += 1
        elif s < 0.6:
            buckets["0.4–0.6"] += 1
        elif s < 0.8:
            buckets["0.6–0.8"] += 1
        else:
            buckets["0.8–1.0"] += 1
    return buckets


def print_results(data: list[dict]) -> None:
    sep = "─" * 70

    print(f"\n{sep}")
    print(f"  Eval results — {len(data)} questions")
    print(f"{sep}\n")
    for item in sorted(data, key=lambda x: x["id"]):
        print(f"  [{item['id']}]")
        for key, label in METRICS:
            print(f"    {label:<22} {_bar(item.get(key))}")
        print()

    print(sep)
    print("  Averages")
    for key, label in METRICS:
        scores = [item[key] for item in data if item.get(key) is not None]
        if scores:
            print(f"    {label:<22} {_bar(sum(scores) / len(scores))}")
    print(f"{sep}\n")

    retries = [item.get("retry_count", 0) for item in data]
    print("  Retry Stats")
    print(sep)
    print(f"    Avg retries:  {sum(retries) / len(retries):.2f}")
    print(f"    Max retries:  {max(retries)}")
    print(f"    Hit cap (3):  {sum(1 for r in retries if r >= 3)} / {len(retries)} questions")
    print()
    for entry in sorted(data, key=lambda x: x.get("retry_count", 0), reverse=True):
        r = entry.get("retry_count", 0)
        bar = "█" * r + "░" * (3 - min(r, 3))
        flag = " ← hit cap" if r >= 3 else ""
        print(f"    {entry['id']:<6} {bar} {r} retries{flag}")
    print(f"{sep}\n")

    print("  Distribution")
    print(sep)
    for key, label in METRICS:
        scores = [item[key] for item in data if item.get(key) is not None]
        if not scores:
            continue
        print(f"\n  {label}")
        buckets = _distribution(scores)
        max_count = max(buckets.values()) or 1
        for bucket, count in buckets.items():
            filled = int((count / max_count) * 20)
            color = "\033[92m" if bucket in ("0.6–0.8", "0.8–1.0") else "\033[93m" if bucket == "0.4–0.6" else "\033[91m"
            print(f"    {bucket}  {color}{'█' * filled}{'░' * (20 - filled)}\033[0m  {count}/{len(scores)}")
    print(f"\n{sep}\n")