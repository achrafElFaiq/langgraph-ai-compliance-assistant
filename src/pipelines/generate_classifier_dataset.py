# datasets/generate_classifier_dataset.py

import asyncio
import csv
import os

import psycopg
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

CONCURRENCY = 10
OUTPUT_FILE = "datasets/classifier/questions.csv"
REGULATIONS = ["MiCA", "AI Act", "GDPR", "DORA"]

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

SYSTEM_PROMPT = """You are a compliance professional generating training data.
Given a regulatory article, generate {n} realistic questions that a business, lawyer, or compliance officer might ask — questions that this article directly answers.

Rules:
- Questions must be in French
- Questions must be realistic — the kind a real person would type
- Vary the phrasing: some short, some detailed, some from a business perspective, some from a legal perspective
- Do NOT reference the article number or regulation name in the question
- Return ONLY a numbered list of questions, nothing else

Example output:
1. Dois-je publier un livre blanc avant de lancer mon token ?
2. Quelles informations doivent figurer dans le livre blanc d'un crypto-actif ?
"""

CROSS_PAIR_PROMPT = """You are a compliance professional generating training data.
Generate {n} realistic questions in French that naturally touch BOTH of these regulations: {reg1} and {reg2}.

These should be questions where a compliance officer would need to consult both regulations to answer fully.

Rules:
- Realistic questions a business would ask
- Do NOT mention regulation names in the question
- Return ONLY a numbered list, nothing else
"""

CROSS_TRIPLE_PROMPT = """You are a compliance professional generating training data.
Generate {n} realistic questions in French that naturally touch ALL THREE of these regulations: {reg1}, {reg2} and {reg3}.

These should be questions where a compliance officer would need to consult all three regulations to answer fully.

Rules:
- Realistic questions a business would ask
- Do NOT mention regulation names in the question
- Return ONLY a numbered list, nothing else
"""

semaphore = asyncio.Semaphore(CONCURRENCY)


async def generate_questions_for_article(
    breadcrumb: str, content: str, regulation: str, n: int = 5
) -> list[dict]:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="openai/gpt-oss-120b:free",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
                    {"role": "user", "content": f"Article ({breadcrumb}):\n{content}"}
                ],
                temperature=0.8,
                max_tokens=512 if n <= 5 else 900
            )
            raw = response.choices[0].message.content.strip()
            questions = []
            for line in raw.split("\n"):
                line = line.strip().lstrip("0123456789.) ").strip()
                if len(line) > 15:
                    questions.append({
                        "question": line,
                        "regulation": regulation,
                        "breadcrumb": breadcrumb,
                        "multi_regulation": False
                    })
            return questions
        except Exception as e:
            print(f"  ERROR {breadcrumb}: {e}")
            return []


async def generate_pair_questions(reg1: str, reg2: str, n: int = 75) -> list[dict]:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="openai/gpt-oss-120b:free",
                messages=[
                    {"role": "system", "content": CROSS_PAIR_PROMPT.format(n=n, reg1=reg1, reg2=reg2)},
                    {"role": "user", "content": "Generate the questions now."}
                ],
                temperature=0.9,
                max_tokens=900
            )
            raw = response.choices[0].message.content.strip()
            questions = []
            for line in raw.split("\n"):
                line = line.strip().lstrip("0123456789.) ").strip()
                if len(line) > 15:
                    questions.append({
                        "question": line,
                        "regulation": f"{reg1},{reg2}",
                        "breadcrumb": f"CROSS:{reg1}+{reg2}",
                        "multi_regulation": True
                    })
            return questions
        except Exception as e:
            print(f"  ERROR pair {reg1}+{reg2}: {e}")
            return []


async def generate_triple_questions(reg1: str, reg2: str, reg3: str, n: int = 40) -> list[dict]:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="openai/gpt-oss-120b:free",
                messages=[
                    {"role": "system", "content": CROSS_TRIPLE_PROMPT.format(n=n, reg1=reg1, reg2=reg2, reg3=reg3)},
                    {"role": "user", "content": "Generate the questions now."}
                ],
                temperature=0.9,
                max_tokens=900
            )
            raw = response.choices[0].message.content.strip()
            questions = []
            for line in raw.split("\n"):
                line = line.strip().lstrip("0123456789.) ").strip()
                if len(line) > 15:
                    questions.append({
                        "question": line,
                        "regulation": f"{reg1},{reg2},{reg3}",
                        "breadcrumb": f"CROSS:{reg1}+{reg2}+{reg3}",
                        "multi_regulation": True
                    })
            return questions
        except Exception as e:
            print(f"  ERROR triple {reg1}+{reg2}+{reg3}: {e}")
            return []


async def fetch_articles(conn) -> list[dict]:
    async with conn.cursor() as cur:
        await cur.execute("""
            SELECT
                split_part(breadcrumb, ' > ', 1) as regulation,
                breadcrumb,
                string_agg(content, ' ' ORDER BY chunk_index) as content
            FROM article_chunks
            GROUP BY breadcrumb
            ORDER BY regulation, breadcrumb
        """)
        rows = await cur.fetchall()
        return [{"regulation": r[0], "breadcrumb": r[1], "content": r[2]} for r in rows]


async def main():
    os.makedirs("datasets/classifier", exist_ok=True)

    print("Connecting to DB...")
    conn = await psycopg.AsyncConnection.connect(os.getenv("DATABASE_URL"))

    print("Fetching articles...")
    articles = await fetch_articles(conn)
    await conn.close()

    from collections import Counter
    reg_counts = Counter(a["regulation"] for a in articles)
    print(f"\nFound {len(articles)} articles:")
    for reg, count in sorted(reg_counts.items()):
        n = 10 if reg == "DORA" else 5
        print(f"  {reg}: {count} articles → ~{count * n} questions (n={n})")

    # Single regulation questions
    print(f"\nGenerating single-regulation questions ({CONCURRENCY} parallel)...")
    tasks = [
        generate_questions_for_article(
            a["breadcrumb"],
            a["content"],
            a["regulation"],
            n=10 if a["regulation"] == "DORA" else 5
        )
        for a in articles
    ]

    all_questions = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        all_questions.extend(result)
        completed += 1
        if completed % 25 == 0:
            print(f"  {completed}/{len(articles)} articles done ({len(all_questions)} questions so far)")

    print(f"  Done — {len(all_questions)} single-regulation questions")

    # Cross pair questions
    print(f"\nGenerating cross-pair questions (75 per pair)...")
    cross_pairs = [
        ("MiCA", "GDPR"),
        ("MiCA", "DORA"),
        ("MiCA", "AI Act"),
        ("AI Act", "GDPR"),
        ("AI Act", "DORA"),
        ("DORA", "GDPR"),
    ]
    pair_tasks = [generate_pair_questions(r1, r2, n=75) for r1, r2 in cross_pairs]
    pair_results = await asyncio.gather(*pair_tasks)
    for r1, r2, result in zip([p[0] for p in cross_pairs], [p[1] for p in cross_pairs], pair_results):
        print(f"  {r1}+{r2}: {len(result)} questions")
        all_questions.extend(result)

    # Cross triple questions
    print(f"\nGenerating cross-triple questions (40 per triple)...")
    cross_triples = [
        ("MiCA", "AI Act", "GDPR"),
        ("MiCA", "DORA", "GDPR"),
        ("AI Act", "DORA", "GDPR"),
        ("MiCA", "AI Act", "DORA"),
    ]
    triple_tasks = [generate_triple_questions(r1, r2, r3, n=40) for r1, r2, r3 in cross_triples]
    triple_results = await asyncio.gather(*triple_tasks)
    for (r1, r2, r3), result in zip(cross_triples, triple_results):
        print(f"  {r1}+{r2}+{r3}: {len(result)} questions")
        all_questions.extend(result)

    # Write CSV
    print(f"\nWriting {len(all_questions)} questions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "regulation", "breadcrumb", "multi_regulation"])
        writer.writeheader()
        writer.writerows(all_questions)

    # Summary
    print("\nFinal dataset summary:")
    reg_q_counts = Counter(q["regulation"] for q in all_questions)
    for reg, count in sorted(reg_q_counts.items()):
        print(f"  {reg}: {count} questions")
    multi = sum(1 for q in all_questions if q["multi_regulation"])
    print(f"  Multi-regulation total: {multi}")
    print(f"  Grand total: {len(all_questions)}")
    print(f"\nDone. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())