#!/usr/bin/env python3
"""memory-bench: pasted-history baseline vs LeapMemory recall.

Two conditions. Same model, same questions, same instructions.
  A (baseline): the entire conversation corpus is pasted into the prompt.
  B (memory):   only a LeapMemory recall result is placed in the prompt.

Token counts are read from the provider's billed usage field, never
counted locally. A probe counts toward the reduction number only if
BOTH conditions answered it correctly.
"""

import argparse
import json
import os
import statistics
import sys
import time

import requests
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)


def env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env var: {name}")
    return v


# Identical for both conditions. Softened wording: the model is told to
# search before abstaining, which helps the baseline, not the memory side.
SYSTEM_PROMPT = (
    "You answer questions about a past conversation using only the context "
    "provided. Search the context carefully before answering. Answer "
    "concisely. Reply exactly UNKNOWN only if you are certain the answer "
    "is not in the context."
)


def load_corpus(path):
    turns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    if not turns:
        sys.exit(f"empty corpus: {path}")
    return turns


def load_probes(path):
    with open(path) as f:
        probes = yaml.safe_load(f)
    if not probes:
        sys.exit(f"empty probes: {path}")
    for p in probes:
        for k in ("id", "question", "expected"):
            if k not in p:
                sys.exit(f"probe missing '{k}': {p}")
    return probes


def corpus_as_text(turns):
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def ask_model(context, question):
    r = requests.post(
        f"{env('OPENAI_BASE_URL')}/chat/completions",
        headers={"Authorization": f"Bearer {env('OPENAI_API_KEY')}"},
        json={
            "model": env("MODEL"),
            "temperature": 0,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
                },
            ],
        },
        timeout=180,
    )
    r.raise_for_status()
    d = r.json()
    answer = d["choices"][0]["message"]["content"].strip()
    billed_input_tokens = d["usage"]["prompt_tokens"]
    return answer, billed_input_tokens


def lm_request(method, path, body=None):
    r = requests.request(
        method,
        f"{env('LM_API_URL')}{path}",
        headers={"Authorization": f"Bearer {env('LM_API_KEY')}"},
        json=body,
        timeout=180,
    )
    d = r.json()
    if not d.get("success"):
        sys.exit(f"LM error on {path}: {d.get('code')}: {d.get('message')}")
    return d["data"]


def lm_recall_context(tenant, question):
    data = lm_request("POST", f"/v1/tenants/{tenant}/recall", {"query": question})
    facts = [f"- {f['sentence']}" for f in data.get("facts", [])]
    chunks = [c["content"] for c in data.get("chunks", [])]
    parts = []
    if facts:
        parts.append("Facts:\n" + "\n".join(facts))
    if chunks:
        parts.append("Excerpts:\n" + "\n\n".join(chunks))
    return "\n\n".join(parts) if parts else "(no memories recalled)"


def ingest(tenant, turns, settle_seconds):
    for i in range(0, len(turns), 100):
        batch = turns[i : i + 100]
        lm_request(
            "POST",
            f"/v1/tenants/{tenant}/turns/batch",
            {"turns": batch},
        )
        print(f"ingested {i + len(batch)}/{len(turns)} turns")
    print(f"waiting {settle_seconds}s for extraction to settle...")
    time.sleep(settle_seconds)


TENANT_FILE = ".bench_tenant"


def fresh_tenant():
    """Fresh tenant per ingest; previous one hard-deleted. demo.py pattern."""
    import random
    import string

    if os.path.exists(TENANT_FILE):
        with open(TENANT_FILE) as f:
            old = f.read().strip()
        if old:
            requests.delete(
                f"{env('LM_API_URL')}/v1/tenants/{old}?hard=true",
                headers={"Authorization": f"Bearer {env('LM_API_KEY')}"},
                timeout=30,
            )
            print(f"previous tenant '{old}' hard-deleted")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    tenant = f"bench_{suffix}"
    lm_request("POST", "/v1/tenants", {"tenant_id": tenant})
    with open(TENANT_FILE, "w") as f:
        f.write(tenant)
    print(f"created tenant {tenant}")
    return tenant


def saved_tenant():
    if not os.path.exists(TENANT_FILE):
        sys.exit(f"no {TENANT_FILE} found; run once with --ingest first")
    with open(TENANT_FILE) as f:
        tenant = f.read().strip()
    if not tenant:
        sys.exit(f"{TENANT_FILE} is empty; run once with --ingest first")
    return tenant


def is_correct(answer, expected):
    if isinstance(expected, str):
        expected = [expected]
    return any(e.lower() in answer.lower() for e in expected)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="jsonl of conversation turns")
    ap.add_argument("--probes", required=True, help="yaml of questions + expected answers")
    ap.add_argument("--ingest", action="store_true", help="ingest the corpus into the LM tenant first")
    ap.add_argument("--settle", type=int, default=120, help="seconds to wait after ingest for extraction")
    ap.add_argument("--out", default="results.jsonl", help="raw per-probe output file (gitignored)")
    args = ap.parse_args()

    turns = load_corpus(args.corpus)
    probes = load_probes(args.probes)

    if args.ingest:
        tenant = fresh_tenant()
        ingest(tenant, turns, args.settle)
    else:
        tenant = saved_tenant()
    print(f"tenant: {tenant}")

    pasted = corpus_as_text(turns)
    rows = []

    for p in probes:
        a_answer, a_tokens = ask_model(pasted, p["question"])
        b_context = lm_recall_context(tenant, p["question"])
        b_answer, b_tokens = ask_model(b_context, p["question"])

        row = {
            "probe": p["id"],
            "question": p["question"],
            "expected": p["expected"],
            "baseline_answer": a_answer,
            "baseline_tokens": a_tokens,
            "baseline_correct": is_correct(a_answer, p["expected"]),
            "recall_answer": b_answer,
            "recall_tokens": b_tokens,
            "recall_correct": is_correct(b_answer, p["expected"]),
        }
        rows.append(row)
        print(
            f"{p['id']}: baseline {a_tokens} tok "
            f"{'OK' if row['baseline_correct'] else 'MISS'} | "
            f"recall {b_tokens} tok "
            f"{'OK' if row['recall_correct'] else 'MISS'}"
        )

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    scored = [r for r in rows if r["baseline_correct"] and r["recall_correct"]]
    print("\n== results ==")
    print(f"probes: {len(rows)}")
    print(f"baseline accuracy: {sum(r['baseline_correct'] for r in rows)}/{len(rows)}")
    print(f"recall accuracy:   {sum(r['recall_correct'] for r in rows)}/{len(rows)}")
    print(f"scored (both correct): {len(scored)}")

    for r in rows:
        if not (r["baseline_correct"] and r["recall_correct"]):
            side = "baseline" if not r["baseline_correct"] else "recall"
            print(f"\nreview {r['probe']} ({side} miss):")
            print(f"  expected: {r['expected']}")
            print(f"  baseline: {r['baseline_answer'][:200]}")
            print(f"  recall:   {r['recall_answer'][:200]}")

    if scored:
        reductions = [1 - r["recall_tokens"] / r["baseline_tokens"] for r in scored]
        print(f"\nmedian token reduction: {statistics.median(reductions):.1%}")
        print(f"range: {min(reductions):.1%} - {max(reductions):.1%}")
    else:
        print("\nno probes scored; no reduction reported")


if __name__ == "__main__":
    main()