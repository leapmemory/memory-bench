#!/usr/bin/env python3
"""memory-bench: pasted-history baseline vs LeapMemory recall.

Two conditions. Same model, same questions, same instructions.
  A (baseline): the entire conversation corpus is pasted into the prompt.
  B (memory):   only a LeapMemory recall result is placed in the prompt.

Token counts are read from the provider's billed usage field, never
counted locally. A probe counts toward the reduction number only if
BOTH conditions answered it correctly. After ingest the harness polls
the LM turns/status endpoint until every turn is indexed; there is no
fixed settle timer.
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


def ingest(tenant, turns):
    for i in range(0, len(turns), 100):
        batch = turns[i : i + 100]
        lm_request(
            "POST",
            f"/v1/tenants/{tenant}/turns/batch",
            {"turns": batch},
        )
        print(f"ingested {i + len(batch)}/{len(turns)} turns")


def wait_indexed(tenant):
    """Poll turn statuses until every turn is indexed. No blind sleeps.

    Interval 10s: digestion runs minutes, the status read is one cheap
    aggregate, finer polling buys nothing. Ceiling 1800s: worst observed
    full digestion ~15 min for a 98-turn corpus, doubled for margin. A
    'failed' count is transient while the reconciler retries, so failure
    only aborts at the ceiling, loudly, with the failed ids.
    """
    interval = int(os.environ.get("BENCH_POLL_INTERVAL_SECONDS", "10"))
    ceiling = int(os.environ.get("BENCH_POLL_CEILING_SECONDS", "1800"))
    start = time.time()
    last = None
    while True:
        s = lm_request("GET", f"/v1/tenants/{tenant}/turns/status")
        state = (s["indexed"], s["pending"], s["failed"], s["total"])
        if state != last:
            print(
                f"indexed {s['indexed']}/{s['total']} "
                f"(pending {s['pending']}, failed {s['failed']})"
            )
            last = state
        if s["indexed"] == s["total"]:
            print(f"digestion complete in {int(time.time() - start)}s")
            return
        if time.time() - start > ceiling:
            sys.exit(
                f"digestion did not finish within {ceiling}s: "
                f"indexed {s['indexed']}/{s['total']}, "
                f"failed ids: {s['failed_ids']}"
            )
        time.sleep(interval)


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


# ─── Presentation (printing only; results.jsonl stays raw) ───────

BAR_WIDTH = 40  # widest bar in columns; every bar scales off the probe's own baseline

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def mark(ok):
    return f"{GREEN}OK{RESET}" if ok else f"{RED}MISS{RESET}"


def bar(tokens, baseline_tokens, ch):
    width = max(1, round(BAR_WIDTH * tokens / baseline_tokens))
    return ch * width


def phase(title):
    print(f"\n{BOLD}── {title} " + "─" * max(0, 50 - len(title)) + RESET)


def print_probe(row):
    pct = 1 - row["recall_tokens"] / row["baseline_tokens"]
    print(f"{BOLD}{row['probe']}{RESET}  {row['question'][:60]}")
    print(
        f"  baseline {str(row['baseline_tokens']).rjust(5)} tok {mark(row['baseline_correct'])}  "
        f"{bar(row['baseline_tokens'], row['baseline_tokens'], '░')}"
    )
    print(
        f"  recall   {str(row['recall_tokens']).rjust(5)} tok {mark(row['recall_correct'])}  "
        f"{GREEN}{bar(row['recall_tokens'], row['baseline_tokens'], '█')}{RESET} ({pct:.0%} less)"
    )


def verdict(rows, scored, reductions):
    label_width = 22
    pairs = [
        ("probes", str(len(rows))),
        ("baseline accuracy", f"{sum(r['baseline_correct'] for r in rows)}/{len(rows)}"),
        ("recall accuracy", f"{sum(r['recall_correct'] for r in rows)}/{len(rows)}"),
        ("scored (both correct)", str(len(scored))),
    ]
    if reductions:
        pairs.append(("median token reduction", f"{statistics.median(reductions):.1%}"))
        pairs.append(("range", f"{min(reductions):.1%} - {max(reductions):.1%}"))
    lines = [f"{label.ljust(label_width)} {value}" for label, value in pairs]
    if not reductions:
        lines.append("no probes scored; no reduction reported")
    inner = max(len(l) for l in lines)
    print("\n┌" + "─" * (inner + 2) + "┐")
    for l in lines:
        print(f"│ {l.ljust(inner)} │")
    print("└" + "─" * (inner + 2) + "┘")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="jsonl of conversation turns")
    ap.add_argument("--probes", required=True, help="yaml of questions + expected answers")
    ap.add_argument("--ingest", action="store_true", help="ingest the corpus into the LM tenant first")
    ap.add_argument("--out", default="results.jsonl", help="raw per-probe output file (gitignored)")
    args = ap.parse_args()

    turns = load_corpus(args.corpus)
    probes = load_probes(args.probes)

    if args.ingest:
        phase("ingest")
        tenant = fresh_tenant()
        ingest(tenant, turns)
        phase("digestion")
        wait_indexed(tenant)
    else:
        tenant = saved_tenant()
    print(f"tenant: {tenant}")
    phase("probes")

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
        print_probe(row)

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    scored = [r for r in rows if r["baseline_correct"] and r["recall_correct"]]
    for r in rows:
        if not (r["baseline_correct"] and r["recall_correct"]):
            side = "baseline" if not r["baseline_correct"] else "recall"
            print(f"\nreview {r['probe']} ({side} miss):")
            print(f"  expected: {r['expected']}")
            print(f"  baseline: {r['baseline_answer'][:200]}")
            print(f"  recall:   {r['recall_answer'][:200]}")

    phase("verdict")
    reductions = (
        [1 - r["recall_tokens"] / r["baseline_tokens"] for r in scored] if scored else []
    )
    verdict(rows, scored, reductions)


if __name__ == "__main__":
    main()