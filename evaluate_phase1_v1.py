#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harel Insurance RAG Evaluation Harness - Phase 1 (semantic + telemetry)

Scores a baseline_answers.jsonl file against reference_questions.json:
  - answer relevance        (LLM judge, factual pool only)
  - hallucination rate      (LLM judge, factual pool only)
  - refusal rate            (LLM judge, over all evaluated questions)
  - answer latency          (read from the predictions file, NOT the judge)
  - judge cost + judge latency (evaluation overhead, reported separately)

Judge is pinned + temperature 0 + forced JSON so runs are comparable.

    export OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
    export OPENAI_API_KEY=<shared course key>
    python evaluate_phase1.py \
        --questions reference_questions.json \
        --predictions baseline_answers.jsonl \
        --output eval_baseline.jsonl

Prices default to the DeepSeek-V4-Pro public rate read off the endpoints page
on 2026-07-19 ($1.75 in / $3.50 out per 1M). VERIFY before trusting the cost
line -- prices drift; pass --price-in / --price-out to override.
"""
import argparse
import json
import os
import statistics
import sys
import time

from openai import OpenAI

JUDGE_MODEL = "deepseek-ai/DeepSeek-V4-Pro"   # pin the judge; keep fixed across ALL runs


def parse_args():
    p = argparse.ArgumentParser(description="Harel eval harness - Phase 1")
    p.add_argument("--questions", required=True, help="reference_questions.json")
    p.add_argument("--predictions", required=True, help="baseline_answers.jsonl")
    p.add_argument("--output", required=True, help="per-question eval output (.jsonl)")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--mode", choices=["baseline", "rag"], default="baseline",
                   help="baseline = no retrieval, so a justified refusal is impossible; "
                        "rag = retrieval exists, judge decides justification")
    p.add_argument("--price-in", type=float, default=1.75, help="$/1M input tokens")
    p.add_argument("--price-out", type=float, default=3.50, help="$/1M output tokens")
    p.add_argument("--retries", type=int, default=1, help="judge retries on API/JSON failure")
    p.add_argument("--selfcheck", type=int, default=3,
                   help="re-judge the first N factual items to verify determinism (0=skip)")
    return p.parse_args()


def load_reference(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):            # staff sets wrap the list
        data = data["questions"]
    return {item["id"]: item for item in data}


def build_judge_messages(question, generated_answer, ground_truth_answer, mode):
    # NOTE: the string "JSON" must appear in the prompt for json_object mode.
    justification_rule = (
        "2. If it IS a refusal: this is the BASELINE dataset with no retrieval context, and every "
        "question has a known ground-truth answer, so any refusal here is UNJUSTIFIED "
        "(is_justified_refusal = false). [baseline-only rule]"
        if mode == "baseline" else
        "2. If it IS a refusal: judge justification against the provided context. A refusal is "
        "JUSTIFIED only if the ground-truth answer could not be established from what the system had."
    )
    system_prompt = (
        "You are a strict, objective evaluator for a Harel Insurance support chatbot. Compare the "
        "chatbot's response against the official ground-truth answer and return ONLY a JSON object "
        "with EXACTLY these keys:\n"
        "{\n"
        '  "response_type": "factual" | "refusal",\n'
        '  "is_justified_refusal": true | false,\n'
        '  "answer_relevance": 0.0 | 0.5 | 1.0,\n'
        '  "hallucination_detected": true | false\n'
        "}\n\n"
        "Rules:\n"
        "1. A refusal = the response declines to answer / says it lacks information / defers to an "
        "agent WITHOUT giving the policy fact.\n"
        f"{justification_rule}\n"
        "3. answer_relevance measures agreement with the ground-truth fact, in discrete steps: "
        "1.0 = states the correct fact; 0.5 = partially correct or incomplete; 0.0 = wrong, "
        "irrelevant, or a refusal.\n"
        "4. hallucination_detected = true ONLY if the response asserts a concrete fact that "
        "contradicts or invents policy detail not supported by the ground truth. A refusal is "
        "NOT a hallucination."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Ground-truth answer:\n{ground_truth_answer}\n\n"
        f"Chatbot response:\n{generated_answer}\n\n"
        "Return the JSON object."
    )
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


def call_judge(client, model, messages, retries):
    """Returns (parsed_dict, in_tokens, out_tokens, latency_s) or raises after retries."""
    last = None
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"})
            dt = time.perf_counter() - t0
            parsed = json.loads(resp.choices[0].message.content)
            return parsed, resp.usage.prompt_tokens, resp.usage.completion_tokens, dt
        except Exception as e:               # network error OR malformed JSON
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def score_row(judge_out, is_refusal_mode_baseline):
    """Apply the resolved refusal rules. Returns (status, relevance_or_None, halluc_or_None)."""
    is_refusal = judge_out.get("response_type") == "refusal"
    if not is_refusal:
        rel = float(judge_out.get("answer_relevance", 0.0))
        hall = bool(judge_out.get("hallucination_detected", True))   # conservative default
        return "factual", rel, hall
    # refusal branch
    justified = bool(judge_out.get("is_justified_refusal", False))
    if is_refusal_mode_baseline:
        justified = False                    # baseline: impossible to be justified
    if justified:
        return "justified_refusal", None, None      # excluded from factual pool
    return "unjustified_refusal", 0.0, False         # kept in pool, forced 0 relevance


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        sys.exit("Missing OPENAI_API_KEY or OPENAI_BASE_URL.")
    client = OpenAI(base_url=base_url, api_key=api_key)

    ref = load_reference(args.questions)

    factual_pool = []            # dicts: {relevance, hallucination}
    answer_latencies_ms = []     # SYSTEM latency, read from predictions
    judge_latencies = []         # evaluation overhead, reported separately
    refusal_count = 0
    evaluated = 0                # successfully judged
    judge_failed = 0
    missing_ref = 0
    in_tok = out_tok = 0
    selfcheck_records = []       # (id, messages) for determinism re-run

    with open(args.predictions, encoding="utf-8") as pred_f, \
         open(args.output, "w", encoding="utf-8") as out_f:
        for line in pred_f:
            if not line.strip():
                continue
            pred = json.loads(line)
            qid = pred.get("id") or pred.get("question_id")
            answer = pred.get("generated_answer") or pred.get("answer") or ""
            if qid not in ref:
                print(f"warn: prediction id {qid!r} not in reference; skipping", file=sys.stderr)
                missing_ref += 1
                continue

            # SYSTEM latency comes from the predictions file (the graded metric)
            if isinstance(pred.get("latency_ms"), (int, float)):
                answer_latencies_ms.append(pred["latency_ms"])

            node = ref[qid]
            messages = build_judge_messages(node["question"], answer,
                                            node["ground_truth_answer"], args.mode)
            try:
                judge_out, i_t, o_t, jdt = call_judge(client, args.judge_model,
                                                      messages, args.retries)
            except Exception as e:
                print(f"judge FAILED on {qid}: {e}", file=sys.stderr)
                judge_failed += 1
                out_f.write(json.dumps({"question_id": qid, "status": "judge_error",
                                        "error": str(e)}, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            evaluated += 1
            in_tok += i_t
            out_tok += o_t
            judge_latencies.append(jdt)
            if len(selfcheck_records) < args.selfcheck:
                selfcheck_records.append((qid, messages, judge_out))

            status, rel, hall = score_row(judge_out, args.mode == "baseline")
            if status == "unjustified_refusal" or status == "justified_refusal":
                refusal_count += 1
            if rel is not None:              # factual OR unjustified refusal -> in pool
                factual_pool.append({"relevance": rel, "hallucination": hall})

            out_f.write(json.dumps({
                "question_id": qid, "status": status,
                "answer_relevance": rel, "hallucination_detected": hall,
                "answer_latency_ms": pred.get("latency_ms"),
                "judge_latency_s": jdt, "raw_judge_output": judge_out,
            }, ensure_ascii=False) + "\n")
            out_f.flush()

    # ---- determinism self-check ---------------------------------------------
    determinism_note = "skipped"
    if args.selfcheck and selfcheck_records:
        mismatches = 0
        for qid, messages, first in selfcheck_records:
            try:
                second, _, _, _ = call_judge(client, args.judge_model, messages, args.retries)
                if second != first:
                    mismatches += 1
                    print(f"determinism MISMATCH on {qid}:\n  {first}\n  {second}", file=sys.stderr)
            except Exception as e:
                print(f"selfcheck judge failed on {qid}: {e}", file=sys.stderr)
        determinism_note = (f"{len(selfcheck_records) - mismatches}/{len(selfcheck_records)} "
                            f"identical on re-run")

    # ---- aggregation ---------------------------------------------------------
    A = len(factual_pool)
    mean_rel = sum(r["relevance"] for r in factual_pool) / A if A else 0.0
    hall_rate = sum(1 for r in factual_pool if r["hallucination"]) / A if A else 0.0
    refusal_rate = refusal_count / evaluated if evaluated else 0.0

    cost = in_tok / 1e6 * args.price_in + out_tok / 1e6 * args.price_out

    def pct(xs, q):
        return statistics.quantiles(xs, n=100)[q - 1] if len(xs) >= 2 else (xs[0] if xs else 0.0)

    print("\n" + "=" * 56)
    print("HAREL EVALUATION REPORT  (Phase 1 - baseline)")
    print("=" * 56)
    print(f"reference questions        : {len(ref)}")
    print(f"predictions evaluated      : {evaluated}")
    print(f"judge failures             : {judge_failed}   (excluded from all metrics)")
    print(f"predictions w/o reference  : {missing_ref}")
    print("-" * 56)
    print(f"factual pool |A|           : {A}")
    print(f"mean answer relevance      : {mean_rel:.4f}   (over A)")
    print(f"hallucination rate         : {hall_rate:.4f}   (over A)")
    print(f"refusal rate               : {refusal_rate:.4f}   (over evaluated)")
    print("-" * 56)
    if answer_latencies_ms:
        print(f"answer latency ms  mean    : {statistics.mean(answer_latencies_ms):.0f}")
        print(f"answer latency ms  median  : {statistics.median(answer_latencies_ms):.0f}")
        print(f"answer latency ms  p95     : {pct(answer_latencies_ms, 95):.0f}   (n={len(answer_latencies_ms)}, indicative)")
        print(f"answer latency ms  max     : {max(answer_latencies_ms):.0f}")
    else:
        print("answer latency             : no latency_ms field in predictions")
    print("-" * 56)
    print(f"judge model                : {args.judge_model}")
    print(f"judge determinism check    : {determinism_note}")
    print(f"judge tokens in/out        : {in_tok} / {out_tok}")
    print(f"judge cost (USD)           : ${cost:.4f}   @ ${args.price_in}/${args.price_out} per 1M")
    if judge_latencies:
        print(f"judge latency s  mean      : {statistics.mean(judge_latencies):.2f}")
    print("=" * 56)
    if judge_failed:
        print(f"WARNING: {judge_failed} judge failure(s) -- numbers are incomplete. "
              f"Re-run before trusting them.", file=sys.stderr)


if __name__ == "__main__":
    main()
