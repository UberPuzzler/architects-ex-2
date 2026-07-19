"""
Stage 1 - PROMPT STRATEGY B: certainty-gated ("answer only if certain").
Identical to baseline_runner.py except for DEFAULT_SYSTEM. 
"""
import argparse
import json
import os
import time

import litellm

DEFAULT_SYSTEM = ("You are a customer-support assistant for Harel Insurance (Israel). "
                  "Answer the customer's question in the language it was asked. "
                  "Answer ONLY if you are certain of the specific policy fact being asked. "
                  "If you are not certain of the exact fact, do NOT guess and do NOT give a "
                  "general explanation instead -- reply with exactly this and nothing else: "
                  "איני יודע. "
                  "If you cite a source, cite the exact document and page.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument("--out", default="baseline_answers_certain.jsonl")
    args = ap.parse_args()

    model, kwargs = args.model, {}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
        model = f"openai/{model.removeprefix('openai/')}"
    elif "/" not in model:
        model = f"openai/{model}"

    questions = json.load(open(args.questions, encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions["questions"]
    with open(args.out, "w", encoding="utf-8") as out:
        for q in questions:
            t0 = time.time()
            resp = litellm.completion(model=model, messages=[
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": q["question"]}],
                timeout=120, **kwargs)
            rec = {"id": q["id"],
                   "answer": resp.choices[0].message.content,
                   "citations": [],
                   "latency_ms": (time.time() - t0) * 1000,
                   "tokens": {"prompt": resp.usage.prompt_tokens,
                              "completion": resp.usage.completion_tokens}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{q['id']}: {rec['answer'][:70]!r}... ({rec['latency_ms']:.0f} ms)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
