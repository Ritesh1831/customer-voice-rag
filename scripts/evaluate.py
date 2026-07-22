"""Measure answer accuracy, abstention correctness, and latency.
Run: python -m scripts.evaluate
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pipeline

CASES = [
    ("How many free ATM withdrawals do I get each month?", ["4"]),
    ("What's the fee for an international transfer?", ["4.99", "0.5"]),
    ("How long does Tier 3 verification take?", ["3", "business day"]),
    ("What's the monthly load limit for Tier 2?", ["5,000"]),
    ("Can I reverse a P2P payment sent to the wrong person?", ["cannot", "no"]),
    ("What interest do savings pockets earn?", ["2.5"]),
    ("How much is the late fee on the credit line?", ["10"]),
    ("What's the fee to pay a new biller?", ["0.50"]),
    ("How long do I have to dispute a charge?", ["60"]),
    ("Which countries can open a NovaPay account?", ["12", "Mexico"]),
    ("What's the foreign transaction fee on my card?", ["2%"]),
    ("How long are statements available for download?", ["24"]),
    ("What's the conversion fee between currencies?", ["0.4"]),
    ("Is there a fee to close my account?", ["no", "free"]),
    ("How long is the account locked after failed logins?", ["30"]),
    ("Can I buy Bitcoin with NovaPay?", None),
    ("How do I open a business merchant account?", None),
    ("Does NovaPay offer life insurance?", None),
    ("Can I open a joint account with my spouse?", None),
    ("Where do I download my 1099 tax form?", None),
    ("What are NovaPay's stock trading fees?", None),
    ("Who is the CEO of NovaPay?", None),
]


def main():
    pipeline.warmup()
    correct = ab_correct = ab_total = ans_total = 0
    lat = []
    failures = []

    for i, (q, expect) in enumerate(CASES):
        # Fresh session per case -> conversation memory never leaks between
        # unrelated eval questions and skews rewrite/cache behavior.
        a = pipeline.answer(q, session_id=f"eval-{i}", use_cache=False)
        lat.append(a.timings.get("total_ms", 0))

        if expect is None:
            ab_total += 1
            if a.abstained:
                ab_correct += 1
            else:
                failures.append(("HALLUCINATED", q, a.text[:110]))
        else:
            ans_total += 1
            low = a.text.lower()
            if not a.abstained and any(e.lower() in low for e in expect):
                correct += 1
            else:
                failures.append(("WRONG/ABSTAINED", q, a.text[:110]))

    print("\n" + "=" * 62)
    print(f"Answerable accuracy : {correct}/{ans_total} "
          f"({100*correct/max(ans_total,1):.1f}%)")
    print(f"Abstention accuracy : {ab_correct}/{ab_total} "
          f"({100*ab_correct/max(ab_total,1):.1f}%)")
    print(f"Hallucination rate  : "
          f"{100*(ab_total-ab_correct)/max(ab_total,1):.1f}%")
    print(f"Latency  p50={statistics.median(lat):.0f}ms  "
          f"mean={statistics.mean(lat):.0f}ms  max={max(lat):.0f}ms")
    print("=" * 62)

    if failures:
        print("\nFailures:")
        for kind, q, txt in failures:
            print(f"  [{kind}] {q}\n     -> {txt}\n")


if __name__ == "__main__":
    main()