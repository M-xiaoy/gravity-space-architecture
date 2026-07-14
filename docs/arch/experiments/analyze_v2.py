import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else "data/run_009/summary.json"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
print("Per-sample detail:")
for s in d["per_sample"]:
    ok = "OK" if s["correct"] else "XX"
    print(f'  {s["id"]:<10} {s["eval_type"]:<8} {ok:<4} div={s["avg_diversity"]:.3f}')
print()
print(f'Choice accuracy: {d["choice_accuracy"]:.1%}')
print(f'Free accuracy: {d["free_accuracy"]:.1%}')
print(f'Uncertainty accuracy: {d["uncertainty_accuracy"]:.1%}')
print(f'Avg diversity: {d["avg_diversity_ratio"]:.3f}')
print(f'Token change: {d["tokens_changed_total"]}/{d["tokens_analyzed_total"]}')
