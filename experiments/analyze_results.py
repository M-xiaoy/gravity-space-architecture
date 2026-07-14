"""
分析 Gravity Reranker 实验结果
"""
import json, sys
from pathlib import Path

result_dir = Path(__file__).parent / "data" / "run_006"

with open(result_dir / "summary.json", encoding="utf-8") as f:
    summary = json.load(f)

with open(result_dir / "results.jsonl", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

print("=" * 60)
print("Gravity Reranker v0.1 实验结果分析")
print("=" * 60)
print(f"模型: {summary['config']['model_name']}")
print(f"样本数: {summary['total_samples']}")
print(f"标准准确率: {summary['standard_accuracy']:.1%}")
print(f"Gravity准确率: {summary['gravity_accuracy']:.1%}")
print()

# 看 diversity 分布
entropies = [r["diversity_metrics"]["cross_path_entropy"] for r in results]
print(f"多样性熵范围: {min(entropies):.3f} ~ {max(entropies):.3f}")
print(f"多样性熵均值: {sum(entropies)/len(entropies):.3f}")
print()

# 看哪些样本被 diversity 影响
print("各样本详情:")
print(f"{'ID':<12} {'正确':<6} {'标准':<6} {'Gravity':<8} {'熵':<8} {'标准对':<8} {'Gravity对':<8}")
print("-" * 60)
for r in results:
    std_ok = "OK" if r["standard_correct"] else "XX"
    grav_ok = "OK" if r["gravity_correct"] else "XX"
    ent = r["diversity_metrics"]["cross_path_entropy"]
    print(f"{r['trial_id']:<12} {r['correct_answer']:<6} "
          f"{str(r['standard_prediction']):<6} {str(r['gravity_prediction']):<8} "
          f"{ent:<8.3f} {std_ok:<8} {grav_ok:<8}")

print()

# 对比标准 vs Gravity 预测不同的样本
diff = [r for r in results if r["standard_prediction"] != r["gravity_prediction"]]
print(f"标准与 Gravity 预测不同的样本: {len(diff)} 条")

# 看几个样本的实际生成内容
print("\n\n=== 样本原始输出（前3条）===")
for r in results[:3]:
    print(f"\n--- {r['trial_id']} ---")
    print(f"正确答案: {r['correct_answer']}")
    print(f"路径数: {len(r['paths'])}")
    print(f"答案分布: {r['diversity_metrics']['answer_distribution']}")
    for p in r['paths'][:2]:
        print(f"  Path {p['path_id']}: ans={p['final_answer']}, avg_lp={p['avg_log_prob']:.3f}")
        resp = p['full_response'][:150]
        print(f"    {resp}")
