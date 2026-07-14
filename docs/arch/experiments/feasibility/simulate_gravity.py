"""
Phase D: 球体创建 + scan_terrain 模拟
读 features.npz -> 模拟感知过程 -> 球体增长曲线 -> p_t -> 相关性分析

用法: python simulate_gravity.py runs/ctr_001_xxxx/features.npz
"""

import json, sys, argparse
from pathlib import Path
import numpy as np


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Path to features.npz (from extract_features.py)")
    parser.add_argument("--match-threshold", type=float, default=0.85,
                        help="Cosine threshold for sphere matching")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K neighbors for scan_terrain aggregation")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    run_dir = npz_path.parent

    print(f"加载特征: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]          # (seq_len, 45)
    features_pca = data["features_pca"]  # (seq_len, pca_dim)

    seq_len = features.shape[0]
    print(f"  序列长度: {seq_len} | 特征维度: {features.shape[1]}")

    # ── 球体创建模拟 ──
    print(f"\n[Phase D] 球体创建匹配模拟 (threshold={args.match_threshold})")
    spheres = []                     # list of feature vectors (45-dim each)
    sphere_activations = []           # count of how many times each sphere is activated
    sphere_created_at = []            # creation step
    sphere_activation_count = []      # cumulative activation count
    sphere_growth = []                # total spheres after each step

    for pos in range(seq_len):
        feat = features[pos]
        matched = False
        for sid, sfeat in enumerate(spheres):
            sim = cosine_sim(feat, sfeat)
            if sim > args.match_threshold:
                # 匹配 → 引力加强（更新为加权平均）
                old_count = sphere_activation_count[sid]
                spheres[sid] = (spheres[sid] * old_count + feat) / (old_count + 1)
                sphere_activation_count[sid] += 1
                sphere_activations[sid].append(pos)
                matched = True
                break

        if not matched:
            # 创建新球体
            spheres.append(feat.copy())
            sphere_activations.append([pos])
            sphere_activation_count.append(1)
            sphere_created_at.append(pos)

        sphere_growth.append(len(spheres))

    n_spheres = len(spheres)
    print(f"  总球体数: {n_spheres}")
    print(f"  球体增长 (首5): {sphere_growth[:5]}")
    print(f"  球体增长 (末5): {sphere_growth[-5:]}")
    print(f"  激活分布: 均值={np.mean(sphere_activation_count):.1f}, "
          f"最大={max(sphere_activation_count)}")

    # 球体数量增长曲线参数
    growth_linear = sphere_growth[-1] / seq_len if seq_len > 0 else 0

    # ── scan_terrain 模拟 ──
    print(f"\n  scan_terrain 模拟 (top-K={args.top_k})...")
    p_t_norms = []     # ||p_t|| per step
    p_t_vectors = []   # p_t per step (45-dim)

    for pos in range(seq_len):
        if pos == 0:
            p_t_norms.append(0.0)
            p_t_vectors.append(np.zeros(features.shape[1]))
            continue

        # 收集 pos 之前的所有球体
        prev_spheres = np.array(spheres[:len(spheres)])  # lazily get up to current count
        # 实际上, 在 pos 这个位置时, 球体数量 = sphere_growth[pos]
        current_sphere_count = sphere_growth[pos]
        if current_sphere_count == 0:
            p_t_norms.append(0.0)
            p_t_vectors.append(np.zeros(features.shape[1]))
            continue

        current_spheres = np.array(spheres[:current_sphere_count])
        query_feat = features[pos]

        # cosine 相似度
        sims = np.array([cosine_sim(query_feat, s) for s in current_spheres])

        # 取 top-K
        top_k = min(args.top_k, len(sims))
        top_indices = np.argsort(-sims)[:top_k]
        top_sims = sims[top_indices]

        # 加权聚合: softmax 权重
        weights = np.exp(top_sims * 2)  # sharpen
        weights = weights / (weights.sum() + 1e-10)

        p_t = np.sum(
            [weights[j] * (current_spheres[top_indices[j]] - query_feat)
             for j in range(top_k)],
            axis=0
        )
        p_t_norms.append(float(np.linalg.norm(p_t)))
        p_t_vectors.append(p_t)

    p_t_norms = np.array(p_t_norms)
    print(f"  p_t 范数: 均值={p_t_norms.mean():.4f}, 最大={p_t_norms.max():.4f}, "
          f"非零率={(p_t_norms > 0).mean():.3f}")

    # ── 与 attention 熵的相关性 ──
    # 读原始数据里的 attention 熵 (从 hidden_states 或独立计算)
    # 但我们在 save_features 时没存 attention 熵. 存了 features 矩阵(含熵)
    # 从 features 里提取位置: per-head entropy 是前 14 个 dim
    attn_entropy_per_pos = features[:, :14].mean(axis=1)  # (seq_len,)
    # 第一个 token 的 p_t=0, 跳过
    valid_mask = p_t_norms > 0
    if valid_mask.sum() > 1:
        corr = np.corrcoef(p_t_norms[valid_mask], attn_entropy_per_pos[valid_mask])[0, 1]
    else:
        corr = 0.0
    print(f"  p_t 范数 vs 注意力熵 相关系数: {corr:.4f}")

    # ── 结果 ──
    results = {
        "match_threshold": args.match_threshold,
        "top_k": args.top_k,
        "n_spheres": n_spheres,
        "sphere_growth": sphere_growth,
        "growth_rate_per_step": round(growth_linear, 4),
        "activation_mean": float(np.mean(sphere_activation_count)),
        "activation_max": int(max(sphere_activation_count)),
        "p_t_norm_mean": round(float(p_t_norms.mean()), 4),
        "p_t_norm_max": round(float(p_t_norms.max()), 4),
        "p_t_vs_attn_entropy_corr": round(float(corr), 4),
    }

    out_path = run_dir / "simulation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    # ── 结论 ──
    print(f"\n{'='*50}")
    checks = []
    if n_spheres < seq_len * 0.8:
        checks.append(f"✅ 球体压缩比 {n_spheres}/{seq_len} = {n_spheres/seq_len:.2f} — 有信息压缩")
    else:
        checks.append(f"⚠️ 球体压缩比 {n_spheres}/{seq_len} = {n_spheres/seq_len:.2f} — 几乎无压缩, 匹配阈值可能太低")

    if abs(corr) > 0.2:
        checks.append(f"✅ p_t 与注意力熵相关 ({corr:.4f}) — scan_terrain 有信号")
    else:
        checks.append(f"🟡 p_t 与注意力熵弱相关 ({corr:.4f}) — 信号微弱")

    for c in checks:
        print(f"  {c}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
