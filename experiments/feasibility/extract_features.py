"""
Phase B + C: 特征提取 + 聚类分析
读 .npz -> 45-dim 特征提取 -> PCA -> K-means -> Silhouette Score

用法: python extract_features.py runs/ctr_001_xxxx/activations.npz
"""

import json, sys, argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def entropy(p, eps=1e-10):
    """单分布熵"""
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


def extract_features(hidden_states, attentions):
    """
    hidden_states: (n_layers, seq_len, hidden_dim) — 所有层
    attentions:   (n_layers, n_heads, seq_len, seq_len) — 所有层

    返回: (seq_len, 45) 特征矩阵
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    n_heads = attentions.shape[1]

    # 只看后 1/3 层
    deep_start = n_layers * 2 // 3  # 第 16 层（0-indexed）
    deep_hs = hidden_states[deep_start:]        # (deep_layers, seq, hidden)
    deep_attn = attentions[deep_start:]         # (deep_layers, heads, seq, seq)
    n_deep = len(deep_hs)

    features = []

    for pos in range(seq_len):
        feat = []

        # ── 语义轴: per-head entropy (14) ──
        # 对所有深层取平均
        pos_attentions = deep_attn[:, :, pos, :]  # (deep_layers, heads, seq)
        head_entropies = np.array([
            [entropy(pos_attentions[l, h]) for h in range(n_heads)]
            for l in range(n_deep)
        ])  # (deep_layers, heads)
        feat.extend(head_entropies.mean(axis=0).tolist())  # 14

        # ── 语义轴: cross-head variance (14) ──
        # 在每个输入 token 上, 14 个 head 给它的注意力权重的方差, 然后对输入维度平均
        for h in range(n_heads):
            h_weights = deep_attn[:, h, pos, :]  # (deep_layers, seq)
            cross_var = h_weights.var(axis=1).mean()  # 对 deep_layers 平均, 再对 seq 平均
            feat.append(float(cross_var))
        # 14

        # ── 语义轴: position center of mass (2) ──
        # 注意力分布的重心位置 (在输入序列的前半还是后半)
        positions = np.arange(seq_len)
        for l in range(n_deep):
            center = np.average(positions, weights=deep_attn[l, :, pos, :].mean(axis=0) + 1e-10)
            # 只看最后 3 个 deep layer 的平均, 不是所有 deep layer
            if l >= n_deep - 3:
                if l == n_deep - 3:
                    com_sum = 0.0
                    count = 0
                com_sum += center
                count += 1
        feat.append(com_sum / count / seq_len)  # 归一化到 0~1
        # 再存一个 "偏移量": 重心 vs 序列中点的距离
        feat.append(abs(com_sum / count / seq_len - 0.5))

        # ── 情感轴: hidden_norm (1) ──
        # 深层各层范数, 对 deep layers 平均
        norms = [np.linalg.norm(deep_hs[l, pos]) for l in range(n_deep)]
        feat.append(float(np.mean(norms)))

        # ── 情感轴: global attention entropy (1) ──
        # 所有深层所有 head 的平均注意力熵
        feat.append(float(head_entropies.mean()))

        # ── 情感轴: layer_norm_velocity (12) ──
        # 相邻层 norm 变化: 对 24 层全量计算
        full_norms = np.array([np.linalg.norm(hidden_states[l, pos]) for l in range(n_layers)])
        if n_layers > 1:
            velocity = np.abs(np.diff(full_norms))  # (n_layers-1,)
            # 只取最后 12 个 (deep layer 间变化)
            feat.extend(velocity[-12:].tolist())
        else:
            feat.extend([0.0] * 12)

        # ── 情感轴: token_cosine (1) ──
        if pos > 0:
            prev_h = deep_hs[:, pos - 1, :].reshape(-1)[:hidden_dim]  # flatten deep layers
            curr_h = deep_hs[:, pos, :].reshape(-1)[:hidden_dim]
            cos_sim = np.dot(prev_h, curr_h) / (np.linalg.norm(prev_h) * np.linalg.norm(curr_h) + 1e-10)
        else:
            cos_sim = 0.0
        feat.append(float(cos_sim))

        features.append(feat)

    return np.array(features)  # (seq_len, 45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Path to activations.npz")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=16)
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    run_dir = npz_path.parent

    print(f"加载: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    seq_len = data["input_ids"].shape[1]

    # 提取层数和维度
    n_layers = sum(1 for k in data.keys() if k.startswith("hidden_"))
    hidden_dim = data["hidden_0"].shape[-1]
    attn_first = data["attn_0"]
    n_heads = attn_first.shape[0]
    print(f"  序列长度: {seq_len} | 层数: {n_layers} | heads: {n_heads} | hidden: {hidden_dim}")

    # 组装 hidden_states
    hidden_states = np.stack([data[f"hidden_{i}"] for i in range(n_layers)])  # (n_layers, seq, hidden)
    attentions = np.stack([data[f"attn_{i}"] for i in range(n_layers)])       # (n_layers, heads, seq, seq)

    # ── Phase B: 特征提取 ──
    print(f"\n[Phase B] 提取 45-dim 特征 (只看后 1/3 层, 第 {n_layers*2//3}~{n_layers-1} 层)...")
    features = extract_features(hidden_states, attentions)
    print(f"  特征矩阵: {features.shape}")

    # ── Phase C: 聚类 ──
    print(f"\n[Phase C] PCA({args.pca_dim}) + K-means(k={args.n_clusters})...")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=min(args.pca_dim, seq_len - 1))
    features_pca = pca.fit_transform(features_scaled)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA 解释方差: {explained_var:.3f}")

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_pca)

    sil = silhouette_score(features_pca, labels)
    print(f"  Silhouette Score: {sil:.4f}")
    print(f"  簇分布: {np.bincount(labels)}")

    # ── 时序一致性 ──
    adjacent_same = sum(1 for i in range(1, seq_len) if labels[i] == labels[i - 1])
    temporal_coherence = adjacent_same / (seq_len - 1)
    print(f"  时序一致性 (相邻 token 同簇率): {temporal_coherence:.3f}")

    # ── 结果 ──
    results = {
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden_dim": hidden_dim,
        "pca_dim": args.pca_dim,
        "pca_explained_var": round(float(explained_var), 4),
        "n_clusters": args.n_clusters,
        "silhouette_score": round(float(sil), 4),
        "temporal_coherence": round(float(temporal_coherence), 4),
        "cluster_distribution": np.bincount(labels).tolist(),
        "cluster_labels": labels.tolist(),
    }

    out_path = run_dir / "features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    # 也存特征矩阵本身, 供 Phase D 使用
    npz_out = run_dir / "features.npz"
    np.savez_compressed(npz_out, features=features, features_pca=features_pca, labels=labels)
    print(f"特征矩阵已保存: {npz_out}")

    # ── 结论 ──
    print(f"\n{'='*50}")
    if sil > 0.25:
        print(f"结论: ✅ Silhouette={sil:.4f} > 0.25 — 特征空间有结构, 可以进入 Phase D")
    elif sil > 0.1:
        print(f"结论: 🟡 Silhouette={sil:.4f} — 弱结构, 需要检查特征或换层范围")
    else:
        print(f"结论: ❌ Silhouette={sil:.4f} — 特征在 0.5B 上无分辨度")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
