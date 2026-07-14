# 离线可行性预演 · 实验计划

> 2026-07-14
> 目标：验证 gravity space 感知过程（45-dim 特征 + 球体创建 + scan_terrain）在真实模型输出上是否有信号

---

## 一、核心问题

| 问题 | 判定标准 | 如果不过的备选方案 |
|------|---------|-------------------|
| ① 45-dim 特征在深层隐藏空间分得出有意义的聚类吗？ | Silhouette Score > 0.25（随机基线约 0.0） | 换信号源：只用注意力分布，或加 logits 分布特征 |
| ② 相邻 token 的场域分布有时序平滑性吗？ | 相邻 token 的余弦相似度均值 > 非相邻 × 1.2 | 「场域每秒跳变」不是不可接受——可能本身就是噪音信号 |
| ③ 一个 session 产生多少球体？增长曲线？ | 纯统计，无通过标准 | 纯信息收集，不决定成败 |
| ④ scan_terrain 的 p_t 与注意力分布相关？ | p_t 范数与 attention 熵的相关系数 > 0.2 | 强信号不需要——即使弱相关，多层聚合也可能有效 |

---

## 二、实验流程

```
Phase A — 数据收集（写脚本跑一次，约 3 分钟）
  加载 Qwen2.5-0.5B
  输入 1 条矛盾文本 prompt（~150 token）
  前向传播，在每层 forward hook 中捕捉：
    · hidden state: (seq_len, 896)
    · attention probs: (14 heads, seq_len, seq_len)
  保存为 .npz（约 50MB）

Phase B — 特征提取（离线，约 1 分钟）
  读 .npz，只取后 8 层（第 16~24 层）
  对每层的每个 token 位置提取：

  语义轴（30 dim）:
    · per-head entropy: 每头的注意力熵 → 14 个标量
    · cross-head variance: 14 heads 在同一位置的注意力方差 → 14 个
    · position center of mass: 注意力分布的位置重心 → 2 个

  情感轴（15 dim）:
    · hidden_norm: ||h|| → 1 个
    · global_attention_entropy: 所有头平均熵 → 1 个
    · layer_norm_velocity: 相邻层范数变化 → ~12 个（24 层跨度的局部）
    · token_cosine: 相邻 token 的 hidden state 余弦 → 1 个

  合并 → seq_len × 45 dim 的特征矩阵

Phase C — 聚类分析（离线，约 30 秒）
  标准化 → PCA 降维到 16 dim（去噪）→ K-means(k=8)
  Silhouette Score + 簇内距离 + 时序标签可视化
  如果 Score > 0.25 → 有意义，继续 Phase D

Phase D — 球体创建 + scan_terrain 模拟（离线，约 30 秒）
  球体创建：每个 token 作为探测点，前面所有 token = 候选球体
  匹配策略：cosine > 0.85 → 更新已有球体；否则创建新球体
  记录球体数量增长曲线
  scan_terrain 模拟：当前 token 与前面所有球体的 cosine 加权聚合 → p_t
  计算 p_t 范数与 attention 熵的相关性
```

---

## 三、Prompt 选型

第一轮跑 2 条：

**A — 冲突证据（实验组）：** 两段矛盾文本，正确答案是「不确定」
```
文本1：小明养了一只猫和一只狗。猫叫咪咪，狗叫旺财。每天早上咪咪会在窗台上等日出。
文本2：小明的宠物中只有狗会在早上叫醒他。猫通常要睡到中午才会活动。
问题：咪咪早上会在哪里？
```

B — 正常对话（对照）：日常问候 + 常识回答
```
Q：告诉我今天的天气怎么样？
A：今天天气晴朗，气温 22~28 度。
```

如果 A 的聚类效果好但 B 不好→特征偏向矛盾检测（也算有结论）
如果 B 好但 A 不好→特征在简单语义上可用但在复杂情况下失效
如果都好→普适特征，最佳情况

---

## 四、通过标准

Phase C 通过 → 继续到 Phase D。不过 → 换特征（去掉情感轴只用语义轴重跑，或换层范围）。

Phase D 中球体数量增长曲线：
- 线性增长（每步+0~1）→ 正常，地形稀疏
- 对数增长（前几步+多，后面几乎不+）→ 地形快速饱和，正常
- 指数增长 → 球体匹配策略有 bug

---

## 五、实验输出文件结构

```
experiments/feasibility/
├── plan.md                            ← 本文件
├── run_001/
│   ├── config.json                    # 实验配置
│   ├── activations.npz                # 原始 hidden state + attention
│   ├── features.npy                   # 45-dim 特征矩阵
│   ├── clustering.json                # K-means 结果 + silhouette
│   └── simulation.json                # 球体创建 + scan_terrain 统计
└── run_002/                           # 后续实验
```

---

## 六、代码清单

预计 3 个文件，约 200 行：

| 文件 | 定位 | 行数 |
|------|------|------|
| `dump_activations.py` | 加载模型、注册 hook、前向传播、存 .npz | ~60 |
| `extract_features.py` | 读 .npz、提取 45-dim 特征、PCA+K-means | ~80 |
| `simulate_gravity.py` | 球体创建模拟 + scan_terrain + 统计 | ~60 |

可以一口气写完，全在 0.5B 上跑，单次前向 + 离线分析 < 5 分钟。
