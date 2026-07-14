"""
Gravity Reranker v0.2 实验
验证核心命题：多头注意力中是否包含可提取的证据多样性信息。

方法：
  - 单次前向传播
  - 提取各 attention head 的注意力分布
  - 计算跨头证据多样性（不同 head 是否在看不同的输入 token）
  - 用多样性加权调整 logits
  - 对比标准选择 vs 调整后选择

用法：
    python run_reranker_v0.2.py          # 跑全量
    python run_reranker_v0.2.py --dry-run  # 只看配置
    python run_reranker_v0.2.py --probe   # 先跑一条验证注意力提取
"""

import json, os, sys, time, math, re, argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset_ctr_v2 import get_dataset


# ========== 配置 ==========

@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    max_new_tokens: int = 128
    temperature: float = 0.7
    random_seed: int = 42
    alpha: float = 0.3       # 多样性权重
    run_id: str = ""
    output_dir: str = ""

    def auto_paths(self):
        base = Path(__file__).parent / "data"
        base.mkdir(parents=True, exist_ok=True)
        existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
        n = len(existing) + 1
        self.run_id = f"run_{n:03d}"
        self.output_dir = str(base / self.run_id)


# ========== 实验引擎 ==========

class GravityRerankerV2:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results = []

    def setup(self):
        """加载模型"""
        cfg = self.config
        print(f"[setup] 加载模型: {cfg.model_name}")

        # 修复 SSL（Windows）
        import httpx, ssl
        from huggingface_hub.utils import _http
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _http.set_client_factory(lambda: httpx.Client(verify=ctx))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if cfg.dtype == "float16" else torch.float32,
            device_map="auto" if cfg.device == "cuda" else None,
            attn_implementation="eager",  # 必须用 eager attention 才能提取注意力分布
        )
        self.model.eval()

        n_params = self.model.num_parameters() / 1e6
        n_layers = self.model.config.num_hidden_layers
        n_heads = self.model.config.num_attention_heads
        print(f"[setup] 参数: {n_params:.1f}M | 层: {n_layers} | 每层头: {n_heads}")
        print(f"[setup] 总 attention heads: {n_layers * n_heads}")

        torch.manual_seed(cfg.random_seed)

    def _generate_with_attention(self, input_ids, max_new_tokens: int, temperature: float = 0.7):
        """
        自定义生成循环，每步记录注意力和 logits。
        
        因为 transformers 的 generate() 不支持 output_attentions，
        这里手动实现一个简化版生成。
        """
        device = self.model.device
        batch_size = input_ids.shape[0]
        all_attentions = []  # 每步：各层的注意力
        all_logits = []      # 每步的 logits
        all_token_ids = []
        
        current_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    current_ids,
                    output_attentions=True,
                    use_cache=True,
                )
            
            # 提取最后一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            
            # 采样（temperature）
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # 记录
            all_attentions.append(outputs.attentions)  # tuple of (layer × (batch, heads, q_len, kv_len))
            all_logits.append(next_token_logits)
            all_token_ids.append(next_token.item())
            
            # 拼接
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # 如果生成了 eos，停止
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return current_ids, all_logits, all_attentions, all_token_ids

    def _extract_attention_per_step(self, all_attentions, input_length: int) -> list:
        """
        从自定义生成循环中提取每步的注意力分布。
        
        all_attentions: list of tuple[torch.Tensor]
          外层 = 生成的每一步
          内层 tuple = 各层的注意力 (batch, heads, q_len, kv_len)
        """
        step_attentions = []
        
        for step_idx, layer_attentions in enumerate(all_attentions):
            step_data = {"step": step_idx}
            all_head_attentions = []

            for layer_idx, attn in enumerate(layer_attentions):
                # attn: (1, num_heads, q_len, kv_len)
                attn = attn.squeeze(0)  # (heads, q_len, kv_len)
                # 取最后一个 query position（当前生成的 token）
                current_attn = attn[:, -1, :]  # (heads, kv_len)

                for head_idx in range(attn.shape[0]):
                    head_attn = current_attn[head_idx]  # (kv_len,)
                    tk = head_attn.topk(min(3, len(head_attn)))
                    all_head_attentions.append({
                        "layer": layer_idx,
                        "head": head_idx,
                        "top3_positions": tk.indices.cpu().tolist(),
                        "top3_values": tk.values.cpu().tolist(),
                        "entropy": self._compute_entropy(head_attn),
                    })

            step_data["kv_length"] = layer_attentions[0].shape[-1]
            step_data["heads"] = all_head_attentions
            step_attentions.append(step_data)

        return step_attentions

    def _compute_entropy(self, attn: torch.Tensor) -> float:
        """计算注意力分布的熵（高熵 = 注意力分散）"""
        p = attn.clamp(min=1e-10)
        return -float((p * p.log()).sum())

    def _compute_diversity(self, step_data: dict) -> dict:
        """
        计算一步的跨头证据多样性。
        
        核心思路：
          336 个 head 各自关注不同的输入 token。
          如果 head 们集中在同一个输入 token 上 → 低多样性
          如果 head 们分散在不同输入 token 上 → 高多样性
        
        度量:
          - 所有 head 的 top-1 输入位置的集合大小
          - 归一化: unique_top1_positions / num_heads
        """
        heads = step_data["heads"]
        n_heads = len(heads)

        # 收集每个 head 的 top-1 输入位置
        top1_positions = [h["top3_positions"][0] for h in heads]
        unique_positions = len(set(top1_positions))
        
        # top-3 位置的整体覆盖度
        all_top3 = []
        for h in heads:
            all_top3.extend(h["top3_positions"])
        unique_top3 = len(set(all_top3))

        # 熵聚合：所有 head 注意力的平均熵
        avg_head_entropy = float(np.mean([h["entropy"] for h in heads]))

        diversity = {
            "top1_unique_positions": unique_positions,
            "top3_unique_positions": unique_top3,
            "top1_diversity_ratio": unique_positions / n_heads if n_heads > 0 else 0,
            "avg_head_entropy": avg_head_entropy,
            "n_heads": n_heads,
        }
        return diversity

    def _apply_diversity_to_logits(
        self, logits: torch.Tensor, diversity: dict, generated_token_id: int
    ) -> dict:
        """
        用多样性对 logits 做二次分配。
        
        策略：
          如果多样性高 → 当前 token 的选择有多个证据源支持 → 加分
          如果多样性低 → 当前 token 的选择依赖单一证据 → 扣分
        
        diversity_ratio 高 = 注意力头分散在不同输入 → 加分
        diversity_ratio 低 = 注意力头集中在一处 → 扣分（可能过拟合单一证据）
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 当前被选中的 token 的 log-prob
        selected_lp = log_probs[generated_token_id].item()
        
        # 多样性调整
        d_ratio = diversity["top1_diversity_ratio"]
        adjusted_lp = selected_lp * (1 + self.config.alpha * (d_ratio - 0.5))
        
        # 检查 top-5 token 排序的改变
        top5_original = log_probs.topk(5)
        original_ranking = top5_original.indices.cpu().tolist()
        
        # 对 logits 应用多样性调整
        adjusted_log_probs = log_probs.clone()
        bias = self.config.alpha * (d_ratio - 0.5)
        adjusted_log_probs = adjusted_log_probs * (1 + bias)
        adjusted_log_probs = adjusted_log_probs - adjusted_log_probs.logsumexp(dim=-1)
        
        top5_adjusted = adjusted_log_probs.topk(5)
        adjusted_ranking = top5_adjusted.indices.cpu().tolist()
        
        # 检查 top-1 是否改变
        top1_changed = original_ranking[0] != adjusted_ranking[0]
        
        return {
            "selected_token_id": int(generated_token_id),
            "selected_token_text": self.tokenizer.decode([generated_token_id]),
            "original_log_prob": round(selected_lp, 4),
            "adjusted_log_prob": round(adjusted_lp, 4),
            "diversity_ratio": round(d_ratio, 4),
            "adjustment_bias": round(bias, 4),
            "original_top5": [self.tokenizer.decode([t]) for t in original_ranking],
            "adjusted_top5": [self.tokenizer.decode([t]) for t in adjusted_ranking],
            "top1_changed": top1_changed,
            "top1_original": self.tokenizer.decode([original_ranking[0]]),
            "top1_adjusted": self.tokenizer.decode([adjusted_ranking[0]]),
        }

    def _judge_open_response(self, response: str, item: dict) -> dict:
        """评判开放题的回答"""
        response_lower = response.lower()
        
        if item["judge"] == "uncertainty":
            # 检查是否表达了不确定性
            expressed_uncertainty = any(
                kw.lower() in response_lower for kw in item["correct_keywords"]
            )
            return {
                "correct": expressed_uncertainty,
                "judge_method": "uncertainty",
                "detail": f"不确定性表达: {expressed_uncertainty}"
            }
        
        elif item["judge"] == "keyword":
            # 检查关键答案词是否出现
            found_keywords = [
                kw for kw in item["correct_keywords"]
                if kw.lower() in response_lower
            ]
            return {
                "correct": len(found_keywords) > 0,
                "judge_method": "keyword",
                "detail": f"命中关键词: {found_keywords}"
            }
        
        return {"correct": False, "judge_method": "unknown", "detail": ""}

    def _judge_choice_response(self, response: str, item: dict) -> dict:
        """评判选择题的回答"""
        # 找 A/B/C
        match = re.search(r'\b([ABC])\b', response.upper())
        if match:
            chosen = match.group(1)
            return {
                "correct": chosen == item["correct_answer"],
                "judge_method": "choice",
                "detail": f"选择了 {chosen}, 正确答案 {item['correct_answer']}"
            }
        return {"correct": False, "judge_method": "choice", "detail": "未提取到答案"}

    def run_probe(self):
        """原型验证：跑一条最简单的矛盾题，确认注意力数据能正确提取"""
        print("\n" + "=" * 60)
        print("原型验证：蓝/绿天空")
        print("=" * 60)

        prompt = """文本1：天是蓝色的。
文本2：天是绿色的。

问题：天是什么颜色的？
请分析以上两段文本的矛盾之处，然后回答。"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_length = inputs.input_ids.shape[1]

        # 使用自定义生成循环
        all_ids, all_logits, all_attentions, all_tokens = self._generate_with_attention(
            inputs.input_ids, max_new_tokens=64, temperature=0.0
        )

        generated_text = self.tokenizer.decode(all_ids[0][input_length:], skip_special_tokens=True)
        print(f"\n生成(前200字符): {generated_text[:200]}\n")

        # 提取注意力
        step_attentions = self._extract_attention_per_step(all_attentions, input_length)
        print(f"共 {len(step_attentions)} 步\n")

        # 分析前 5 步的注意力
        for step in step_attentions[:5]:
            diversity = self._compute_diversity(step)
            top1s = [h["top3_positions"][0] for h in step["heads"]]
            top1_counter = Counter(top1s).most_common(5)

            # 将位置映射回 token 文本
            input_tokens = self.tokenizer.convert_ids_to_tokens(
                inputs.input_ids[0].tolist()
            )

            print(f"  Step {step['step']}:")
            print(f"    多样性比率: {diversity['top1_diversity_ratio']:.3f}")
            print(f"    唯一 Top1 位置: {diversity['top1_unique_positions']}")
            print(f"    注意力头最集中的 5 个输入 token:")
            for pos, count in top1_counter:
                if pos < len(all_tokens):
                    print(f"      [pos {pos}]: {count} heads")
                else:
                    print(f"      [{pos}] (generated): {count} heads")
            print()

        print("\n原型验证完成。确认注意力数据提取正常后，进入全量实验。")
        return step_attentions

    def run_trial(self, item: dict, verbose: bool = False) -> dict:
        """对一条测试数据运行实验"""
        # 构建 prompt
        prompt = self._build_prompt(item)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_length = inputs.input_ids.shape[1]

        # 使用自定义生成循环（捕获注意力分布）
        all_ids, all_logits, all_attentions, all_tokens = self._generate_with_attention(
            inputs.input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        generated_text = self.tokenizer.decode(all_ids[0][input_length:], skip_special_tokens=True)

        # 提取注意力
        step_attentions = self._extract_attention_per_step(all_attentions, input_length)

        # 对每一步计算多样性 + logits 调整
        token_adjustments = []
        for step_idx, step_data in enumerate(step_attentions):
            if step_idx < len(all_logits):
                step_logits = all_logits[step_idx].squeeze(0)  # (1, vocab) -> (vocab,)
                generated_token_id = all_tokens[step_idx]
            else:
                continue

            diversity = self._compute_diversity(step_data)
            adjustment = self._apply_diversity_to_logits(
                step_logits, diversity, generated_token_id
            )
            token_adjustments.append(adjustment)

        # 评判回答
        if item["eval_type"] == "choice":
            judgement = self._judge_choice_response(generated_text, item)
        else:
            judgement = self._judge_open_response(generated_text, item)

        # 统计多样性对 token 选择的影响
        changes = [a for a in token_adjustments if a["top1_changed"]]
        changes_toward_correct = 0
        changes_away_correct = 0
        for c in changes:
            # 简化版：只看调整方向（详细分析在数据中）
            pass

        result = {
            "trial_id": item["id"],
            "eval_type": item["eval_type"],
            "correct_answer": item.get("correct_answer", ""),
            "generated_text": generated_text,
            "judgement": judgement,
            "tokens_analyzed": len(token_adjustments),
            "tokens_changed": len(changes),
            "change_ratio": len(changes) / len(token_adjustments) if token_adjustments else 0,
            "avg_diversity_ratio": float(np.mean([a["diversity_ratio"] for a in token_adjustments])),
            "diversity_range": [
                float(np.min([a["diversity_ratio"] for a in token_adjustments])),
                float(np.max([a["diversity_ratio"] for a in token_adjustments])),
            ],
            "token_adjustments": token_adjustments,
        }
        return result

    def _build_prompt(self, item: dict) -> str:
        """构建 prompt"""
        if item["eval_type"] == "choice":
            return f"""文本1：{item["text1"]}

文本2：{item["text2"]}

问题：{item["question"]}

选项：
{item["options"]}

请分析以上两段文本，回答这个问题。直接说出你的答案（A、B 或 C），然后简要解释推理过程。"""
        else:
            return f"""文本1：{item["text1"]}

文本2：{item["text2"]}

问题：{item["question"]}

请分析以上两段文本，然后回答。"""

    def run(self):
        """运行完整实验"""
        cfg = self.config
        cfg.auto_paths()
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Gravity Reranker v0.2")
        print(f"Run ID: {cfg.run_id}")
        print(f"模型: {cfg.model_name}")
        print(f"Alpha: {cfg.alpha}")
        print(f"输出: {output_dir}")
        print(f"{'='*60}\n")

        # 保存配置
        config_dict = asdict(cfg)
        config_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # 加载模型
        self.setup()

        # 原型验证
        print("\n>>> 原型验证 <<<")
        self.run_probe()
        print("\n>>> 进入全量实验 <<<\n")

        # 加载数据集
        dataset = get_dataset()
        print(f"数据集: {len(dataset)} 条\n")

        all_results = []
        correct_standard = 0  # 标准生成的结果（judgement.correct）
        total_open = 0
        total_choice = 0
        total_tokens_changed = 0
        total_tokens = 0

        for i, item in enumerate(dataset):
            print(f"[{i+1}/{len(dataset)}] {item['id']}... ", end="", flush=True)
            t0 = time.time()

            result = self.run_trial(item)
            all_results.append(result)

            if result["judgement"]["correct"]:
                if item["eval_type"] == "choice":
                    correct_standard += 1
                total_open += 1 if item["eval_type"] == "free" else 0
            else:
                total_choice += 1 if item["eval_type"] == "choice" else 0

            total_tokens_changed += result["tokens_changed"]
            total_tokens += result["tokens_analyzed"]

            elapsed = time.time() - t0
            ok = "OK" if result["judgement"]["correct"] else "XX"
            print(f"{elapsed:.1f}s [{ok}] (改变: {result['tokens_changed']}/{result['tokens_analyzed']} tokens)")

            # 每 5 条保存一次中间结果
            if (i + 1) % 5 == 0:
                self._save_intermediate(all_results, output_dir)

        # 保存最终结果
        self._save_final(all_results, output_dir)
        return all_results

    def _save_intermediate(self, results: list, output_dir: Path):
        """保存中间结果（精简版，不含完整注意力数据）"""
        slim = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "token_adjustments"}
            s["change_summary"] = [
                {"step": i, "diversity": a["diversity_ratio"],
                 "top1_changed": a["top1_changed"],
                 "original": a["top1_original"], "adjusted": a["top1_adjusted"]}
                for i, a in enumerate(r["token_adjustments"])
            ]
            slim.append(s)
        with open(output_dir / "results_intermediate.jsonl", "w", encoding="utf-8") as f:
            for s in slim:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def _save_final(self, results: list, output_dir: Path):
        """保存最终结果"""
        # 完整结果（注意力数据过大，保存精简版）
        with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                # 只保留 token 调整摘要，不保留完整注意力矩阵
                record = {k: v for k, v in r.items() if k != "token_adjustments"}
                record["change_summary"] = [
                    {"step": i, "token": a["selected_token_text"],
                     "original_lp": a["original_log_prob"],
                     "adjusted_lp": a["adjusted_log_prob"],
                     "diversity": a["diversity_ratio"],
                     "top1_changed": a["top1_changed"],
                     "original": a["top1_original"],
                     "adjusted": a["top1_adjusted"]}
                    for i, a in enumerate(r["token_adjustments"])
                ]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 计算聚合统计
        choice_results = [r for r in results if r["eval_type"] == "choice"]
        free_results = [r for r in results if r["eval_type"] == "free"]
        uncertainty_results = [r for r in results 
                               if r["eval_type"] == "free" 
                               and r["judgement"]["judge_method"] == "uncertainty"]

        summary = {
            "run_id": self.config.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": asdict(self.config),
            "total_samples": len(results),
            "choice_accuracy": sum(1 for r in choice_results if r["judgement"]["correct"]) / len(choice_results) if choice_results else 0,
            "free_accuracy": sum(1 for r in free_results if r["judgement"]["correct"]) / len(free_results) if free_results else 0,
            "uncertainty_accuracy": sum(1 for r in uncertainty_results if r["judgement"]["correct"]) / len(uncertainty_results) if uncertainty_results else 0,
            "avg_diversity_ratio": float(np.mean([r["avg_diversity_ratio"] for r in results])),
            "avg_token_change_ratio": float(np.mean([r["change_ratio"] for r in results])),
            "tokens_changed_total": sum(r["tokens_changed"] for r in results),
            "tokens_analyzed_total": sum(r["tokens_analyzed"] for r in results),
            "per_sample": [
                {
                    "id": r["trial_id"],
                    "eval_type": r["eval_type"],
                    "correct": r["judgement"]["correct"],
                    "detail": r["judgement"]["detail"],
                    "token_change_ratio": r["change_ratio"],
                    "avg_diversity": r["avg_diversity_ratio"],
                    "diversity_range": r["diversity_range"],
                }
                for r in results
            ]
        }

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印结果
        print(f"\n{'='*60}")
        print(f"实验结果: {self.config.run_id}")
        print(f"{'='*60}")
        print(f"选择题准确率: {summary['choice_accuracy']:.1%}")
        print(f"开放题准确率: {summary['free_accuracy']:.1%}")
        print(f"不确定题准确率: {summary['uncertainty_accuracy']:.1%}")
        print(f"\n注意力多样性分析:")
        print(f"  平均多样性比率: {summary['avg_diversity_ratio']:.3f}")
        print(f"  高频多样性范围: 查看 per_sample\n")
        print(f"Token 选择改变:")
        print(f"  平均改变率: {summary['avg_token_change_ratio']:.1%}")
        print(f"  总改变 token 数: {summary['tokens_changed_total']}/{summary['tokens_analyzed_total']}")
        print(f"\n详细结果: {output_dir / 'summary.json'}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Gravity Reranker v0.2")
    parser.add_argument("--dry-run", action="store_true", help="只显示配置")
    parser.add_argument("--probe", action="store_true", help="只跑原型验证")
    parser.add_argument("--alpha", type=float, default=0.3, help="多样性权重")
    args = parser.parse_args()

    config = ExperimentConfig(alpha=args.alpha)

    if args.dry_run:
        print("=== Dry Run ===")
        print(json.dumps(asdict(config), indent=2))
        ds = get_dataset()
        print(f"\n数据集: {len(ds)} 条")
        print(f"  选择题: {sum(1 for d in ds if d['eval_type']=='choice')}")
        print(f"  开放题: {sum(1 for d in ds if d['eval_type']=='free')}")
        return

    exp = GravityRerankerV2(config)

    if args.probe:
        exp.setup()
        exp.run_probe()
        return

    exp.run()


if __name__ == "__main__":
    main()
