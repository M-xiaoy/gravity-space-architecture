"""
Gravity Reranker v0.3 — Logits 分布熵分析
验证目标：矛盾文本推理任务中，token 选择时的 logits 分布是否包含多样性信息。

核心假设同 v0.2，但信号源不同：
  v0.2: attention head 跨头注意力位置多样性 → 无效（0.5B 不分头）
  v0.3: 直接看 softmax 输出的分布熵 — 平坦分布 = 多路径并存 = 多样性

度量方法：
  对每个 token 生成步，计算：
    1. 分布熵 H(p) = -Σ p_i log p_i（全词表）
    2. top-5 概率集中度
    3. 正确答案 token 在该步的概率
    4. p(top-1) - p(top-2)：确定性率
  
  关键决策点：模型在"选哪个答案方向"时的分布形态
"""

import json, os, sys, time, math, re, argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset_ctr_v2 import get_dataset


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    max_new_tokens: int = 128
    temperature: float = 0.7
    random_seed: int = 42
    run_id: str = ""
    output_dir: str = ""
    
    def auto_paths(self):
        base = Path(__file__).parent / "data"
        base.mkdir(parents=True, exist_ok=True)
        existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
        n = len(existing) + 1
        self.run_id = f"run_{n:03d}"
        self.output_dir = str(base / self.run_id)


class LogitsEntropyAnalysis:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.vocab_size = None
        self.results = []

    def setup(self):
        cfg = self.config
        print(f"[setup] 加载模型: {cfg.model_name}")

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
        )
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size

        n_params = self.model.num_parameters() / 1e6
        print(f"[setup] 参数: {n_params:.1f}M | Vocab: {self.vocab_size}")
        torch.manual_seed(cfg.random_seed)

    def _generate_and_capture(self, input_ids, max_new_tokens: int, temperature: float = 0.7):
        """自回归生成，每步记录 logits 的分布特征"""
        device = self.model.device
        current_ids = input_ids.clone()
        step_data = []  # 每步一条

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=True)
            
            next_token_logits = outputs.logits[:, -1, :]         # (1, V)
            
            # ── 提取分布指标（使用原始 softmax，temperature 仅用于采样）──
            logits = next_token_logits.squeeze(0)                 # (V,)
            raw_probs = torch.softmax(logits, dim=-1)              # (V,) 原始分布
            
            # 采样用 temperature（用 unsqueezed logits 保持 2D）
            sampling_logits = next_token_logits.squeeze(0)  # (V,)
            if temperature > 0:
                sampled_probs = torch.softmax(sampling_logits / temperature, dim=-1)
                next_token = torch.multinomial(sampled_probs, num_samples=1)  # (1,)
            else:
                next_token = sampling_logits.argmax(dim=-1, keepdim=False)
            next_token = next_token.unsqueeze(0)  # (1, 1) 匹配 batch 维度

            # 1. 熵（只看 top-100，避 float16+151k 词表的数值问题）
            #    用原始分布 raw_probs，不是 tempered 后的
            topk_100 = raw_probs.topk(100)
            top100_p = topk_100.values.clamp(min=1e-10)
            entropy = -float((top100_p * top100_p.log()).sum().item())
            
            # 2. top-10 概率（用原始分布）
            topk_probs, topk_indices = raw_probs.topk(10)
            topk_probs = topk_probs.cpu().tolist()
            
            # 3. top-1 和 top-2 的差距
            p1 = topk_probs[0]
            p2 = topk_probs[1] if len(topk_probs) > 1 else 0.0
            certainty_gap = p1 - p2

            # 4. 局部熵（只看 top-20，细粒度）
            top20_p = raw_probs.topk(20).values.clamp(min=1e-10)
            top20_entropy = -float((top20_p * top20_p.log()).sum().item())

            # 5. top-1 概率占比
            p1_ratio = p1
            
            # 采样结果
            sampled_id = next_token.item()
            sampled_token = self.tokenizer.decode([sampled_id])

            record = {
                "step": step,
                "sampled_id": sampled_id,
                "sampled_token": sampled_token,
                "entropy_full": round(entropy, 4),
                "entropy_top20": round(top20_entropy, 4),
                "p1": round(p1, 6),
                "p2": round(p2, 6),
                "certainty_gap": round(certainty_gap, 6),
                "p1_ratio": round(p1_ratio, 6),
                "top10_probs": [round(p, 6) for p in topk_probs],
                "top10_tokens": [self.tokenizer.decode([idx.item()]) for idx in topk_indices],
            }
            step_data.append(record)

            # 拼接
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return current_ids, step_data

    def _judge_open_response(self, response: str, item: dict) -> dict:
        response_lower = response.lower()
        if item["judge"] == "uncertainty":
            expressed = any(kw.lower() in response_lower for kw in item["correct_keywords"])
            return {"correct": expressed, "method": "uncertainty"}
        elif item["judge"] == "keyword":
            found = [kw for kw in item["correct_keywords"] if kw.lower() in response_lower]
            return {"correct": len(found) > 0, "method": "keyword", "found": found}
        return {"correct": False, "method": "unknown"}

    def _judge_choice(self, response: str, item: dict) -> dict:
        match = re.search(r'\b([ABC])\b', response.upper())
        if match:
            chosen = match.group(1)
            return {"correct": chosen == item["correct_answer"], "method": "choice", "chosen": chosen}
        return {"correct": False, "method": "choice", "chosen": None}

    def _build_prompt(self, item: dict) -> str:
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

    def run_trial(self, item: dict, verbose: bool = False) -> dict:
        prompt = self._build_prompt(item)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_length = inputs.input_ids.shape[1]

        all_ids, step_data = self._generate_and_capture(
            inputs.input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        generated_text = self.tokenizer.decode(all_ids[0][input_length:], skip_special_tokens=True)

        if item["eval_type"] == "choice":
            judgement = self._judge_choice(generated_text, item)
        else:
            judgement = self._judge_open_response(generated_text, item)

        # 提取关键决策步的特征
        decision_entropy = {
            "avg_entropy": round(float(np.mean([s["entropy_full"] for s in step_data])), 4),
            "max_entropy": round(float(np.max([s["entropy_full"] for s in step_data])), 4),
            "min_entropy": round(float(np.min([s["entropy_full"] for s in step_data])), 4),
            "avg_certainty_gap": round(float(np.mean([s["certainty_gap"] for s in step_data])), 6),
            "avg_p1_ratio": round(float(np.mean([s["p1_ratio"] for s in step_data])), 6),
            "low_confidence_steps": sum(1 for s in step_data if s["p1"] < 0.5),
            "high_entropy_steps": sum(1 for s in step_data if s["entropy_full"] > 6.0),
        }

        return {
            "trial_id": item["id"],
            "eval_type": item["eval_type"],
            "correct_answer": item.get("correct_answer", ""),
            "generated_text": generated_text,
            "judgement": judgement,
            "tokens_analyzed": len(step_data),
            "entropy_summary": decision_entropy,
            "step_data": step_data,  # 每步的分布特征
        }

    def run(self):
        cfg = self.config
        cfg.auto_paths()
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Logits 分布熵分析 v0.3")
        print(f"Run ID: {cfg.run_id}")
        print(f"模型: {cfg.model_name}")
        print(f"输出: {output_dir}")
        print(f"{'='*60}\n")

        config_dict = asdict(cfg)
        config_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        self.setup()

        dataset = get_dataset()
        print(f"数据集: {len(dataset)} 条\n")

        all_results = []
        correct_count = 0
        all_entropies = []
        all_certainty_gaps = []
        low_confidence_total = 0
        high_entropy_total = 0

        for i, item in enumerate(dataset):
            print(f"[{i+1}/{len(dataset)}] {item['id']}... ", end="", flush=True)
            t0 = time.time()

            result = self.run_trial(item)
            all_results.append(result)

            if result["judgement"]["correct"]:
                correct_count += 1

            es = result["entropy_summary"]
            all_entropies.append(es["avg_entropy"])
            all_certainty_gaps.append(es["avg_certainty_gap"])
            low_confidence_total += es["low_confidence_steps"]
            high_entropy_total += es["high_entropy_steps"]

            elapsed = time.time() - t0
            ok = "OK" if result["judgement"]["correct"] else "XX"
            print(f"{elapsed:.1f}s [{ok}] 平均熵={es['avg_entropy']:.2f}")

            if (i + 1) % 5 == 0:
                self._save_intermediate(all_results, output_dir)

        self._save_final(all_results, output_dir)

        print(f"\n{'='*60}")
        print(f"结果汇总: {cfg.run_id}")
        print(f"{'='*60}")
        print(f"准确率: {correct_count}/{len(dataset)} = {correct_count/len(dataset):.1%}")
        print(f"\n分布熵（全步平均，跨样本平均）:")
        print(f"  平均熵: {np.mean(all_entropies):.3f}")
        print(f"  最高样本平均熵: {max(all_entropies):.3f}")
        print(f"  最低样本平均熵: {min(all_entropies):.3f}")
        print(f"\n置信度:")
        print(f"  平均 certainty_gap (p1-p2): {np.mean(all_certainty_gaps):.6f}")
        print(f"  低置信步 (p1<0.5): {low_confidence_total}")
        print(f"  高熵步 (H>6.0): {high_entropy_total}")
        print(f"\n详细结果: {output_dir}")

    def _save_intermediate(self, results: list, output_dir: Path):
        slim = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "step_data"}
            s["entropy_trace"] = [
                {"step": sd["step"], "token": sd["sampled_token"],
                 "entropy": sd["entropy_full"],
                 "p1": sd["p1"], "p2": sd["p2"]}
                for sd in r["step_data"]
            ]
            slim.append(s)
        with open(output_dir / "results_intermediate.jsonl", "w", encoding="utf-8") as f:
            for s in slim:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def _save_final(self, results: list, output_dir: Path):
        # 完整结果（含每步分布摘要，不含全 logits 向量）
        with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                record = {k: v for k, v in r.items() if k != "step_data"}
                record["entropy_trace"] = [
                    {k: v for k, v in sd.items() if k in ("step", "sampled_token",
                     "entropy_full", "entropy_top20", "p1", "p2", "certainty_gap",
                     "p1_ratio")}
                    for sd in r["step_data"]
                ]
                # 关键决策步（高熵步）详细记录
                record["high_entropy_steps"] = [
                    {k: v for k, v in sd.items() if k in ("step", "sampled_token",
                     "entropy_full", "p1", "p2", "certainty_gap", "top10_tokens",
                     "top10_probs")}
                    for sd in r["step_data"]
                    if sd["entropy_full"] > 6.0
                ]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 聚合统计
        choice_results = [r for r in results if r["eval_type"] == "choice"]
        free_correct = [r for r in results if r["eval_type"] == "free" and r["judgement"]["correct"]]
        free_wrong = [r for r in results if r["eval_type"] == "free" and not r["judgement"]["correct"]]

        def avg_entropy(results_list):
            if not results_list:
                return 0
            return float(np.mean([r["entropy_summary"]["avg_entropy"] for r in results_list]))

        summary = {
            "run_id": self.config.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": asdict(self.config),
            "total_samples": len(results),
            "correct_count": sum(1 for r in results if r["judgement"]["correct"]),
            "accuracy": sum(1 for r in results if r["judgement"]["correct"]) / len(results) if results else 0,
            "choice_accuracy": sum(1 for r in choice_results if r["judgement"]["correct"]) / len(choice_results) if choice_results else 0,
            "free_accuracy": sum(1 for r in results if r["eval_type"] == "free" and r["judgement"]["correct"]) / max(1, sum(1 for r in results if r["eval_type"] == "free")),
            "entropy_by_outcome": {
                "correct_vs_wrong": {
                    "correct_avg_entropy": avg_entropy(free_correct),
                    "wrong_avg_entropy": avg_entropy(free_wrong),
                },
                "per_sample": [
                    {
                        "id": r["trial_id"],
                        "eval_type": r["eval_type"],
                        "correct": r["judgement"]["correct"],
                        "avg_entropy": r["entropy_summary"]["avg_entropy"],
                        "max_entropy": r["entropy_summary"]["max_entropy"],
                        "avg_certainty_gap": r["entropy_summary"]["avg_certainty_gap"],
                        "avg_p1_ratio": r["entropy_summary"]["avg_p1_ratio"],
                        "low_confidence_steps": r["entropy_summary"]["low_confidence_steps"],
                        "high_entropy_steps": r["entropy_summary"]["high_entropy_steps"],
                    }
                    for r in results
                ]
            }
        }

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print(f"\n{'='*60}")
        print(f"最终结果: {self.config.run_id}")
        print(f"{'='*60}")
        print(f"准确率: {summary['accuracy']:.1%}")
        print(f"选择题准确率: {summary['choice_accuracy']:.1%}")
        print(f"开放题准确率: {summary['free_accuracy']:.1%}")
        print(f"\n正确 vs 错误样本的平均熵对比:")
        print(f"  正确: {avg_entropy(free_correct):.3f}")
        print(f"  错误: {avg_entropy(free_wrong):.3f}")
        print(f"  差异: {avg_entropy(free_correct) - avg_entropy(free_wrong):+.3f}")
        print(f"\n详细结果: {output_dir / 'summary.json'}")

def main():
    parser = argparse.ArgumentParser(description="Logits 分布熵分析 v0.3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig()

    if args.dry_run:
        print("=== Dry Run ===")
        print(json.dumps(asdict(config), indent=2))
        ds = get_dataset()
        print(f"\n数据集: {len(ds)} 条")
        return

    exp = LogitsEntropyAnalysis(config)
    exp.run()


if __name__ == "__main__":
    main()
