"""
Gravity Reranker v0.1 实验
验证核心命题：多样性信息是否被 softmax 丢弃，能否被重新利用。

用法:
    python run_reranker_v0.1.py                    # 下载模型并跑实验
    python run_reranker_v0.1.py --model Qwen/Qwen2.5-1.5B  # 换模型
    python run_reranker_v0.1.py --dry-run          # 只看配置，不跑

输出:
    experiments/data/run_001/  (每次递增)
"""

import json, os, sys, time, math, random, re, argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import numpy as np

# ---------- 引入数据集 ----------
sys.path.insert(0, str(Path(__file__).parent))
from dataset_ctr_v1 import get_dataset, build_prompt

# ========== 配置 ==========

@dataclass
class ExperimentConfig:
    # 模型
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    
    # 采样
    temperature: float = 0.7
    num_paths: int = 10
    max_new_tokens: int = 128
    top_p: float = 0.9
    random_seed: int = 42
    
    # Gravity Score
    alpha: float = 0.3         # 多样性权重
    diversity_metric: str = "cross_path_entropy"  # v0.1 只用这一个
    
    # 输出
    output_dir: str = ""
    run_id: str = ""
    
    def auto_paths(self):
        base = Path(__file__).parent / "data"
        base.mkdir(parents=True, exist_ok=True)
        existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
        n = len(existing) + 1
        self.run_id = f"run_{n:03d}"
        self.output_dir = str(base / self.run_id)


# ========== 实验引擎 ==========

class GravityRerankerExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        
    def setup(self):
        """加载模型"""
        cfg = self.config
        print(f"[setup] 加载模型: {cfg.model_name}")
        print(f"[setup] 设备: {cfg.device}, dtype: {cfg.dtype}")
        
        # 修复 Windows SSL 证书问题
        import httpx, ssl
        from huggingface_hub.utils import _http
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        def unverified_client_factory():
            return httpx.Client(verify=ssl_context)
        _http.set_client_factory(unverified_client_factory)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        torch_dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if cfg.device == "cuda" else None,
            output_attentions=False,
            output_hidden_states=False,
        )
        if cfg.device == "cpu":
            self.model = self.model.to("cpu")
        
        self.model.eval()
        print(f"[setup] 模型参数: {self.model.num_parameters() / 1e6:.1f}M")
        
        # 固定随机种子
        torch.manual_seed(cfg.random_seed)
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """从生成的文本中提取最终答案 A/B/C"""
        # 找 "A." / "B." / "C." 或 "答案是A" 等模式
        text_upper = text.upper()
        # 优先找明确的答案标记
        patterns = [
            r'(?:答案|选择|选)\s*[：:]\s*([ABC])',
            r'([ABC])\s*[.。]',
            r'^([ABC])\b',
            r'\b([ABC])\s*(?:是|选项|正确)',
        ]
        for pat in patterns:
            m = re.search(pat, text_upper)
            if m:
                return m.group(1)
        return None
    
    def _generate_single_path(self, prompt: str, seed: int, path_id: int) -> dict:
        """生成一条推理路径，返回 token logprobs 和最终答案"""
        cfg = self.config
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        # 设置随机种子（transformers generate 不支持 seed 参数）
        torch.manual_seed(seed)
        if self.config.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 提取生成的 token 和 logits
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logits_list = outputs.logits  # tuple of tensors, one per generated token
        
        # 计算每个 token 的 log probability
        token_logprobs = []
        tokens_decoded = []
        for i, token_id in enumerate(generated_ids):
            if i < len(logits_list):
                logits = logits_list[i][0]  # (vocab_size,)
                log_probs = torch.log_softmax(logits, dim=-1)
                token_logprob = log_probs[token_id].item()
                token_logprobs.append(token_logprob)
                token_text = self.tokenizer.decode([token_id])
                tokens_decoded.append(token_text)
        
        # 提取最终答案
        answer = self._extract_final_answer(generated_text)
        
        # 路径内熵：token logprob 的方差（高方差 = 推理过程犹豫）
        path_entropy = float(np.var(token_logprobs)) if token_logprobs else 0.0
        
        return {
            "path_id": path_id,
            "tokens": tokens_decoded,
            "token_logprobs": token_logprobs,
            "avg_log_prob": float(np.mean(token_logprobs)) if token_logprobs else 0.0,
            "path_entropy": path_entropy,
            "final_answer": answer,
            "full_response": generated_text,
        }
    
    def _compute_cross_path_diversity(self, paths: list) -> dict:
        """跨路径多样性度量"""
        answers = [p["final_answer"] for p in paths if p["final_answer"]]
        if not answers:
            return {"answer_distribution": {}, "cross_path_entropy": 0.0}
        
        counter = Counter(answers)
        total = len(answers)
        probs = [c / total for c in counter.values()]
        # 熵: -Σ p*log(p)
        entropy = -sum(p * math.log(p) for p in probs)
        # 归一化熵 (除以 log(N) 使值在 [0,1] 之间)
        max_entropy = math.log(len(answers))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return {
            "answer_distribution": dict(counter),
            "cross_path_entropy": normalized_entropy,
            "num_unique_answers": len(counter),
        }
    
    def _compute_scores(self, paths: list, diversity: dict) -> dict:
        """计算标准分数和 Gravity Score"""
        # 按答案聚合
        answer_stats = {}
        for p in paths:
            ans = p.get("final_answer")
            if not ans:
                continue
            if ans not in answer_stats:
                answer_stats[ans] = {"logprobs": [], "paths": 0}
            answer_stats[ans]["logprobs"].append(p["avg_log_prob"])
            answer_stats[ans]["paths"] += 1
        
        if not answer_stats:
            return {"standard_prediction": None, "gravity_prediction": None,
                    "standard_confidence": 0.0, "gravity_diversity_bonus": 0.0}
        
        # 标准方法：平均 log-prob 最高者
        std_scores = {ans: float(np.mean(stats["logprobs"])) 
                      for ans, stats in answer_stats.items()}
        standard_prediction = max(std_scores, key=std_scores.get)
        standard_confidence = std_scores[standard_prediction]
        
        # Gravity Score: avg_log_prob × (1 + α × diversity)
        cross_entropy = diversity["cross_path_entropy"]
        gravity_scores = {}
        for ans, stats in answer_stats.items():
            avg_lp = float(np.mean(stats["logprobs"]))
            # 该答案的「支持多样性」= 达到该答案的路径比例 × 整体多样性
            support_ratio = stats["paths"] / len(paths)
            diversity_bonus = support_ratio * cross_entropy
            gravity_scores[ans] = avg_lp * (1 + self.config.alpha * diversity_bonus)
        
        gravity_prediction = max(gravity_scores, key=gravity_scores.get)
        gravity_bonus = self.config.alpha * diversity["cross_path_entropy"]
        
        return {
            "standard_prediction": standard_prediction,
            "gravity_prediction": gravity_prediction,
            "standard_scores": std_scores,
            "gravity_scores": gravity_scores,
            "standard_confidence": float(standard_confidence),
            "gravity_diversity_bonus": float(gravity_bonus),
        }
    
    def run_trial(self, item: dict) -> dict:
        """对一条测试数据运行实验"""
        prompt = build_prompt(item)
        
        # 生成 N 条路径
        paths = []
        for i in range(self.config.num_paths):
            seed = self.config.random_seed + i * 1000 + hash(item["id"]) % 10000
            path = self._generate_single_path(prompt, seed, i)
            paths.append(path)
        
        # 计算多样性
        diversity = self._compute_cross_path_diversity(paths)
        
        # 计算分数
        scores = self._compute_scores(paths, diversity)
        
        # 构造结果
        correct = item["correct_answer"]
        result = {
            "trial_id": item["id"],
            "prompt": prompt,
            "correct_answer": correct,
            "paths": paths,
            "diversity_metrics": diversity,
            "standard_prediction": scores["standard_prediction"],
            "gravity_prediction": scores["gravity_prediction"],
            "standard_correct": scores["standard_prediction"] == correct if scores["standard_prediction"] else False,
            "gravity_correct": scores["gravity_prediction"] == correct if scores["gravity_prediction"] else False,
            "gravity_diversity_bonus": scores["gravity_diversity_bonus"],
        }
        
        return result
    
    def run(self):
        """运行完整实验"""
        cfg = self.config
        cfg.auto_paths()
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Gravity Reranker v0.1")
        print(f"Run ID: {cfg.run_id}")
        print(f"模型: {cfg.model_name}")
        print(f"路径数: {cfg.num_paths}, Temperature: {cfg.temperature}")
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
        
        # 加载数据集
        dataset = get_dataset()
        print(f"\n[run] 数据集: {len(dataset)} 条样本\n")
        
        # 运行每个 trial
        all_results = []
        std_correct = 0
        grav_correct = 0
        total = 0
        
        for i, item in enumerate(dataset):
            print(f"[{i+1}/{len(dataset)}] {item['id']}... ", end="", flush=True)
            t0 = time.time()
            
            result = self.run_trial(item)
            all_results.append(result)
            
            std_correct += 1 if result["standard_correct"] else 0
            grav_correct += 1 if result["gravity_correct"] else 0
            total += 1
            
            elapsed = time.time() - t0
            status = ""
            if result["standard_correct"] and not result["gravity_correct"]:
                status = " [!] Std OK Grav FAIL"
            elif not result["standard_correct"] and result["gravity_correct"]:
                status = " [+] Std FAIL Grav OK"
            elif result["standard_correct"] and result["gravity_correct"]:
                status = " [OK]"
            else:
                status = " [FAIL]"
            print(f"{elapsed:.1f}s{status}")
            
            # 每 5 条保存一次中间结果
            if (i + 1) % 5 == 0:
                self._save_intermediate(all_results, output_dir)
        
        # 保存完整结果
        self._save_final(all_results, output_dir, std_correct, grav_correct, total)
        
        return all_results
    
    def _save_intermediate(self, results: list, output_dir: Path):
        """保存中间结果"""
        # 只保存精简版（不含完整 generation 文本）
        slim = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "paths"}
            s["paths_summary"] = [
                {"path_id": p["path_id"], "final_answer": p["final_answer"], 
                 "avg_log_prob": p["avg_log_prob"], "path_entropy": p["path_entropy"]}
                for p in r["paths"]
            ]
            slim.append(s)
        
        with open(output_dir / "results_intermediate.jsonl", "w", encoding="utf-8") as f:
            for s in slim:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    def _save_final(self, results: list, output_dir: Path, 
                    std_correct: int, grav_correct: int, total: int):
        """保存最终结果"""
        # 完整结果（不含原始 tokens 列表，保留 full_response）
        with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                record = {k: v for k, v in r.items() if k != "paths"}
                record["paths"] = [
                    {k: v for k, v in p.items() if k != "tokens"}
                    for p in r["paths"]
                ]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # 计算统计
        std_accuracy = std_correct / total if total > 0 else 0
        grav_accuracy = grav_correct / total if total > 0 else 0
        
        # 不确定性分析
        uncertain_items = [r for r in results if r["correct_answer"] == "C"]
        std_uncertain_correct = sum(1 for r in uncertain_items if r["standard_prediction"] == "C")
        grav_uncertain_correct = sum(1 for r in uncertain_items if r["gravity_prediction"] == "C")
        uncertain_total = len(uncertain_items)
        
        # 错误自信分析
        std_false_confidence = sum(1 for r in results 
                                   if r["standard_prediction"] and r["standard_prediction"] != r["correct_answer"])
        grav_false_confidence = sum(1 for r in results
                                    if r["gravity_prediction"] and r["gravity_prediction"] != r["correct_answer"])
        
        # 多样性贡献分析
        diversity_helpful = sum(1 for r in results 
                                if not r["standard_correct"] and r["gravity_correct"])
        diversity_harmful = sum(1 for r in results
                                if r["standard_correct"] and not r["gravity_correct"])
        
        summary = {
            "run_id": self.config.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": asdict(self.config),
            "total_samples": total,
            "standard_accuracy": round(std_accuracy, 4),
            "gravity_accuracy": round(grav_accuracy, 4),
            "accuracy_delta": round(grav_accuracy - std_accuracy, 4),
            "uncertain_total": uncertain_total,
            "standard_uncertain_correct": std_uncertain_correct,
            "gravity_uncertain_correct": grav_uncertain_correct,
            "standard_false_confidence": std_false_confidence,
            "gravity_false_confidence": grav_false_confidence,
            "diversity_helpful": diversity_helpful,
            "diversity_harmful": diversity_harmful,
            "per_sample": [
                {
                    "id": r["trial_id"],
                    "correct": r["correct_answer"],
                    "std_pred": r["standard_prediction"],
                    "grav_pred": r["gravity_prediction"],
                    "std_correct": r["standard_correct"],
                    "grav_correct": r["gravity_correct"],
                    "diversity_entropy": r["diversity_metrics"]["cross_path_entropy"],
                    "diversity_bonus": r["gravity_diversity_bonus"],
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
        print(f"标准方法准确率:  {std_correct}/{total} = {std_accuracy:.1%}")
        print(f"Gravity 准确率:   {grav_correct}/{total} = {grav_accuracy:.1%}")
        print(f"差值:             {grav_accuracy - std_accuracy:+.1%}")
        print(f"\n不确定性拒绝 (正确答案为 C 的样本):")
        print(f"  标准方法: {std_uncertain_correct}/{uncertain_total}")
        print(f"  Gravity:  {grav_uncertain_correct}/{uncertain_total}")
        print(f"\n错误自信:")
        print(f"  标准方法: {std_false_confidence}/{total}")
        print(f"  Gravity:  {grav_false_confidence}/{total}")
        print(f"\n多样性贡献分析:")
        print(f"  帮助纠正: {diversity_helpful} 条")
        print(f"  反而变差: {diversity_harmful} 条")
        print(f"\n详细结果: {output_dir / 'summary.json'}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Gravity Reranker v0.1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="HuggingFace 模型名")
    parser.add_argument("--num-paths", type=int, default=10, help="每个问题的采样路径数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--alpha", type=float, default=0.3, help="多样性权重")
    parser.add_argument("--dry-run", action="store_true", help="只显示配置，不跑实验")
    args = parser.parse_args()
    
    config = ExperimentConfig(
        model_name=args.model,
        num_paths=args.num_paths,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    
    if args.dry_run:
        print("=== Dry Run ===")
        print(json.dumps(asdict(config), indent=2))
        dataset = get_dataset()
        print(f"\n数据集: {len(dataset)} 条样本")
        from collections import Counter
        c = Counter(d["correct_answer"] for d in dataset)
        print(f"答案分布: {dict(c)}")
        return
    
    exp = GravityRerankerExperiment(config)
    exp.run()


if __name__ == "__main__":
    main()
