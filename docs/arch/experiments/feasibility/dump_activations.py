"""
Phase A: 数据收集
加载 Qwen2.5-0.5B -> 前向传播 -> 存 .npz

用法: python dump_activations.py [--prompt-id ctr_001]
"""

import json, os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch

# SSL workaround for HF hub
import httpx, ssl
from huggingface_hub.utils import _http
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
_http.set_client_factory(lambda: httpx.Client(verify=ctx))

PROMPTS = {
    "ctr_001": (
        "文本1：小明养了一只猫和一只狗。猫叫咪咪，狗叫旺财。每天早上咪咪会在窗台上等日出。\n\n"
        "文本2：小明的宠物中只有狗会在早上叫醒他。猫通常要睡到中午才会活动。\n\n"
        "问题：咪咪早上会在哪里？\n"
        "A. 窗台上  B. 窝里睡到中午  C. 不确定\n\n"
        "请分析以上两段文本，回答这个问题。直接说出你的答案（A、B 或 C），然后简要解释推理过程。"
    ),
    "normal": (
        "Q：今天天气怎么样？\nA：今天天气晴朗，气温 22 到 28 度，适合外出活动。"
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", default="ctr_001", choices=list(PROMPTS.keys()))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prompt = PROMPTS[args.prompt_id]
    run_dir = Path(__file__).parent / "runs" / f"{args.prompt_id}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.prompt_id}] 加载模型 {args.model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if args.device == "cuda" else None,
        output_attentions=True,
        output_hidden_states=True,
    )
    model.eval()
    cfg = model.config
    print(f"  参数: {model.num_parameters() / 1e6:.1f}M | 层: {cfg.num_hidden_layers} | "
          f"heads: {cfg.num_attention_heads} | hidden: {cfg.hidden_size}")

    # ── 前向传播 ──
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids_np = inputs.input_ids.cpu().numpy()
    seq_len = input_ids_np.shape[1]
    print(f"  prompt 长度: {seq_len} tokens")

    print(f"  前向传播中 ...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    elapsed = time.time() - t0

    # ── 提取数据 ──
    # hidden_states[0] = embedding layer output
    # hidden_states[1..L] = layer outputs (24 层), each (1, seq_len, hidden_dim)
    # attentions[0..L-1] = attention probs per layer, each (1, n_heads, seq_len, seq_len)
    hs = [h[0].detach().cpu().float().numpy() for h in outputs.hidden_states[1:]]  # (n_layers, seq, hidden)
    attn = [a[0].detach().cpu().float().numpy() for a in outputs.attentions]       # (n_layers, n_heads, seq, seq)
    logits = outputs.logits[0].detach().cpu().float().numpy()  # (seq_len, vocab)

    n_layers = len(hs)
    print(f"  收集: {n_layers} 层 hidden, {len(attn)} 层 attention, 耗时 {elapsed:.2f}s")

    # ── 存盘 ──
    npz_path = run_dir / "activations.npz"
    save_dict = {
        "input_ids": input_ids_np,
        "logits": logits,
        "prompt": prompt,
    }
    for i in range(n_layers):
        save_dict[f"hidden_{i}"] = hs[i]
        save_dict[f"attn_{i}"] = attn[i]

    np.savez_compressed(npz_path, **save_dict)
    file_mb = os.path.getsize(npz_path) / 1e6
    print(f"  已保存: {npz_path} ({file_mb:.1f} MB)")

    config = {
        "prompt_id": args.prompt_id,
        "model": args.model,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": cfg.num_attention_heads,
        "hidden_dim": cfg.hidden_size,
        "elapsed": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Done.")


if __name__ == "__main__":
    main()
