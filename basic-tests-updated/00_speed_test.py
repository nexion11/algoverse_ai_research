#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    balanced_selection,
    batched_generate_with_confidence,
    clear_accelerator_cache,
    context_block,
    environment_metadata,
    gold_resolved_hops,
    hop_prompt,
    load_musique,
    load_subject_model,
    memory_status,
    seed_everything,
    write_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--output", default="outputs/00_speed_test.json")
    args = ap.parse_args()

    seed_everything(args.seed)
    model, tok, device, dtype = load_subject_model(args.model, args.device, args.dtype)
    ds = load_musique()

    indices = balanced_selection(ds, max(4, args.n_prompts), args.seed)
    prompts = []
    for idx in indices:
        ex = ds[idx]
        ctx = context_block(ex)
        for h in gold_resolved_hops(ex):
            prompts.append(hop_prompt(tok, ctx, h["resolved_question"]))
            if len(prompts) >= args.n_prompts:
                break
        if len(prompts) >= args.n_prompts:
            break

    # Warm-up.
    print("\nWarm-up...")
    batched_generate_with_confidence(
        model, tok, device, prompts[:1], batch_size=1,
        max_new_tokens=4, progress_label="warmup"
    )

    rows = []
    for bs in args.batch_sizes:
        print(f"\n=== batch size {bs} ===")
        clear_accelerator_cache(device)
        t0 = time.perf_counter()
        try:
            results = batched_generate_with_confidence(
                model, tok, device, prompts, batch_size=bs,
                max_new_tokens=args.max_new_tokens,
                progress_label=f"bs={bs}",
            )
            elapsed = time.perf_counter() - t0
            gen_tokens = sum(r["generated_tokens"] for r in results)
            prompt_tokens = sum(r["prompt_tokens"] for r in results)
            rows.append({
                "batch_size": bs,
                "status": "ok",
                "elapsed_sec": elapsed,
                "prompts_per_sec": len(prompts) / elapsed,
                "generated_tokens_per_sec": gen_tokens / elapsed if elapsed else None,
                "prompt_tokens_total": prompt_tokens,
                "generated_tokens_total": gen_tokens,
                "memory": memory_status(device),
            })
        except RuntimeError as e:
            clear_accelerator_cache(device)
            rows.append({
                "batch_size": bs,
                "status": "failed",
                "error": str(e),
                "memory": memory_status(device),
            })
            print(f"Batch size {bs} failed: {e}")

    payload = {
        "environment": environment_metadata(args.model, device, dtype),
        "n_prompts": len(prompts),
        "results": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print("\nRESULTS")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
