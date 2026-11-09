"""Render a concise operator benchmark summary from structured results."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    path = RESULTS / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def med(values):
    values = sorted(values)
    return values[len(values) // 2]


def main():
    final = load("final_validation.json")
    mha_base = load("mha_baseline_compare.json")
    group_norm = load("group_norm_validation.json")
    audio = load("audio_ops_validation.json")
    snake = load("snake_beta_nt_dynamic.json")
    aot = load("aot_snake_beta_runtime.json")
    weight_norm = load("weight_norm_check.json")
    media = load("media_contract_check.json")

    env = final or snake or audio or {}
    print("=" * 78)
    print(" MiniMax-H3 单卡初赛算子运行与性能汇总")
    print("=" * 78)
    print(f" host       : {env.get('hostname', 'n/a')}")
    print(f" torch      : {env.get('torch', 'n/a')}")
    print(f" hip        : {env.get('hip', 'n/a')}")
    print(f" device     : {env.get('device', 'n/a')}")
    print(" provenance : legacy result files predate the compliance rewrite")
    print(" scope      : historical measurements are not evidence for the current source")
    print()
    print("运行入口:")
    print("  srun --gres=dcu:1 bash scripts/run_operator_demo.sh")
    print("结果目录:")
    print("  results/*.json  results/screenshots/*.png")
    print()
    print("实际主链:")
    print("  Qwen条件 -> 首帧VAE编码 -> packed DiT x49 -> VAE解码 -> MP4")
    print("@@PAGE:02 DiT 热点算子")
    print("=" * 78)
    print(" [历史测量] DiT 候选算子；当前已恢复基础组合")
    print("=" * 78)
    if final:
        for row in final["results"]:
            if row.get("case") == "adaln_real_stride":
                print(
                    f" AdaLN       seq={row['seq']:<6} "
                    f"{row['baseline_ms']:.4f} -> {row['candidate_ms']:.4f} ms  "
                    f"{row['speedup']:.3f}x  cos={row['candidate_vs_baseline']['cosine']:.8f}"
                )
                print(
                    f" GatedRes    seq={row['seq']:<6} "
                    f"{row['residual_baseline_ms']:.4f} -> {row['residual_candidate_ms']:.4f} ms  "
                    f"{row['residual_speedup']:.3f}x  cos={row['residual_vs_baseline']['cosine']:.8f}"
                )
    if mha_base:
        for row in mha_base["results"]:
            if "formal_ms" in row and "ratio" in row:
                print(
                    f" MHA         seq={row['seq']:<6} "
                    f"reference {row['dense_ms']:.4f} -> formal {row['formal_ms']:.4f} ms  "
                    f"ratio={row['ratio']:.3f}x  cos={row['correctness']['cosine']:.8f}"
                )
            else:
                print(
                    f" MHA         seq={row['seq']:<6} historical SDPA result; "
                    "not attributable to the current explicit-attention source"
                )
    print()
    print("[正确性与正式路径检查]")
    if final:
        for row in final["results"]:
            if row.get("case") == "rms_norm_head_dim":
                seq = row["shape"][0]
                print(
                    f" Q/K RMSNorm seq={seq:<6} {row['native_ms']:.4f} ms "
                    f"cos={row['correctness']['cosine']:.8f}"
                )
            if row.get("case") == "mrope_h3_shape":
                print(
                    f" MRoPE       seq={row['seq']:<6} {row['native_ms']:.4f} ms "
                    f"cos={row['q_correctness']['cosine']:.8f} max_abs={row['q_correctness']['max_abs']:.1e}"
                )
    print()
    print("[长序列 MHA 显存]")
    if final:
        for row in final["results"]:
            if row.get("case") == "mha_non_causal" and row["seq"] in (8192, 39216):
                print(
                    f" seq={row['seq']:<6} time={row['native_ms']:.3f} ms "
                    f"peak={row['peak_allocated_gib']:.3f} GiB"
                )
    print("@@PAGE:03 音频算子与接口补齐")
    print("=" * 78)
    print(" [历史测量] NineToothed SnakeBeta；新增 exp 融合需复测")
    print("=" * 78)
    if snake:
        for row in snake["results"]:
            print(
                f" JIT L={row['shape'][2]:<7} {row['torch_ms']:.4f} -> "
                f"{row['ninetoothed_ms']:.4f} ms  {row['speedup']:.3f}x  "
                f"cos={row['correctness']['cosine']:.8f}"
            )
    if aot:
        for row in aot["results"]:
            print(
                f" AOT L={row['length']:<7} {row['torch_ms']:.4f} -> "
                f"{row['aot_ms']:.4f} ms  {row['speedup']:.3f}x  "
                f"cos={row['correctness']['cosine']:.8f}"
            )
    if weight_norm:
        print(
            f" WeightNorm  {weight_norm['conv1d_before_ms']:.4f} -> "
            f"{weight_norm['conv1d_after_ms']:.4f} ms  "
            f"{weight_norm['speedup']:.3f}x  modules={weight_norm['modules_materialized']}"
        )
    print()
    print("[实验 ATen 接口历史结果；未纳入当前正式构建]")
    if group_norm:
        for row in group_norm["results"]:
            print(
                f" GroupNorm shape={row['shape']} "
                f"ratio={row['speed_ratio']:.3f}x cos={row['correctness']['cosine']:.8f}"
            )
    if audio:
        for row in audio["results"]:
            print(
                f" {row['case']:<18} ratio={row['speed_ratio']:.3f}x "
                f"cos={row['correctness']['cosine']:.8f}"
            )
    print("@@PAGE:04 测试、规格与结论")
    print("=" * 78)
    print(" [历史算子测试；当前构建未复验]")
    print(" GroupNorm 18/18 | Conv1d 12/12 | ConvTranspose1d 12/12 | SnakeBeta 9/9")
    print()
    print("[媒体输出契约]")
    if media:
        valid = media["valid_contract"]
        print(
            f" frames={valid['frames']} channels={valid['channels']} "
            f"audio_ch={valid['audio_channels']} rate={valid['audio_sample_rate']}"
        )
        print(
            f" encoder={media['encoder_contract']['codec']} "
            f"{media['encoder_contract']['pixel_format']} "
            f"{media['encoder_contract']['fps']}fps"
        )
        print(" negative case:", media["negative_case"]["error"])
    print()
    print("[结论]")
    print(" 历史结果仅覆盖当时的源码；当前合规改造后需重新测量。")
    print(" 当前正式路径：AdaLN/GatedResidual 基础算子组合、显式 Attention、SnakeBeta JIT。")
    print(" 当前保留等价 weight-norm 常量展开；ATen 四算子桥未纳入正式构建。")
    print(" 待八卡复验：完整 FL2VA、音视频差异、端到端耗时、SSIM。")


if __name__ == "__main__":
    main()
