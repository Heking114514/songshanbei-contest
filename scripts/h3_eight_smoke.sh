#!/usr/bin/env bash
# Full model, short eight-GPU inference; not the formal comparison score.
set -euo pipefail
export H3_EXPECTED_GPUS=8 H3_FORBID_JIT_COMPILATION=0
export OUTPUT_DIR="${OUTPUT_DIR:-$H3_REPRO_WORK/eight-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
python -u "$H3_SOURCE_ROOT/scripts/h3_preflight.py" --source-root "$H3_SOURCE_ROOT" \
    --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" --expected-gpus 8 \
    --output "$H3_REPRO_WORK/reports/preflight-eight-smoke.json"
bash --noprofile --norc "$H3_SOURCE_ROOT/scripts/run_h3_smoke.sh"
python -u - <<'VERIFY'
from pathlib import Path
import av,json,os
p=Path(os.environ["OUTPUT_DIR"])/"smoke.mp4"
with av.open(str(p)) as container:
    frames=list(container.decode(video=0))
assert len(frames)==22 and all(f.width==256 and f.height==256 for f in frames)
r=dict(passed=True,gpus=8,full_model=True,frames=22,steps=2,seed=42,width=256,height=256,
       prompt="A dog falls asleep",video=str(p),bytes=p.stat().st_size,formal_compare=False)
(Path(os.environ["H3_REPRO_WORK"])/"reports/eight-smoke-result.json").write_text(json.dumps(r,indent=2))
print("H3_EIGHT_FULL_MODEL_SMOKE_OK",json.dumps(r),flush=True)
VERIFY
