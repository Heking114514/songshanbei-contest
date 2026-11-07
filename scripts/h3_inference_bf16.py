"""Run H3 with the attention backend and short-smoke geometry verified on one DCU."""
import importlib.util
import os
from pathlib import Path
import sys
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
path = Path(os.environ["H3_SOURCE_ROOT"]) / "InfiniLM/examples/minimax_h3.py"
spec = importlib.util.spec_from_file_location("h3_validated_entry", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if os.environ.get("H3_SHORT_SMOKE") == "1":
    original = module.load_pipeline
    def load_short_pipeline(*args, **kwargs):
        pipe, text_encoder, transformer = original(*args, **kwargs)
        class ShortSmokePipeline(type(pipe)):
            @property
            def min_duration(self):
                return 0.0
        pipe.__class__ = ShortSmokePipeline
        return pipe, text_encoder, transformer
    module.load_pipeline = load_short_pipeline
print("H3_VALIDATED_RUNTIME attention=math_bf16 short_smoke=" + os.environ.get("H3_SHORT_SMOKE", "0"), flush=True)
original_runner_call = module.MiniMaxH3TransformerRunner.__call__
def audited_runner_call(self, inputs):
    if not getattr(self, "_formal_shape_logged", False):
        print("H3_FORMAL_DIT_INPUTS", {k: (str(getattr(inputs,k).dtype), tuple(getattr(inputs,k).shape)) for k in ["video_hidden_states","audio_hidden_states","encoder_hidden_states","position_ids"]}, flush=True)
        self._formal_shape_logged = True
    return original_runner_call(self, inputs)
module.MiniMaxH3TransformerRunner.__call__ = audited_runner_call
module.main()
