"""Apply identical runtime settings to both official generator subprocesses."""
import os
import sys

if os.environ.get("H3_COMPARE_BF16_MATH") == "1" and os.path.basename(sys.argv[0]) == "minimax_h3.py":
    try:
        from h3_preflight import bf16_math, gpu_check
        gpu_check()
        bf16_math()
        print("H3_COMPARE_CHILD_RUNTIME attention=math_bf16", flush=True)
    except BaseException:
        # Python normally ignores sitecustomize errors; fail before model loading.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(78)
