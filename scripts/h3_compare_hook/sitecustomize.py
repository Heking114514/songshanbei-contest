"""Apply identical runtime settings to both official generator subprocesses."""
import os
import sys

if os.environ.get("H3_COMPARE_BF16_MATH") == "1" and os.path.basename(sys.argv[0]) == "minimax_h3.py":
    try:
        from h3_preflight import bf16_math, gpu_check, run_preflight
        if os.environ.get("H3_PREFLIGHT_OUTPUT"):
            # Check in the real generator, reusing its imports and GPU state.
            # Failures still stop before the model script is executed.
            run_preflight(os.environ["H3_SOURCE_ROOT"], os.environ["MODEL_PATH"],
                          os.environ["FIRST_FRAME"], os.environ["H3_PREFLIGHT_OUTPUT"],
                          int(os.environ.get("H3_EXPECTED_GPUS", "8")))
        else:
            gpu_check()
            bf16_math()
        if "--legacy" in sys.argv and int(os.environ.get("H3_REFERENCE_MAX_HEADS", "0")) > 0:
            from h3_reference_sdpa import install_reference_head_batch
            install_reference_head_batch()
            print("H3_REFERENCE_RUNTIME head_batch=" + os.environ["H3_REFERENCE_MAX_HEADS"], flush=True)
        print("H3_COMPARE_CHILD_RUNTIME attention=math_bf16", flush=True)
    except BaseException:
        # Python normally ignores sitecustomize errors; fail before model loading.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(78)
