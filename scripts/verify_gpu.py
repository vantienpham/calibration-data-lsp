#!/usr/bin/env python3
"""Check that the installed torch build actually runs on this GPU.

    uv run --no-sync python scripts/verify_gpu.py
    srun --gres=gpu:1 --time=0-00:05:00 uv run --no-sync python scripts/verify_gpu.py

`torch.cuda.is_available()` is not the test: it returns True on a wheel that
carries no kernels for the device's compute capability, and the failure only
surfaces at kernel launch. This launches real kernels -- a matmul, a reduction,
a float16 matmul (fp16 paths compile separately from fp32 ones) and a cuSOLVER
eigendecomposition (a different library from cuBLAS) -- and checks that the
results are finite and numerically right.

On a cluster with several GPU generations, run it on the oldest one you will
schedule on: that is the one a too-new wheel breaks on. The experiments need
bfloat16 and at least 40 GB of GPU memory (80 GB for Llama-3.1-8B).

Exit 0: the build runs here. Exit 1: do not run jobs on this GPU.
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"torch          : {torch.__version__}")
    print(f"built for cuda : {torch.version.cuda}")
    print(f"arch list      : {torch.cuda.get_arch_list()}")

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible", file=sys.stderr)
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"device         : {name}")
    print(f"compute cap    : sm_{major}{minor}")

    # An exact cubin is not required. CUDA is binary compatible *forward within a
    # major generation*, so an sm_60 cubin runs on sm_61 and an sm_86 one runs on
    # sm_89. Report which mechanism is in play; the kernel launches below are the
    # actual verdict either way.
    arch_list = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" in arch_list:
        print(f"cubin          : exact sm_{major}{minor}")
    else:
        same_major = [a for a in arch_list if a.startswith(f"sm_{major}")]
        compatible = [a for a in same_major if int(a.split("_")[1][1:]) <= minor]
        if compatible:
            print(f"cubin          : no exact sm_{major}{minor}; falling back to "
                  f"{max(compatible)} (forward-compatible within gen {major})")
        else:
            print(f"cubin          : NONE for gen {major}; only PTX JIT could save this",
                  file=sys.stderr)

    failures = []

    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        out = (a @ b).sum().item()
        assert out == out, "matmul produced NaN"  # NaN != NaN
        print(f"fp32 matmul    : ok (sum={out:.4f})")
    except Exception as exc:
        failures.append(f"fp32 matmul: {type(exc).__name__}: {exc}")

    try:
        x = torch.arange(1000, dtype=torch.float32, device="cuda")
        total = x.sum().item()
        assert abs(total - 499500.0) < 1.0, f"reduction wrong: {total} != 499500"
        print(f"fp32 reduction : ok (sum={total:.0f})")
    except Exception as exc:
        failures.append(f"fp32 reduction: {type(exc).__name__}: {exc}")

    try:
        # Models commonly load in fp16; fp16 kernels compile separately from fp32
        # ones, so a wheel can pass the fp32 checks and fail here.
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        out = (a @ a).float().sum().item()
        assert out == out, "fp16 matmul produced NaN"
        print(f"fp16 matmul    : ok (sum={out:.4f})")
    except Exception as exc:
        failures.append(f"fp16 matmul: {type(exc).__name__}: {exc}")

    try:
        # eigh/svd come from cuSOLVER, a different library from cuBLAS -- and the
        # one that misbehaves on older GPUs.
        m = torch.randn(64, 64, device="cuda")
        psd = m @ m.T + torch.eye(64, device="cuda")
        evals = torch.linalg.eigvalsh(psd)
        assert bool(torch.isfinite(evals).all()), "eigvalsh produced non-finite values"
        print(f"cusolver eigh  : ok (min eig={evals.min().item():.4f})")
    except Exception as exc:
        failures.append(f"cusolver eigh: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\nFAIL on {name} (sm_{major}{minor}):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nPASS: torch {torch.__version__} runs on {name} (sm_{major}{minor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
