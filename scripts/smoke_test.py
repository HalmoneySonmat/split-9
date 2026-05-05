"""Phase 0 smoke test.

Verifies the development environment can:
1. Import the core ML stack (torch, transformers, transformer_lens).
2. See a CUDA-capable GPU.
3. Load TinyLlama-1.1B and generate a few tokens.
4. Launch a 9x9 Go game via OpenSpiel and play one random rollout.

Run from the repo root:

    python scripts/smoke_test.py

Exits 0 on full success, 1 on failure. Each section prints a clear OK/FAIL line
so partial successes (e.g. CPU-only, no GPU) are still informative.
"""

from __future__ import annotations

import sys
import traceback


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def _version(mod, name: str) -> str:
    """Return a module's version, falling back to importlib.metadata or 'n/a'.

    Some libraries (notably transformer_lens) don't expose __version__ as an
    attribute, so a naive `mod.__version__` access can raise AttributeError.
    """
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version as _md_version

        return _md_version(name)
    except Exception:
        return "n/a"


def check_imports() -> bool:
    section("imports")
    try:
        import torch
        import transformers
        import transformer_lens
        import pyspiel
        import numpy as np
        import hydra
        import omegaconf

        print(f"torch              {_version(torch, 'torch')}")
        print(f"transformers       {_version(transformers, 'transformers')}")
        print(f"transformer_lens   {_version(transformer_lens, 'transformer_lens')}")
        print(f"pyspiel            {_version(pyspiel, 'open_spiel')}")
        print(f"numpy              {_version(np, 'numpy')}")
        print(f"hydra              {_version(hydra, 'hydra-core')}")
        print(f"omegaconf          {_version(omegaconf, 'omegaconf')}")
        return True
    except Exception:
        traceback.print_exc()
        print("FAIL: imports")
        return False


def check_cuda() -> bool:
    section("cuda")
    import torch

    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if available:
        n = torch.cuda.device_count()
        print(f"device_count = {n}")
        for i in range(n):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        # bf16 availability (RTX 30/40 should be True)
        print(f"bf16 supported     = {torch.cuda.is_bf16_supported()}")
        return True
    print("FAIL: no CUDA. Install the CUDA-12.1 PyTorch wheel and rerun.")
    return False


def check_llm() -> bool:
    section("llm load + generate")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        ids = tok("Hello", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        text = tok.decode(out[0])
        print(f"generated: {text!r}")
        return True
    except Exception:
        traceback.print_exc()
        print("FAIL: llm")
        return False


def check_go() -> bool:
    section("openspiel go 9x9")
    try:
        import random

        import pyspiel

        game = pyspiel.load_game("go(board_size=9)")
        state = game.new_initial_state()
        print(f"game        {game.get_type().short_name}")
        print(f"obs shape   {game.observation_tensor_shape()}")
        print(f"legal       {len(state.legal_actions())} (expect 82 = 81 cells + pass)")

        random.seed(0)
        moves = 0
        while not state.is_terminal() and moves < 200:
            state.apply_action(random.choice(state.legal_actions()))
            moves += 1
        print(f"random rollout finished in {moves} moves, returns={state.returns()}")
        return True
    except Exception:
        traceback.print_exc()
        print("FAIL: openspiel")
        return False


def main() -> int:
    results = {
        "imports": check_imports(),
        "cuda": check_cuda(),
        "llm": check_llm(),
        "go": check_go(),
    }

    section("summary")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
