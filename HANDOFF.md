# Handoff — what to do first when you enter WSL2

This document is the bridge between "Claude prepared the project for me" and "I'm running it on my own machine". Read this **first** when you finish the WSL2 install and open Ubuntu for the first time.

---

## What's already done (by Claude, in `D:\brain\split_brain_go\`)

- Full directory skeleton: `src/split_brain_go/{env,gonet,llm,adapter,data,training,eval}`, `tests/`, `scripts/`, `configs/`, `docs/`, `notebooks/`, `runs/`.
- Config files: `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `LICENSE`.
- `README.md` — one-page project intro.
- `scripts/smoke_test.py` — verifies imports, CUDA, TinyLlama load, OpenSpiel Go 9×9.
- `docs/decisions.md` — 7 ADRs (board size, LLM choice, language, environment, env library, adapter family, data policy).
- `docs/architecture.md` — interface specs for env, gonet, adapter, llm, data, eval modules.
- `docs/{evaluation_protocol.md, codebase_study.md, glossary.md}` — copies of the Phase 0 reference docs.

The Phase 0 sandbox compatibility check passed for OpenSpiel 9×9, Hydra, pandas, pytest, ruff, mypy. PyTorch and TinyLlama could not be tested in the sandbox (no GPU, restricted network for the PyTorch wheel index) — those are validated by *you* via `smoke_test.py` once you're in WSL2.

---

## What you must do (in order)

### 1. Fix the broken `.git` folder

Claude tried to `git init` from the sandbox but got blocked by Windows-mount permissions. The result is an empty, half-initialized `.git/` directory in this folder that you need to delete.

```bash
cd /mnt/d/brain/split_brain_go
rm -rf .git
```

If `rm -rf` itself fails (Windows permissions), open Windows Explorer, navigate to `D:\brain\split_brain_go\`, enable hidden files, and delete the `.git` folder manually.

### 2. Initialize git fresh

```bash
cd /mnt/d/brain/split_brain_go
git init -b main
git config user.email "<your email>"
git config user.name "<your name>"
git add .
git commit -m "Phase 0: skeleton"
```

If you plan to push to GitHub:

```bash
git remote add origin git@github.com:<you>/split-brain-go.git
git push -u origin main
```

### 3. Create the Python virtual environment

```bash
cd /mnt/d/brain/split_brain_go
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

You should now see `(.venv)` in your shell prompt.

### 4. Install PyTorch (with CUDA)

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

This downloads ~2 GB. Patience.

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want to see `2.3.1+cu121 True NVIDIA GeForce RTX <model>`.

### 5. Install everything else

```bash
pip install -r requirements.txt
```

Some packages (notably `bitsandbytes`) require a working CUDA install. If install fails, scroll up and read the actual error message — usually it tells you exactly what's missing.

### 6. Run the smoke test

```bash
python scripts/smoke_test.py
```

Expected output (last few lines):

```
=== summary ===
  OK    imports
  OK    cuda
  OK    llm
  OK    go
```

If any line says `FAIL`, scroll up to find the traceback and either:

- Fix it yourself (most issues are missing system packages or CUDA mismatch).
- Re-open the conversation with Claude and paste the failing section.

### 7. Tell Claude when you're at this point

Once `python scripts/smoke_test.py` shows 4 OKs, say so in chat. From there we'll move into the Week 2 tasks (notebooks, ADRs already drafted but verifiable, wandb login, first commit) and then Phase 1.1 (the actual 9×9 Go environment wrapper).

---

## Things you might be tempted to skip — don't

- **Don't skip the smoke test.** It's the Phase 0 quality gate. Half the bugs in later phases are environment misconfigurations that the smoke test catches in 30 seconds.
- **Don't skip `git init`.** Without git, you can't roll back when something breaks. And something will break.
- **Don't `pip install torch` without the CUDA index URL.** PyPI's default `torch` is the CPU build. You'll spend a day wondering why training is slow.
- **Don't put the `.venv` folder in OneDrive / Dropbox sync.** The repo lives in `D:\brain\split_brain_go\` — fine for sync if you want — but `.venv/` is gigabytes and would saturate your sync client. It's already in `.gitignore`; just don't sync it.

---

## What about WSL2 itself?

If you haven't installed WSL2 yet, the steps are in `D:\brain\phase0\how_to_proceed.md`, Day 1. Come back here after step 5 of that file.

---

## Found a problem in what Claude prepared?

Things that might bite later:

- **`HANDOFF.md` is in the repo.** Once Phase 0 is done, you can delete it or move it to `docs/`. It exists because of the .git permission problem; once you've done step 1, the document's job is done.
- **`.git/` half-initialized.** Same root cause. Delete and re-init.
- **`runs/` folder is empty but committed.** It exists so checkpoints have somewhere to go. The `.gitignore` keeps `runs/*` out of git, but the folder itself is tracked. Fine to leave as-is.
- **`flamingo-pytorch/` reference clone.** Mentioned in `phase0/how_to_proceed.md` Day 9 but not yet cloned. You'll do this yourself in Day 9.

---

## tl;dr

1. `rm -rf .git`
2. `git init -b main && git add . && git commit -m "Phase 0: skeleton"`
3. `python3.10 -m venv .venv && source .venv/bin/activate`
4. Install torch (CUDA wheel) and `pip install -r requirements.txt`
5. `python scripts/smoke_test.py` — 4 OKs
6. Ping Claude.
