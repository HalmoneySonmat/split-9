"""Phase 3 adapter training — only the adapter learns; LLM and Go-Net frozen.

Loop summary, one step:

    boards (B, 8, 9, 9)            # from dataset
        │  Go-Net.forward_with_acts(layers=...)   [no_grad, frozen]
        ▼
    activations dict {layer: (B, C, 9, 9)}
        │  Resampler                              [trainable]
        ▼
    adapter_tokens (B, N, d_model)  ──────────────┐
                                                  │
    input_ids (B, T)  ── InstrumentedLLM ─────────┘   [base frozen,
        │                                              cross-attn blocks
        ▼                                              trainable]
    logits (B, T, vocab)
        │  shift + cross_entropy(ignore_index=-100, labels...)
        ▼
    loss → backward → optimizer.step (adapter params only)

The LLM forward dominates wall time; the adapter forward is a rounding
error in comparison. So *gradient_checkpointing* on the LLM would be the
biggest VRAM win if we hit OOM, even though it's "wasted" compute on
frozen layers — torch's autograd graph still wants them for backprop
through cross-attention insertion points.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.logging import TrainLogger, make_run_name
from ..utils.seed import set_global_seed
from .checkpoint import CheckpointManager

if TYPE_CHECKING:
    from ..adapter.projection import PerceiverResampler
    from ..gonet.network import GoNet
    from ..llm.instrumented import InstrumentedLLM


# ============================================================ config


@dataclass
class AdapterTrainConfig:
    """All knobs for ``train_adapter``. Sane PoC defaults."""

    seed: int = 42

    # ----- Adapter architecture (변경 1: 비대칭 + 후반 편중) -----
    #
    # Which Go-Net residual blocks feed the resampler. We extract from the
    # *upper half* (3, 4, 5) of the 6-block Go-Net, dropping low-level
    # blocks. Per-layer token counts bias toward the deepest layer:
    #
    #   layer 3 → 8 tokens   (mid-level: shape, liberty)
    #   layer 4 → 12 tokens  (high-mid: connection, group)
    #   layer 5 → 16 tokens  (high: territory, strategic value)
    #
    # Total adapter tokens = 36.
    act_layers: list[int] = field(default_factory=lambda: [3, 4, 5])
    act_layer_tokens: list[int] = field(default_factory=lambda: [8, 12, 16])

    # Which TinyLlama layers receive a gated cross-attention block. We
    # bias toward the *late* half (post-layer-10) so the adapter most
    # influences the final generation decision rather than low-level
    # word-form processing.
    inject_layers: list[int] = field(default_factory=lambda: [10, 16, 18, 20])

    # Optimization
    n_epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Logging / eval
    log_every: int = 50
    eval_every: int = 200
    eval_max_batches: int = 50  # cap so val isn't slow

    # IO
    checkpoint_dir: str = "runs/adapter_checkpoints"
    log_to: str = "csv"
    log_run_name: str | None = None
    log_dir: str = "runs"
    log_train_steps: bool = False  # accepted for API parity with script flag

    # Casting
    adapter_dtype: str = "bfloat16"  # match LLM


# ============================================================ loss


def adapter_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy with ``-100`` ignore.

    Args:
        logits: ``(B, T, V)`` from the LLM.
        labels: ``(B, T)`` with -100 marking positions to skip
            (prompt prefix + padding).

    Returns:
        Scalar loss.
    """
    # Shift: predict token at t+1 from logits at t.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ======================================================== train step


def train_step(
    instrumented: "InstrumentedLLM",
    gonet: "GoNet",
    resampler: "PerceiverResampler",
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    act_layers: list[int],
    device: torch.device,
    grad_clip: float = 1.0,
    adapter_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """One forward+backward+update on a single batch."""
    instrumented.train()  # adapter blocks need train mode (BN-like behaviour)
    resampler.train()

    boards = batch["board"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # 1) Go-Net activations — frozen, no grad through it.
    gonet.eval()
    with torch.no_grad():
        _, _, acts = gonet.forward_with_acts(boards, layers=act_layers)
    # Cast to the adapter's dtype so subsequent matmuls match the LLM.
    acts = {k: v.to(adapter_dtype) for k, v in acts.items()}

    # 2) Resampler → adapter tokens.
    adapter_tokens = resampler(acts)

    # 3) Instrumented LLM forward.
    logits = instrumented(
        input_ids=input_ids,
        adapter_tokens=adapter_tokens,
        attention_mask=attention_mask,
    )

    # 4) Next-token loss.
    loss = adapter_loss(logits, labels)

    # 5) Backprop only through adapter trainable params.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        # Gather the trainable params actually attached to optimizer.
        trainable = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "perplexity": float(torch.exp(loss.detach()).item()),
    }


# ======================================================= evaluation


@torch.no_grad()
def evaluate(
    instrumented: "InstrumentedLLM",
    gonet: "GoNet",
    resampler: "PerceiverResampler",
    val_loader: DataLoader,
    act_layers: list[int],
    device: torch.device,
    max_batches: int | None = None,
    adapter_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Compute average loss and perplexity on the validation loader."""
    instrumented.eval()
    resampler.eval()
    gonet.eval()

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        boards = batch["board"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        _, _, acts = gonet.forward_with_acts(boards, layers=act_layers)
        acts = {k: v.to(adapter_dtype) for k, v in acts.items()}
        adapter_tokens = resampler(acts)

        logits = instrumented(
            input_ids=input_ids,
            adapter_tokens=adapter_tokens,
            attention_mask=attention_mask,
        )

        # Per-batch CE; weight by # of non-ignored labels for a clean mean.
        n_real = (labels[:, 1:] != -100).sum().item()
        loss = adapter_loss(logits, labels)
        total_loss += float(loss.item()) * n_real
        total_tokens += n_real
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    if total_tokens == 0:
        return {"val_loss": float("nan"), "val_perplexity": float("nan")}

    avg_loss = total_loss / total_tokens
    return {
        "val_loss": avg_loss,
        "val_perplexity": float(torch.exp(torch.tensor(avg_loss)).item()),
    }


# ============================================================ main loop


def trainable_adapter_params(
    instrumented: "InstrumentedLLM", resampler: "PerceiverResampler"
) -> list[torch.nn.Parameter]:
    """Resampler + cross-attention block parameters. Excludes LLM base."""
    return list(resampler.parameters()) + list(
        instrumented.adapter_blocks.parameters()
    )


def train_adapter(
    instrumented: "InstrumentedLLM",
    gonet: "GoNet",
    resampler: "PerceiverResampler",
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: AdapterTrainConfig,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run the Phase 3 training loop.

    Returns final metrics ``{val_loss, val_perplexity, train_loss}``.
    """
    set_global_seed(cfg.seed)
    if device is None:
        device = next(instrumented.parameters()).device

    adapter_dtype = getattr(torch, cfg.adapter_dtype)

    params = trainable_adapter_params(instrumented, resampler)
    print(f"Trainable adapter params: {sum(p.numel() for p in params) / 1e6:.1f} M")

    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    run_name = cfg.log_run_name or make_run_name("phase3-adapter")
    logger = TrainLogger(
        run_name=run_name,
        config=asdict(cfg),
        mode=cfg.log_to,
        run_dir=cfg.log_dir,
    )
    if logger.is_active and logger.run_dir is not None:
        print(f"Logging to: {logger.run_dir}")

    ckpt_mgr = CheckpointManager(
        Path(cfg.checkpoint_dir),
        metric_name="val_loss",
        higher_is_better=False,  # lower loss is better
    )

    final_metrics: dict[str, float] = {
        "train_loss": float("inf"),
        "val_loss": float("inf"),
        "val_perplexity": float("inf"),
    }

    global_step = 0
    for epoch in range(cfg.n_epochs):
        epoch_t0 = time.time()
        for batch in train_loader:
            metrics = train_step(
                instrumented,
                gonet,
                resampler,
                optimizer,
                batch,
                cfg.act_layers,
                device,
                grad_clip=cfg.grad_clip,
                adapter_dtype=adapter_dtype,
            )

            if global_step % cfg.log_every == 0:
                print(
                    f"  epoch={epoch} step={global_step:5d} "
                    f"loss={metrics['loss']:.4f} "
                    f"ppl={metrics['perplexity']:.2f}"
                )
                logger.log(
                    {
                        "train/loss": metrics["loss"],
                        "train/perplexity": metrics["perplexity"],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            # Periodic eval.
            if (
                val_loader is not None
                and cfg.eval_every > 0
                and global_step > 0
                and global_step % cfg.eval_every == 0
            ):
                val_metrics = evaluate(
                    instrumented,
                    gonet,
                    resampler,
                    val_loader,
                    cfg.act_layers,
                    device,
                    max_batches=cfg.eval_max_batches,
                    adapter_dtype=adapter_dtype,
                )
                print(
                    f"  [eval] step={global_step:5d} "
                    f"val_loss={val_metrics['val_loss']:.4f} "
                    f"val_ppl={val_metrics['val_perplexity']:.2f}"
                )
                logger.log(
                    {
                        "val/loss": val_metrics["val_loss"],
                        "val/perplexity": val_metrics["val_perplexity"],
                    },
                    step=global_step,
                )
                _save_adapter_checkpoint(
                    ckpt_mgr,
                    instrumented,
                    resampler,
                    global_step,
                    metric=val_metrics["val_loss"],
                    optimizer=optimizer,
                )
                final_metrics.update(val_metrics)

            final_metrics["train_loss"] = metrics["loss"]
            global_step += 1

        elapsed = time.time() - epoch_t0
        print(f"epoch {epoch} done in {elapsed:.1f}s")

    # Final eval + save.
    if val_loader is not None:
        val_metrics = evaluate(
            instrumented,
            gonet,
            resampler,
            val_loader,
            cfg.act_layers,
            device,
            max_batches=cfg.eval_max_batches,
            adapter_dtype=adapter_dtype,
        )
        final_metrics.update(val_metrics)
        logger.log(
            {
                "val/loss": val_metrics["val_loss"],
                "val/perplexity": val_metrics["val_perplexity"],
            },
            step=global_step,
        )
        _save_adapter_checkpoint(
            ckpt_mgr,
            instrumented,
            resampler,
            global_step,
            metric=val_metrics["val_loss"],
            optimizer=optimizer,
        )

    logger.finish()
    return final_metrics


def _save_adapter_checkpoint(
    ckpt_mgr: CheckpointManager,
    instrumented: "InstrumentedLLM",
    resampler: "PerceiverResampler",
    step: int,
    metric: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save *only* the adapter components (resampler + xattn blocks).

    The LLM is frozen so we don't store it. We bundle the two adapter
    state_dicts into one file under a synthetic ``nn.Module`` for compat
    with the existing CheckpointManager API.
    """

    class _AdapterBundle(torch.nn.Module):
        def __init__(self, resampler, blocks) -> None:
            super().__init__()
            self.resampler = resampler
            self.adapter_blocks = blocks

    bundle = _AdapterBundle(resampler, instrumented.adapter_blocks)
    ckpt_mgr.save(step=step, model=bundle, optimizer=optimizer, metric=metric)


__all__ = [
    "AdapterTrainConfig",
    "adapter_loss",
    "train_step",
    "evaluate",
    "trainable_adapter_params",
    "train_adapter",
]
