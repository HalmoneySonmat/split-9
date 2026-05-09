"""PyTorch Dataset for Phase 3 adapter training.

Wraps tokenized ``Phase3Example`` instances. Each item the trainer sees:

    {
        "board"          : (8, 9, 9) float32 — Go-Net input
        "input_ids"      : (T,) int64  — full sequence (prompt + explanation)
        "attention_mask" : (T,) int64
        "labels"         : (T,) int64 — input_ids with prompt + padding masked
                                         to -100 (PyTorch cross-entropy
                                         ignore_index)
    }

The board is what the trainer feeds Go-Net to recompute activations on
the fly. We don't store activations on disk (see generation.py for the
rationale) so the training loop pays a Go-Net forward per example.

Tokenization is split out into ``tokenize_examples`` so the dataset
itself is tokenizer-agnostic — tests construct ``TokenizedExample``
instances directly without needing TinyLlama's tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .generation import Phase3Example

# Standard PyTorch ignore_index for cross_entropy.
LABEL_IGNORE = -100

DEFAULT_PROMPT = "Reasoning trace:\n"


# ============================================================ types


@dataclass
class TokenizedExample:
    """A Phase3Example after tokenization. Ready to feed the trainer."""

    board: torch.Tensor          # (8, 9, 9) float32
    input_ids: torch.Tensor      # (T,) int64
    attention_mask: torch.Tensor  # (T,) int64
    labels: torch.Tensor         # (T,) int64; -100 = ignore


# ============================================================ dataset


class Phase3Dataset(Dataset):
    """Tiny wrapper — TokenizedExamples in, dict-of-tensors out."""

    def __init__(self, tokenized: list[TokenizedExample]) -> None:
        self.tokenized = tokenized

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.tokenized[idx]
        return {
            "board": ex.board,
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "labels": ex.labels,
        }


def phase3_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a batch of dicts into one dict of stacked tensors.

    All tensors in a batch must have identical shapes; ``tokenize_examples``
    pads to ``max_length`` so that's automatic.
    """
    return {
        "board": torch.stack([b["board"] for b in batch], dim=0),
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }


# ============================================================ tokenize


def tokenize_examples(
    examples: list[Phase3Example],
    tokenizer: Any,
    max_length: int = 128,
    prompt_template: str = DEFAULT_PROMPT,
    verbose: bool = False,
) -> list[TokenizedExample]:
    """Tokenize ``Phase3Example`` list using a HuggingFace-style tokenizer.

    Each output ``TokenizedExample`` has the *prompt* portion of its
    ``labels`` masked to -100 so the LLM only learns to predict the
    *explanation* tokens, not the prompt prefix.

    Args:
        examples: From ``generate_dataset``.
        tokenizer: HF tokenizer with ``__call__`` accepting
            ``max_length``, ``padding``, ``truncation``, ``return_tensors``.
        max_length: Pad/truncate sequences to this length.
        prompt_template: Prefix string before each explanation. The exact
            string used at training time should match what's used at
            evaluation / generation time.
        verbose: Print a one-line progress message every 1000 examples.

    Returns:
        List of TokenizedExample, in the same order as ``examples``.
    """
    # Compute prompt length once (without special tokens, since we'll
    # tokenize the full text together with them).
    prompt_only_ids = tokenizer(
        prompt_template, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    prompt_len = int(prompt_only_ids.shape[0])
    # Add 1 if BOS will be prepended in the full encoding (most causal LMs do).
    bos_id = getattr(tokenizer, "bos_token_id", None)
    use_bos_offset = bos_id is not None

    out: list[TokenizedExample] = []
    for i, ex in enumerate(examples):
        full_text = prompt_template + ex.explanation
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[0].long()
        attention_mask = encoded.attention_mask[0].long()

        # Build labels = copy of input_ids, mask prompt + padding.
        labels = input_ids.clone()
        # Mask BOS + prompt tokens
        n_mask_prefix = prompt_len + (1 if use_bos_offset else 0)
        labels[: n_mask_prefix] = LABEL_IGNORE
        # Mask padding (where attention_mask == 0)
        labels[attention_mask == 0] = LABEL_IGNORE

        out.append(
            TokenizedExample(
                board=ex.board,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        )

        if verbose and (i + 1) % 1000 == 0:
            print(f"  tokenized {i + 1}/{len(examples)}")

    return out


# ============================================================ utilities


def split_train_val(
    tokenized: list[TokenizedExample],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[TokenizedExample], list[TokenizedExample]]:
    """Split by *game_id* parity is the right thing in real practice, but
    here we split randomly over examples — caller can pre-group by game
    if game-level split is needed."""
    import random

    rng = random.Random(seed)
    indices = list(range(len(tokenized)))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_fraction)
    val_idx = set(indices[:n_val])
    train = [t for i, t in enumerate(tokenized) if i not in val_idx]
    val = [t for i, t in enumerate(tokenized) if i in val_idx]
    return train, val


__all__ = [
    "TokenizedExample",
    "Phase3Dataset",
    "phase3_collate",
    "tokenize_examples",
    "split_train_val",
    "DEFAULT_PROMPT",
    "LABEL_IGNORE",
]
