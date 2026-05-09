"""Tests for Phase 3 dataset wrapping and tokenization.

We use a small mock tokenizer so the test runs without downloading
TinyLlama. The real tokenizer is exercised via integration scripts.
"""

from __future__ import annotations

import torch

from split_brain_go.data.dataset import (
    DEFAULT_PROMPT,
    LABEL_IGNORE,
    Phase3Dataset,
    TokenizedExample,
    phase3_collate,
    split_train_val,
    tokenize_examples,
)
from split_brain_go.data.generation import Phase3Example


# ---------------------------------------------------- mock tokenizer


class _MockTokenizer:
    """Minimal HF-tokenizer-shaped mock: encode each char as ord%50."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        max_length: int | None = None,
        truncation: bool = False,
        padding: str | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ):
        ids = [ord(c) % 50 + 3 for c in text]  # +3 to avoid pad/bos/eos
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        attn = [1] * len(ids)
        if padding == "max_length" and max_length is not None:
            pad_n = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_n
            attn = attn + [0] * pad_n
        if return_tensors == "pt":
            return type(
                "Out",
                (),
                {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attn], dtype=torch.long),
                },
            )()
        return {"input_ids": ids, "attention_mask": attn}


# ---------------------------------------------------- helpers


def _make_example(seed: int) -> Phase3Example:
    g = torch.Generator().manual_seed(seed)
    return Phase3Example(
        game_id=seed // 10,
        move_number=seed % 10,
        board=torch.randn(8, 9, 9, generator=g),
        action=seed % 82,
        explanation=f"Test explanation number {seed}.",
        selected_confidence=0.4,
        value_before=0.1,
        value_after=0.2,
    )


# ---------------------------------------------------- tokenize


def test_tokenize_returns_correct_shapes():
    examples = [_make_example(i) for i in range(5)]
    tok = _MockTokenizer()
    out = tokenize_examples(examples, tok, max_length=32)
    assert len(out) == 5
    for t in out:
        assert isinstance(t, TokenizedExample)
        assert t.board.shape == (8, 9, 9)
        assert t.input_ids.shape == (32,)
        assert t.attention_mask.shape == (32,)
        assert t.labels.shape == (32,)
        assert t.input_ids.dtype == torch.long


def test_prompt_is_masked_in_labels():
    """Labels for prompt-prefix positions must be LABEL_IGNORE."""
    examples = [_make_example(0)]
    tok = _MockTokenizer()
    [t] = tokenize_examples(
        examples, tok, max_length=64, prompt_template="PROMPT: "
    )
    # First (1 BOS + len('PROMPT: ')) = 1 + 8 = 9 positions should be ignored
    assert (t.labels[:9] == LABEL_IGNORE).all().item()


def test_padding_is_masked_in_labels():
    examples = [_make_example(0)]
    tok = _MockTokenizer()
    [t] = tokenize_examples(examples, tok, max_length=128, prompt_template="P: ")
    # Where attention_mask is 0, labels must be LABEL_IGNORE.
    pad_positions = (t.attention_mask == 0)
    assert (t.labels[pad_positions] == LABEL_IGNORE).all().item()


def test_explanation_tokens_kept_as_labels():
    """At least *some* labels should be non-ignore (the explanation part)."""
    examples = [_make_example(0)]
    tok = _MockTokenizer()
    [t] = tokenize_examples(examples, tok, max_length=64)
    real_labels = (t.labels != LABEL_IGNORE).sum().item()
    assert real_labels > 0


# ---------------------------------------------------- dataset


def test_dataset_len_and_indexing():
    tokenized = [
        TokenizedExample(
            board=torch.zeros(8, 9, 9),
            input_ids=torch.zeros(16, dtype=torch.long),
            attention_mask=torch.ones(16, dtype=torch.long),
            labels=torch.full((16,), LABEL_IGNORE, dtype=torch.long),
        )
        for _ in range(5)
    ]
    ds = Phase3Dataset(tokenized)
    assert len(ds) == 5
    item = ds[2]
    assert "board" in item and item["board"].shape == (8, 9, 9)
    assert "input_ids" in item and item["input_ids"].shape == (16,)


def test_collate_stacks_correctly():
    tokenized = [
        TokenizedExample(
            board=torch.randn(8, 9, 9),
            input_ids=torch.zeros(16, dtype=torch.long),
            attention_mask=torch.ones(16, dtype=torch.long),
            labels=torch.full((16,), LABEL_IGNORE, dtype=torch.long),
        )
        for _ in range(4)
    ]
    ds = Phase3Dataset(tokenized)
    batch = phase3_collate([ds[i] for i in range(4)])
    assert batch["board"].shape == (4, 8, 9, 9)
    assert batch["input_ids"].shape == (4, 16)
    assert batch["attention_mask"].shape == (4, 16)
    assert batch["labels"].shape == (4, 16)


# ---------------------------------------------------- splits


def test_split_train_val_proportions():
    tokenized = [
        TokenizedExample(
            board=torch.zeros(8, 9, 9),
            input_ids=torch.zeros(8, dtype=torch.long),
            attention_mask=torch.ones(8, dtype=torch.long),
            labels=torch.zeros(8, dtype=torch.long),
        )
        for _ in range(20)
    ]
    train, val = split_train_val(tokenized, val_fraction=0.2, seed=0)
    assert len(train) == 16
    assert len(val) == 4
    # No overlap (we can't compare TokenizedExample directly so check ids match originals)


def test_split_train_val_deterministic():
    tokenized = [
        TokenizedExample(
            board=torch.zeros(8, 9, 9),
            input_ids=torch.zeros(8, dtype=torch.long) + i,
            attention_mask=torch.ones(8, dtype=torch.long),
            labels=torch.zeros(8, dtype=torch.long),
        )
        for i in range(10)
    ]
    a1, b1 = split_train_val(tokenized, val_fraction=0.3, seed=42)
    a2, b2 = split_train_val(tokenized, val_fraction=0.3, seed=42)
    # Same seed → same split
    assert [t.input_ids[0].item() for t in a1] == [t.input_ids[0].item() for t in a2]


def test_default_prompt_is_used():
    """Smoke: DEFAULT_PROMPT applies if no prompt_template passed."""
    examples = [_make_example(0)]
    tok = _MockTokenizer()
    out = tokenize_examples(examples, tok, max_length=64)
    assert len(out) == 1
    assert isinstance(DEFAULT_PROMPT, str) and len(DEFAULT_PROMPT) > 0
