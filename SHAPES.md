# Shapes through the pipeline

Quick reference for every tensor shape between raw text and `h0`.

## Config

| name | value | meaning |
|---|---|---|
| `B` | 4 | batch — chunks per batch |
| `T` | 256 | context — slots per chunk (`CONTEXT`) |
| `D` | 128 | model width (`EMB_DIM`) |
| `V` | 1001 | vocab size — 256 bytes + 744 merges + 1 special |
| stride | 128 | half of `T`, so chunks overlap |

## The chain

```
raw text              20,479 chars
  ↓ tokenizer
token_ids              6,998              flat list
  ↓ chunking (context 256, stride 128)
53 chunks              [256] + [256]      input, target
  ↓ DataLoader (batch_size 4)
inputs                 [4, 256]           ints
targets                [4, 256]           ints
  ↓ token embedding      We [1001, 128]
token_emb              [4, 256, 128]      floats
  ↓ add position         Wp  [256, 128]   broadcast over batch
sum                    [4, 256, 128]
  ↓ dropout                                shape unchanged
h0                     [4, 256, 128]      ← FINAL
```

In symbols:

```
[B, T]  --embed-->  [B, T, D]
```

## The two tables

```
We  [1001, 128]  = [V, D]    128,128 params
Wp  [ 256, 128]  = [T, D]     32,768 params
                             ─────────
                              160,896
```

`We` has one row per vocabulary entry. `Wp` has one row per **slot**, which is why
`T` can never exceed the number of rows in `Wp` — the hard cap on context length.

## Broadcasting in the add

```
token_emb   [4, 256, 128]
pos_emb     [   256, 128]
            ─────────────
sum         [4, 256, 128]
```

Torch aligns the trailing dimensions and reuses the same 256 position rows for all 4
sequences. That is correct: slot 3 means the same thing in every chunk.

## Output side (weight tying)

```
h0 @ We.T     [4, 256, 128] @ [128, 1001]  ->  logits [4, 256, 1001]
```

Flattened for the loss:

```
logits.flatten(0, 1)    [1024, 1001]
targets.flatten()       [1024]
```

`1024 = 4 × 256` — every slot in every chunk is one training example, flattened into
a single list.

## The part worth remembering

**`[B, T, D]` stays `[B, T, D]` all the way through the transformer.** Every attention
block and MLP takes `[4, 256, 128]` in and returns `[4, 256, 128]`. That fixed-width
pipe is the *residual stream* — blocks read from it and write back into it.

Only two places change shape:

| where | change |
|---|---|
| embedding (front) | `[B, T]` → `[B, T, D]` |
| output projection (end) | `[B, T, D]` → `[B, T, V]` |

Everything in between is shape-preserving.
