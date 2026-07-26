# LLM Data Pipeline

The data-loading and embedding stages of a GPT-style language model, built from
scratch to understand how they actually work.

Tokenization is handled by the companion repo
([BPT-Tokenizer](https://github.com/AkshGupta2003/BPT-Tokenizer)). This repo covers
everything between *"a flat list of token ids"* and *"the tensor the first
transformer block sees."*

## The target

GPT-1, equation (2):

```
h0 = U·We + Wp
```

| symbol | what it is |
|---|---|
| `U`  | the input tokens as one-hot rows — notation only; in code it is a row lookup |
| `We` | token embedding table, **learned** |
| `Wp` | position embedding table, **learned** (GPT-1 explicitly rejected the sinusoidal encodings of the original Transformer) |

Everything follows the GPT papers rather than any single book, scaled down to a
corpus that fits on a laptop CPU.

## Progress

| step | what | status |
|---|---|---|
| 0 | train the BPE tokenizer on `the-verdict.txt`, vocab 1000 | done → `verdict_tokenizer.json` |
| 1 | chunking: token stream → `(input, target)` pairs | done — 53 chunks |
| 2 | `Dataset` + `DataLoader` (batching) | pending |
| 3 | token embedding table (`We`) | done |
| 4 | position table (`Wp`) + add + dropout → `h0` | pending |
| 5 | wrap as a module the transformer plugs into | pending |

All of it lives in [`data_pipeline.ipynb`](data_pipeline.ipynb), built up cell by
cell with the verification checks kept in place.

## Numbers

Corpus is *The Verdict* by Edith Wharton (1908, public domain), ~20 KB.

```
characters        20,479
tokens             6,998        (2.93 chars/token)
vocab_size         1,001        256 bytes + 744 merges + 1 special
context              256
stride               128        deliberate departure from the papers — see below
chunks                53
predictions       13,568        53 chunks x 256 slots
embedding table  1001 x 128     128,128 parameters
```

## Design notes

**`vocab_size` is derived, never hardcoded.** Asking the tokenizer to train to
`vocab_size=1000` yields 1000 entries (256 bytes + 744 merges), and then
`<|endoftext|>` takes id 1000 — so valid ids run `0..1000` and the embedding table
needs **1001** rows. Hardcoding 1000 crashes the moment a special token appears.

**`stride < context` is a deliberate deviation.** GPT-1 trains on *"contiguous
sequences of 512 tokens"* (stride equal to context) because it had far more text
than compute. Our situation is inverted — 6,998 tokens and compute to spare — so
halving the stride doubles the chunk count (27 → 53) and the optimizer steps per
epoch. The notebook is explicit that this buys **no new information**: coverage is
the same 6,913 tokens either way, and the cost is that the middle of the corpus
receives twice the gradient of its edges.

**Embedding gradients are sparse.** Only rows whose ids appear in a batch receive
gradient. 270 of the 1001 rows never appear in this corpus at all and stay random
forever — the price of byte-level BPE, which in exchange never produces an unknown
token.

**Position embeddings are unaffected by stride.** Every chunk is a full `context`
tokens long, so every slot `0..255` appears in all 53 chunks — no row of `Wp` is
starved relative to another.

## Setup

Requires the companion tokenizer repo as a sibling directory:

```bash
git clone git@github.com:AkshGupta2003/LLM-Data-Pipeline.git
git clone git@github.com:AkshGupta2003/BPT-Tokenizer.git
cd LLM-Data-Pipeline
```

Or point at a clone anywhere:

```bash
export BPE_TOKENIZER_PATH=/path/to/BPT-Tokenizer
```

Dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy regex loguru jupyterlab
```

CPU-only torch is intentional — the corpus is 7k tokens and a GPU buys nothing.

Then run `data_pipeline.ipynb` top to bottom from the repo root.

## Credits

- **Sebastian Raschka** — *Build a Large Language Model (From Scratch)* and the
  [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository, used
  as reference throughout. His chapter-2 notebooks are excellent and are *not*
  redistributed here — get them from his repo.
- **Andrej Karpathy** — [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE).
- Radford et al., *Improving Language Understanding by Generative Pre-Training*
  (GPT-1) and *Language Models are Unsupervised Multitask Learners* (GPT-2).
- `the-verdict.txt` — *The Verdict* by Edith Wharton, public domain via
  [Wikisource](https://en.wikisource.org/wiki/The_Verdict).
