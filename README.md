# LEPTON

**Lightweight Efficient Processing Transformer with Optimized Neurons**

An experimental LLM architecture I'm building from scratch. It has some ideas I've been wanting to try out that I haven't seen in other models. One of the Acceleron series alongside [Tachyon](https://github.com/Supergoatscriptguy/Tachyon) (MoE) and [Pion](https://github.com/Supergoatscriptguy/Pion) (dense).

## The Ideas

**Thought Tokens** — The model gets extra "thinking" tokens at the start that it can use for internal reasoning. They never show up in the output. Kind of like learned chain-of-thought instead of prompting it.

**Memory Crystallization** — Important info gets compressed into memory slots that persist. So the model can actually remember things across the context instead of losing everything.

**Sparse Depth Routing** — Not every token needs the full model. Easy tokens route through 2 layers, hard tokens use all of them. Dynamic compute based on what's actually needed.

**Future Peek** — A small bidirectional hint about what's coming next. Similar to how your eyes jump ahead when you read.

**Speculative Decoding** — Predicts 4 tokens at once, then verifies. Gets you 2-4x faster generation when it works.

**Uncertainty Estimation** — The model actually tells you when it's confident vs guessing. Most LLMs hide this.

## Model Sizes

| Model | Params | When to use it |
|-------|--------|----------------|
| NANO | ~10M | Debugging |
| MICRO | ~25M | Quick experiments |
| TINY | ~112M | Development |
| SMALL | ~160M | Overnight runs |
| MEDIUM | ~350M | Real training |
| LARGE | ~760M | Full scale |
| XL | ~1.5B | If you have the GPU |
| XXL | ~3B | Go big or go home |

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Get the Tokenizer

Grab `pile_tokenizer.json` from: https://github.com/Supergoatscriptguy/Tokenizers

Put it in the same folder as the scripts.

### Get Data

I have a preprocessed dataset on HuggingFace: [SuperGoatScriptGuy/PreprocessedMIXED](https://huggingface.co/datasets/SuperGoatScriptGuy/PreprocessedMIXED)

Download the shards (or just some of them for testing) and point `--data_dir` at the folder.

### Train

```bash
# Basic training
python train.py --model_size tiny --batch_size 2 --grad_accum 32

# Custom paths
python train.py --model_size tiny --data_dir /path/to/data --checkpoint_dir /path/to/checkpoints

# Resume from checkpoint
python train.py --model_size tiny --resume checkpoints/step_5000.pt
```

### Training Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--model_size` | tiny | Pick your size |
| `--batch_size` | 4 | Batch size |
| `--grad_accum` | 16 | Gradient accumulation |
| `--lr` | 6e-4 | Learning rate |
| `--max_steps` | 50000 | When to stop |
| `--seq_len` | 512 | Sequence length |
| `--data_dir` | - | Where your shards are |
| `--checkpoint_dir` | checkpoints | Where to save |
| `--resume` | - | Checkpoint to continue from |

## How It Flows

```
Input Tokens
     │
     ▼
┌─────────────┐
│ Token Embed │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Thought    │  ← Internal reasoning tokens
│  Tokens     │
└─────────────┘
     │
     ▼
┌─────────────┐
│   Memory    │  ← Persistent memory read/write
│  Crystal    │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Future     │  ← Bidirectional peek
│   Peek      │
└─────────────┘
     │
     ▼
┌─────────────┐
│   Depth     │  ← Route to different depths
│  Router     │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Lepton     │  ← The actual transformer blocks
│  Blocks     │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Speculative │  ← Multi-token prediction
│   Heads     │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Uncertainty │  ← Confidence scores
│ Estimator   │
└─────────────┘
     │
     ▼
  Output
```

## Generation

```python
from model import create_lepton_tiny
import torch

model = create_lepton_tiny()
model.load_state_dict(torch.load('checkpoint.pt')['model'])
model.eval()

# Fast generation with speculative decoding
tokens = model.generate(prompt_ids, max_new_tokens=100, use_speculative=True)

# Get uncertainty scores too
tokens, uncertainties = model.generate_with_uncertainty(prompt_ids, max_new_tokens=100)
```

## License

MIT
