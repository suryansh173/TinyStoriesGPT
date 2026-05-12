# TinyStories GPT 🤖📖

A GPT-2 style language model trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The model learns to generate simple, coherent children's stories using a small transformer architecture trained entirely on Google Colab.

---

## 🧠 Model Architecture

| Parameter | Value |
|---|---|
| Layers | 6 |
| Attention Heads | 6 |
| Embedding Dimension | 384 |
| Context Length | 256 tokens |
| Vocabulary Size | 50,257 (GPT-2 tokenizer) |
| Dropout | 0.1 |
| **Total Parameters** | **~30M** |

The model uses:
- Token + Positional embeddings
- Causal Self Attention (Flash Attention supported)
- Pre-LayerNorm transformer blocks
- Tied input/output embeddings
- GELU activation in MLP layers

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Dataset | TinyStories |
| Tokenizer | GPT-2 BPE via `tiktoken` |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight decay=0.1) |
| Learning Rate | 3e-4 with cosine decay → 1e-5 |
| Warmup Steps | 1,000 |
| Total Iterations | 20,000 |
| Batch Size | 32 |
| Gradient Accumulation | 4 steps (effective batch = 128) |
| Precision | bfloat16 / float16 (AMP) |
| Gradient Clipping | 1.0 |
| Training Platform | Google Colab (T4 GPU) |

---

## 📓 Notebook Structure

| Cell | Description |
|---|---|
| 1 | Mount Google Drive |
| 2 | Load TinyStories Dataset and Dependencies |
| 3 | Tokenization & Data Preparation |
| 4 | Model Architecture (LayerNorm, Attention, MLP, Block) |
| 5 | GPT Config & Model Initialization |
| 6 | Training Configuration & Device Setup |
| 7 | Data Loader (get_batch) |
| 8 | Loss Estimation (estimate_loss) |
| 9 | Optimizer & Learning Rate Scheduler |
| 10 | Training Loop with Checkpointing |
| 11 | Plot Training & Validation Loss |
| 12 | Load Best Model & Generate Story |

---

## 📁 Project Structure

```
TinyStoriesGPT/
├── TinyStoriesGPT.ipynb   # Full training + generation notebook
└── README.md
```

> Checkpoints and `.bin` data files are saved to Google Drive and not tracked here.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch tiktoken datasets numpy tqdm matplotlib
```

### Training
1. Open `TinyStoriesGPT.ipynb` in **Google Colab**
2. Use a **GPU runtime** (T4 or better)
3. Mount your Google Drive when prompted
4. Run all cells

Training automatically:
- Tokenizes and caches data to Google Drive on first run
- Evaluates on validation set every 1,000 steps
- Saves best model whenever validation loss improves
- Saves full checkpoint every 2,000 steps for resuming

### Generating Stories
```python
prompt = "Once upon a time"
encoded = enc.encode(prompt)
input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

output = model.generate(
    input_ids,
    max_new_tokens = 300,
    temperature    = 0.8,   # lower = more coherent
    top_k          = 40     # restricts to top 40 tokens
)
print(enc.decode(output[0].tolist()))
```

---

## 📖 Sample Output

> *Once upon a time, there was a little bunny named Benny. One day, Benny saw a big tree in the woods. He wanted to climb it, but the tree was too tall for him. Benny decided to climb the tree and find a shady spot to rest...*

---

## 📊 Results


| Metric | Value |
|---|---|
| Final Train Loss | 1.7937 |
| Final Val Loss | 1.8058 |
| Best Val Loss | 1.8058 |
| Training Time | ~3-4 hours on T4 GPU |

---

## 🙏 Acknowledgements

- Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan & Yuanzhi Li
- Architecture inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Tokenization via [tiktoken](https://github.com/openai/tiktoken) by OpenAI

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
