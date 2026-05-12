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
| **Total Parameters** | **~25M** |

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

> *Once upon a time, there was a little girl named Lily. She loved to play in the park near her house. One day, she saw a small dog sitting by the pond. "Are you lost?" she asked. The dog wagged its tail and looked up at her with big brown eyes...*

---

## 📊 Results

- Validation loss converges to approximately **1.8 – 2.0** after 20,000 steps
- Model generates grammatically correct and coherent children's stories
- Training takes roughly **3–4 hours** on a T4 GPU

---

## 🙏 Acknowledgements

- Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan & Yuanzhi Li
- Architecture inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Tokenization via [tiktoken](https://github.com/openai/tiktoken) by OpenAI

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
