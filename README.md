# TorchTongue 👅🔥

*A minimal, custom Transformer model built from scratch in PyTorch for English-to-Japanese translation.*

## 📖 Description
TorchTongue is an educational Neural Machine Translation (NMT) project that implements the core architecture of the original *Attention Is All You Need* paper from the ground up. It trains an English-to-Japanese translation model using a subset of the `Helsinki-NLP/opus-100` dataset. 

The codebase is contained entirely within a Jupyter Notebook and features custom, easy-to-read implementations of:
- **Multi-Head Self-Attention & Cross-Attention** (with masked look-ahead for the decoder)
- **Transformer Encoder & Decoder Blocks** (configured with 4 layers, 8 heads, and 512 embedding dimensions)
- **Positional Embeddings** to retain sequence order

This makes it a great resource for understanding how modern LLMs and seq2seq models work under the hood without the abstraction overhead of large libraries.

## 🛠 Tech Stack
- **Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim.AdamW`)
- **Data Pipeline:** Hugging Face `datasets` (300k sample subset of `opus-100`, split 80/20)
- **Tokenization:** Hugging Face `transformers` (Pre-trained `MarianTokenizer` for en-jap)
- **Metrics:** `torchmetrics` (MulticlassAccuracy, BLEUScore)
- **Environment:** Jupyter Notebook

## 💡 Usage Example

Once the environment is set up and the model is initialized, you can pass English source tokens and target Japanese tokens through the model. Here is a quick look at how the custom `Translator` handles a forward pass:

```python
import torch

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the custom Transformer model
model = Translator().to(device)

# Example batch size of 16, sequence length (block size) of 128
src_ids = torch.randint(low=0, high=vocab_size, size=(16, 128)).to(device) # English input
tgt_ids = torch.randint(low=0, high=vocab_size, size=(16, 128)).to(device) # Japanese input

# Forward pass to get next-token prediction logits
logits = model(src_ids, tgt_ids)

print(f"Logits shape: {logits.shape}") 
# Expected Output: torch.Size([16, 128, vocab_size]) (Batch, Sequence Length, Vocab Size)
```
