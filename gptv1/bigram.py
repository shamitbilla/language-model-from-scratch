import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

BATCH_SIZE = 32
BLOCK_SIZE = 8
EPOCHS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_ITERS = 200

torch.manual_seed(1337)

with open("./data/shakesphere.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Generate a small batch of data of inputs x and targets y"""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, VOCAB_SIZE) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, MAX_NEW_TOKENS):
        for _ in range(MAX_NEW_TOKENS):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


model = BigramLanguageModel(VOCAB_SIZE)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    if epoch % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"Step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, MAX_NEW_TOKENS=500)[0].tolist()))
