import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

BATCH_SIZE = 64
BLOCK_SIZE = 256
EPOCHS = 5000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EMBD = 384
EVAL_ITERS = 200
N_TRANSFORMER_LAYER = 6
N_HEAD = 6
DROPOUT = 0.2

# ------------


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


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, HEAD_SIZE) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.tril: torch.Tensor
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) => (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-Attention in parallel"""

    def __init__(self, NUM_HEADS, HEAD_SIZE) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(HEAD_SIZE) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, N_EMBD) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: Communication followed by computation"""

    def __init__(self, N_EMBD, N_HEAD) -> None:
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedForward(N_EMBD)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD) for _ in range(N_TRANSFORMER_LAYER)],
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)

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
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


model = BigramLanguageModel()
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
