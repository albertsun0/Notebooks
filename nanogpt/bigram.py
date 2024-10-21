import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32  # how many independent sequences will we process in parallel
block_size = 8  # maximum context lenght for prediction
max_iters = 10000
eval_iters = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

with open("dataset.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

sp = int(len(data) * 0.9)
train_data = data[:sp]
val_data = data[sp:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(
        len(data) - block_size - 1, (batch_size,)
    )  # get batch_size random numbers for start indexes of our blocks
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # for each index
    # context = x[batch_index][:index + 1]
    # target = y[index]
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Xb, Yb = get_batch(split)
            logits, loss = model(Xb, Yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, VOCAB_SIZE):
        super().__init__()
        # create a VOCAB_SIZE x VOCAB_SIZE lookup table for probabilities
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

    def generate(self, idx, max_new_tokens):
        # idk is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(VOCAB_SIZE)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    if step % 1000 == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']}, val loss {losses['val']}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print output
idx = torch.zeros(
    (1, 1), dtype=torch.long, device=device
)  # [0] as input (newline char)
print(
    decode(
        m.generate(
            idx,
            max_new_tokens=300,
        )[0].tolist()
    )
)
