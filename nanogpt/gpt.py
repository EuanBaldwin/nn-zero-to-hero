# Char-level transformer based on the structure of the decoder block of the Attention Is All You Need paper (with pre-norm)

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32 # how many sequences (blocks) are processed at once, B
block_size = 128 # maximum context length for predictions (how many token positions each sequence has, T)
max_iters = 5000
eval_interval = max_iters / 10
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128 # how many numbers represent each token at each position (sometimes C, othertimes C is vocab_size)
n_head = 4 # number of self-attention heads
n_layer = 4 # number of layers in the block
dropout = 0.2 # prob of dropout (prevents overfitting)

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train split (90%) and val split (10%)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # for cuda
    return x, y

@torch.no_grad() # never calling .backward() so don't need to store grads
def estimate_loss():
    out = {}
    model.eval() # set model to eval mode (currently irrelevant)
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set model back to train mode (currently irrelevant)
    return out


class Head(nn.Module):
    """a self-attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # as tril is not a param of the module and is instead what pytorch calls a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # (T, T)
        self.dropout = nn.Dropout(dropout) # randomly zeroes some of the elements of the input tensor with probability 'dropout'

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C=head_size)
        q = self.query(x) # (B, T, C)

        # compute affinity score
        wei = q @ k.transpose(-1, -2) * self.head_size**(-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T); not sure why he uses C instead of head_size in the video?
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiAttentionHead(nn.Module):
    """Multiple attention heads in parallel which we concat. over C"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # randomly zeroes some of the elements of the input tensor with probability 'dropout'

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """Simple linear layer followed by a non-linearity and dropout"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # 4* is based on attention is all you need paper
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # the projection layer of the residual connections
            nn.Dropout(dropout), # randomly zeroes some of the elements of the input tensor with probability 'dropout'
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block
    Enables per-token computation in ffwd after inter-token communication in sa heads.
    Also implements layer pre-normalisation.
    """
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # so everything works out channel-wise
        self.sa = MultiAttentionHead(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embd) # computation
        self.ln1 = nn.LayerNorm(n_embd) # 1st layer norm
        self.ln2 = nn.LayerNorm(n_embd) # 2nd layer norm

    def forward(self, x):
        # the x+ are residual connections to try and combat vanishing grads
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer-norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers, logtis is (B, T, C)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) x now holds positional info as well as token embd values
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # x-entropy expects (B*T, C)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # -1 would also work
            loss = F.cross_entropy(logits, targets) 
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx = (B, T)
        for _ in range(max_new_tokens):
            cond_idx = idx[:, -block_size:] # crop idx to the last block_size tokens as we only have positional embds for up to block_size context length
            logits, loss = self(cond_idx) # get the predictions
            logits = logits[:,-1,:] # keep only the logits from the last time step, as last-position logits are used because this is an autoregressive language model -> (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sample index along 1st (time) dim -> (B, T+1)
        return idx


model = TransformerLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()),' parameters')

# create the optimiser
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every so often eval the loss on train & val data
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# sample from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 0 corresponds to newline so we start with a new line
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))