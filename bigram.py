import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def gen_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad() #tell pytorch not to track gradients for backprop
def estimate_loss():
    out = {}
    model.eval() #set to eval phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = gen_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #set back to train phase
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,16)
        q = self.query(x) # (B,T,16)

        #compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5
        #transpose last and second to last, (B,T,16) @ (B,16,T) -> (B,T,T), sqrt(C) so that we dont favor just one node too much
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) #all elements where tril = 0 become -inf in wei, remove for all nodes communicate with each other, like sentiment analysis
        wei = F.softmax(wei, dim=-1) # normalizes, makes the same weighted multiplier matrix, research paper 3.2.1 (1)
        #softmax exponentiates (to the power of smth) each element, and divides by the sum
        #in our case it just uses the 0s, makes them 1s and then normalizes them
        wei = self.dropout(wei) # (B,T,T), randomly prevents some nodes from communicating to prevent overfitting
        v = self.value(x) # x is private info, v is what we use to aggregate
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """" Multi heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ linear layer followed by a nonlinearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(), #nonlinearity
            nn.Linear(4* n_embd, n_embd), #projection
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block with communication and computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #self attention head
        self.ffwd = FeedForward(n_embd) #feed forward layer
        self.ln1 = nn.LayerNorm(n_embd) #layer norm
        self.ln2 = nn.LayerNorm(n_embd) #layer norm

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # (B,T,C), apply to self attention head
        x = x + self.ffwd(self.ln2(x)) # (B,T,C), apply to feed forward layer
        return x   

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #lookup table for next token
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #lookup table for position
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)]) #3 transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) #layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #linear layer to get logits

    def forward(self, idx, targets=None):  #idx and targets are both (B,T) tensors
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) 
        x = tok_emb + pos_emb # (B,T,C) 
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,C=vocab_size) 

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T) #cross entropy has to have C as second dimension of logits
            loss = F.cross_entropy(logits, targets) # finding the loss of logits with respect to target

        return logits, loss

    def generate(self, idx, max_new_tokens): #extends (B,T) by max_new_tokens in the time direction
        #idx is a (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            #only use at max block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on last time step
            logits = logits[:, -1, :] # becomes (B,C) tensor, using last element in time dimension since that is the prediction for what is next
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            #append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) #pytorch optimizer

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = gen_batch('train') # get batch
    logits, loss = m(xb,yb) #evaluate loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))



# Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
# There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
# Each example across batch dimension is of course processed completely independently and never "talk" to each other
# In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
# "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
# "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. 