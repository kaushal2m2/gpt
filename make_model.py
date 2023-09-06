import torch
from bigram import BigramLanguageModel, device, learning_rate, estimate_loss, gen_batch, max_iters, eval_interval

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

torch.save(m.state_dict(), 'model.pth')