import torch
from bigram import BigramLanguageModel, decode
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

f = open('output.txt', 'w', encoding='utf-8')

md = BigramLanguageModel()
md.to(device)
md.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
context = torch.zeros((1,1), dtype=torch.long, device=device)
f.write(decode(md.generate(context, max_new_tokens=10000)[0].tolist()))