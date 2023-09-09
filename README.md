# gpt

This project attempts to build the transformer architecture described in the Attention Is All You Need [1] paper, using pytorch. This project currently only builds the decoder architecture of the transformer–I am working on building the encoder block and adding additional functionality to this. This was only an exercise for me to learn and understand the architecture behind the transformer, in practice, this code is slow and while it is able to make output that is more meaningful than random characters, the output is not usable. bigram.py contains the code and comments, and gpt-trial.ipynb is the colab notebook I used to train the model, to make use of GPU acceleration. 

This project relied heavily on Andrej Karpathy's building GPT [youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY), but I am continually learning and implementing new things to build on his project.


References:
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
