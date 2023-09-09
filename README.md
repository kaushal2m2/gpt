# gpt

This project attempts to build the transformer architecture described in the Attention Is All You Need [1] paper, using pytorch. This project currently only builds the decoder architecture of the transformer–I am working on building the encoder block and adding additional functionality to this. This was only an exercise for me to learn and understand the architecture behind the transformer, in practice, this code is slow and while it is able to make output that is more meaningful than random characters, the output is not usable. bigram.py contains the code and comments, and gpt-trial.ipynb is the colab notebook I used to train the model, to make use of GPU acceleration. 

The important equation from this paper which is implemented in the code is as follows:

> _Attention(Q,K,V) = softmax(QK<sup>T</sup> / sqrt(d<sub>k</sub>) )V_

This is used to make each token that is being queried pay attention to the keys and the values of all previous tokens, so it can take into account in ranging degrees how relavant something before it is.

This project relied heavily on Andrej Karpathy's building GPT [youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY), but I am continually learning and implementing new things to build on his project.


References:
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
