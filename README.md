# Implement-Self-attention
The self-attention mechanism is highly significant in contemporary NLP and computer vision. This repository demonstrates how to implement self-attention from scratch, offering a valuable resource for people looking to gain a deeper understanding of the attention concept underpinning numerous scientific papers.

For more information, readers may interested in the [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper.
# This project aims to introduce self-attention. 
The self-attention process involves the following steps:

1. The input for self-attention is a sequence. From this input sequence, we multiply it by three weight matrices: query (Q), key (K), and value (V) to get Q, K and V matrices.

2. We calculate the attention weights by multiplying Q with the transpose of K and then applying a softmax function to the resulting values.

3. The attention is calculated by multiplying the output of the softmax function with the value (V).
