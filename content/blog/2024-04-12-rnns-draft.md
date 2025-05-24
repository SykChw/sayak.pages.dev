---
title: Recurrent Machine Learning Models
draft: true
date: 2024-04-12
---


```mermaid
flowchart LR

mul1((* Wₕ))
mul2((* Wₓ))
add((+ b))
tanh([tanh])
yin[yₜ₋₁]
xin[Xₜ]
out[Yₜ]

yin --> mul1
xin --> mul2
tanh --> out

subgraph "RNN node"
mul1 & mul2 --> add --> tanh
end

subgraph "prev. hidden state"
yin
end

subgraph input
xin
end

subgraph output
out
end

```

**Recurrent Neural Networks (RNNs)** have a recurring set of weights that have been trained with the help of *Backpropagation Through Time (BPTT)* and utilises inputs from both the previous hidden state(s) and current input in order to predict a possible output prediction.

```mermaid
flowchart LR

mul1((* Wₕ))
mul2((* Wₓ))
add((+ b))
tanh([tanh])
hin[hₜ₋₁]
hout[hₜ]
af[Softmax]
in[Xₜ]
out[Yₜ]

hin --> mul1
in --> mul2
tanh --> af & hout
af --> out

subgraph RNN
mul1 & mul2 --> add --> tanh
end

```
The function representing vanilla RNNs is:
$$
y_t = a \left(W_x \cdot X_t + W_y \cdot y_{t-1} + b \right)
$$
where $a(\dots)$ is the activation function used and $W_x, W_y$ are the weights corresponding to the current input and previous hidden state for the single node in the Recurrent Neural Network.

There are various improvements upon this architecture such as LSTMs whose individual node structure contains 