---
title: Theories of Learning
draft: true
date: 2024-08-14
---
### Aristotle:

Things are learnt through memorization and rationalization through **associations**.


### Alexander Bain:

Learnt information is in the connections 
Connectivism/Connectionist Machine


### Hebbian Learning:

$x_i$ triggers $y$ repeatedly, this connection strengthens 
$$
w_i = w_i + \eta x_i y
$$
There is only positive feedback so this causes an unstable learning problem


### Generalized Hebbian Learning a.k.a. Sanger's Rule:

$$
w_{ij} = w_{ij} + \eta y_i \left( x_i - \sum^j_{k=1} w_{ik} y_k \right)
$$

### Perceptron Learning Function:

$$
\begin{cases}
w = w + \eta \big( d(x) - y(x) \big)X \\
\\
d(x) \rightarrow \text{ desired o/p in response to input } x \\
\\
y(x) \rightarrow \text{ actual o/p in response to } x
\end{cases}
$$

We need Multilayered Perceptron (But all i/p o/p are Boolean)