---
title: The Basics (?)
draft: false
date: 2024-07-11
---
>[!info]
> Consider a set $S = \{ 1, 2, 3 \}$
> 
> $x_S$ is some subset of the set $S$, the complement of this subset can be demonstrated as $x_{-S}$.  


$$
\begin{equation}
\mathsf{D} = \alpha \mathsf{B} + c \implies D_{i,j} = \alpha B_{i,j} + c
\end{equation}
$$ 
**Broadcasting:** 
$$
\begin{equation}
\mathsf{D} = \mathsf{A} + b \implies D_{i,j} = A_{i,j} + b_j
\end{equation}
$$  
Elements of $b$ are added to each row of the matrix $\mathsf{D}$.

Since dot product gives a scalar it is a commutative operation in the following way:
$$x^\top y = (x^\top y)^\top = y^\top x$$
###### Matrix Multiplication

$$
\begin{gather}
\mathsf{C} = \mathsf{AB} \\
C_{i,j} = \sum_k A_{i,k} B_{k,j}
\end{gather}
$$
(each element in the matrix $\mathsf{C}$ is calculated via a dot product of $i$th row of $\mathsf{A}$ and $j$th column of $\mathsf{B}$.)

If $x$ and $y$ are solutions to $\mathsf{Ax} = \mathsf{b}$, then $z = \alpha x + (1 - \alpha)y$ is also a solution for any $\alpha \in \mathbb{R}$. 

>[!info]
>A visualization for $\mathsf{Ax} = \mathsf{b}$ can be seen as:
![[matmul.png]] 
>
>$x_1$ determines how much to go in $A_{:,1}$ direction. 
>
> Therefore for each element of the resulting vector we will have $\mathsf{A} x = \sum_i x_i A_{:,i}$, where $x_i A_{:,i}$ is a scalar multiplication of $x_i$ and all elements of the $A_{:,i}$ column vector. 
> 
> The vector-matrix multiplication operation can be translated into a linear combination of the column vectors of the original matrix, multiplied with each element of the vector column:
> $$
> \alpha A_{:,1} + \beta A_{:,2} + \gamma A_{:,3}
> $$


From the above, we see that $\mathsf{b}$ will be a solution of $\mathsf{Ax} = \mathsf{b}$ only when it is in the column span (range) of $\mathsf{A}$.

![[Abasis.drawio.png]]

$\mathsf{b}$ can be thought of as being pulled $x_1$ quantity in $A_{:,1}$ direction $x_2$ times in $A_{:,2}$ direction and $x_3$ times in $A_{:,3})$ direction. We need $\mathsf{A}$ to have at least $m$ independent columns in order to define $\mathbb{R}^m$ as their column space ($n \ge m$).

if $\mathsf{A}$ is a $3 \times 2$ matrix and $\mathsf{b}$ is $3 \times 1$, $x$ is 2-dimensional, so $x$ defines only a 2-D plane within $\mathbb{R}^3$, the equation has a solution iff $\mathsf{b}$ lies on that plane. 

```mermaid
flowchart TD

a["conditions"] --> b["have to at least describe m-dimensions"] & c["m-vectors should be linearly independent"]
b --> c
```

$x = \mathsf{A}^{-1}\mathsf{b}$ : at most one solution for each value of $\mathsf{b}$

If overdetermined ($n \ge m$), there is more than one way to parameterize the solution.

Any **Norm** function will be of the following form:   $f : V \rightarrow \mathbb{R}$

>[!note]
>Norm are generally a way of measuring distance which scales from scalar multiplication. 
>
>$L^0$ norm which corresponds to the number of non-zero entries in a vector is not a valid distance metric as scaling the vector by $\alpha$ does not change the number of non-zero entries.

**Infinity Norm:**
$$
\begin{equation}
\lVert x \rVert_\infty = \underset{i}{\max} | x_i |
\end{equation}
$$ 
This norm measures the maximum discrepancy between two elements in a vector.


**Frobenius Norm:**
$$
\begin{equation}
\lVert \mathsf{A} \rVert_F = \sqrt{\sum_{i,j}A^2_{i,j}}
\end{equation}
$$ 
Frobenius Norm is considered w.r.t a matrix, for vectors (or single column matrices) this will be equivalent to the $L^2$ norm of the vector.

Vectors having $\lVert \cdot \rVert_2 = 1$ and $x^\top y = 0$ are called *orthonormal vectors*.

**Diagonal Matrices:**
Diagonal Matrices have only diagonal elements, rest are all zeroes. 

$$
\text{diag}(v) = 
\begin{pmatrix}
v_1 & 0 & 0 & \dots \\
0 & v_2 & 0 & \dots \\
0 & 0 & v_3 & \dots \\
\vdots & \vdots & \vdots & \ddots 
\end{pmatrix}
$$ 
$$
\text{diag}(v)^{-1} = \text{diag}\left(\left[\frac{1}{v_1}, \dots, \frac{1}{v_n}\right]^\top \right)
$$ 
$$
\text{diag}(v)x = v \odot x
$$ 
$$
\underbrace{\huge{\mathsf{D}}}_{\text{diagonal}}x
$$ 

For a non-square tall diagonal matrix zeroes are added to the end of the matrix. For a wide matrix, elements of $x$ are dropped during multiplication.

**Symmetric Matrices:** $\mathsf{A} = \mathsf{A}^\top$

**Singular Matrices:** Square matrices with linearly dependent columns.

**Orthogonal Matrices:** rows (columns) mutually orthonormal
$$
\begin{gather*}
\mathsf{A}^\top \mathsf{A} = \mathsf{A} \mathsf{A}^\top = I \\
\mathsf{A}^{-1} = \mathsf{A}^\top
\end{gather*}
$$ 
###### Eigenvectors and Eigenvalues
$$
\mathsf{A} v = \underbrace{\lambda}_{\text{eigenvalue}} \overbrace{v}^{eigenvector}
$$ 
$$
v^\top \mathsf{A} = v^\top \lambda \, \, : \, \, \text{ right eigenvector \& eigenvalue}
$$ 

$$
\mathsf{A} (sv) = \lambda (sv) \, \, , \, \, \forall s \in \mathbb{R}
, s \not = o
$$ 
$$
\begin{gather*}
\mathsf{A}v = \lambda v \, \, \text{ and } \, \, \mathsf{A}u = \lambda u \\
\mathsf{A} v + \mathsf{A} u = \lambda v + \lambda u \\
\mathsf{A}(v + u) = \lambda (v + u) \\
\implies \mathsf{A} ( \alpha v + \beta u ) = \lambda ( \alpha v + \beta u )
\end{gather*}
$$ 

Matrix $\mathsf{A}$ has $n$ linearly independent eigenvectors $\{ v^{(1)}, \dots, v^{(n)} \}$ and corresponding eigenvalues $\{ \lambda_1, \dots, \lambda_n \}$.

Concatenating the vectors (one vector per column):
$$
\begin{gather*}
V = \left[ v^{(1)}, v^{(2)}, \dots, v^{(n)} \right] \\
\lambda = [\lambda_1, \lambda_2, \dots, \lambda_n]^\top
\end{gather*}
$$ 
**Eigendecomposition:**
$$
\begin{equation}
\mathsf{A} = V \text{diag}(\lambda) V^{-1}
\end{equation}
$$  
For any **Real Symmetric Matrix** we have:
$$
\begin{equation}
\mathsf{A} = Q \Lambda Q^\top 
\end{equation}
$$ 
where $Q$ is the diagonal matrix of eigenvalues and $Q^\top$ is orthogonal matrix of eigenvalues of $\mathsf{A}$. Eigendecomposition is guaranteed for real symmetric matrices, but they might not unique.

$\Lambda_{i,i}$ is eigenvalue related to $i$th column of $Q$ ( $Q_{:,i}$ ), since it's orthogonal (independent basis) we can think of $\mathsf{A}$ as scaling space by $\lambda_i$ in direction $v^{(i)}$ . If $\lambda_i = 0$, matrix is *singular*.

$$
\begin{align*}
&\mathsf{A} = Q \Lambda Q^\top& \\
&Q^\top \mathsf{A} \, Q = \Lambda& \\
&x^\top \mathsf{A} \, x = f(x)& \, \, s.t. \lVert x \rVert_2 = 1 \\
&v_1^\top \mathsf{A} \, v_1 = \lambda_1&
\end{align*}
$$ 
where $\lambda_1$ is eigenvalue corresponding to the specific eigenvectors.

$x^\top \mathsf{A} \, x = 0 \implies x = 0$  : A matrix whose eigenvalues are all positive : **positive definite**

$\forall x,  x^\top \mathsf{A} \, x \ge 0$  : A matrix whose eigenvalues are all positive or zero : **positive semidefinite**

A matrix whose eigenvalues are all negative : **negative definite**

A matrix whose eigenvalues are all negative or zero : **negative semidefinite**

