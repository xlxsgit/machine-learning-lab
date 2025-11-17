# Linear Algebra

<!-- TOC -->
* [Linear Algebra](#linear-algebra)
  * [Linear/Vector Space](#linearvector-space)
  * [Matrix Operations](#matrix-operations)
  * [Special Matrices](#special-matrices)
  * [Matrix Decompositions](#matrix-decompositions)
<!-- TOC -->

## Linear/Vector Space

**Vector Space:**
A set $V$ over field $\mathbb{R}$ is a vector space if:
- $\forall \mathbf{a}, \mathbf{b} \in V, \mathbf{a} + \mathbf{b} \in V$ (closure under addition)
- $\forall \mathbf{a} \in V, c \in \mathbb{R}, c \cdot \mathbf{a} \in V$ (closure under scalar multiplication)

**Basis Vectors:**
A set $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n\}$ is a basis if:
- $\forall \mathbf{v} \in V, \exists \lambda_1, \lambda_2, \ldots, \lambda_n \in \mathbb{R}$ such that $\mathbf{v} = \sum_{i=1}^{n} \lambda_i \mathbf{e}_i$
- The vectors are linearly independent

**Inner/Dot/Scalar Product:**
- $\langle \mathbf{a}, \mathbf{b} \rangle = \mathbf{a}^T \mathbf{b} = \sum_{i=1}^{n} a_i b_i$
- Properties: symmetric, linear, positive definite

**Outer Product:**
- $\mathbf{C} = \mathbf{a} \mathbf{b}^T$ where $C_{ij} = a_i b_j$
- Results in a matrix of rank 1

**Orthogonality:**
- Vectors $\mathbf{a}$ and $\mathbf{b}$ are orthogonal if $\langle \mathbf{a}, \mathbf{b} \rangle = 0$
- Orthonormal basis: $\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \delta_{ij}$

**Vector Norms:**
- $l_p$-norm: $\| \mathbf{a} \|_p = \left( \sum_{i=1}^{n} |a_i|^p \right)^{1/p}$
- $l_1$-norm (Manhattan): $\| \mathbf{a} \|_1 = \sum_{i=1}^{n} |a_i|$
- $l_2$-norm (Euclidean): $\| \mathbf{a} \|_2 = \sqrt{\langle \mathbf{a}, \mathbf{a} \rangle}$
- $l_\infty$-norm (Maximum): $\| \mathbf{a} \|_\infty = \max(|a_1|, |a_2|, \ldots, |a_n|)$

## Matrix Operations

**Linear Transformation:**
A mapping $T: V \to W$ is linear if:
- $T(\mathbf{a} + \mathbf{b}) = T(\mathbf{a}) + T(\mathbf{b})$
- $T(c \cdot \mathbf{a}) = c \cdot T(\mathbf{a})$
- Every linear transformation can be represented by matrix multiplication

**Affine Transformation:**
- $\mathbf{y} = M\mathbf{x} + \mathbf{b}$
- Combination of linear transformation and translation

**Matrix Addition:**
- $\mathbf{C} = \mathbf{A} + \mathbf{B}$ where $C_{ij} = A_{ij} + B_{ij}$
- Requires matrices of same dimensions

**Matrix Multiplication:**
- $\mathbf{C} = \mathbf{A} \mathbf{B}$ where $C_{ij} = \sum_{k} A_{ik} B_{kj}$
- Number of columns of $\mathbf{A}$ must equal number of rows of $\mathbf{B}$

**Hadamard Product (Element-wise):**
- $\mathbf{C} = \mathbf{A} \circ \mathbf{B}$ where $C_{ij} = A_{ij} B_{ij}$
- Requires matrices of same dimensions

**Kronecker Product:**
- $\mathbf{C} = \mathbf{A} \otimes \mathbf{B}$ where $C_{ik,jl} = A_{ij} B_{kl}$
- Block matrix construction

**Trace:**
- $tr(\mathbf{A}) = \sum_{i} A_{ii}$
- Properties: $tr(\mathbf{A} + \mathbf{B}) = tr(\mathbf{A}) + tr(\mathbf{B})$, $tr(\mathbf{A}\mathbf{B}) = tr(\mathbf{B}\mathbf{A})$

**Determinant:**
- $det(\mathbf{A}) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} A_{i, \sigma(i)}$
- Geometric interpretation: scaling factor of linear transformation
- $det(\mathbf{A}\mathbf{B}) = det(\mathbf{A})det(\mathbf{B})$

**Matrix Rank:**
- Dimension of column/row space
- $rank(\mathbf{A}\mathbf{B}) \leq \min(rank(\mathbf{A}), rank(\mathbf{B}))$
- Full rank: $rank(\mathbf{A}) = \min(m,n)$

**Matrix Norms:**
- $p$-norm: $\| \mathbf{A} \|_p = \sup_{\mathbf{x} \neq 0} \frac{\| \mathbf{A}\mathbf{x} \|_p}{\| \mathbf{x} \|_p}$
- Frobenius norm: $\| \mathbf{A} \|_F = \left( \sum_{m} \sum_{n} |A_{mn}|^2 \right)^{1/2}$


## Special Matrices

**Diagonal Matrix:**
- $A_{ij} = 0$ if $i \neq j$
- Denoted as $diag(\lambda_1, \lambda_2, \ldots, \lambda_n)$

**Identity Matrix:**
- $I_{ij} = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{otherwise} \end{cases}$
- $\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}$

**Symmetric Matrix:**
- $\mathbf{A} = \mathbf{A}^T$ ($A_{ij} = A_{ji}$)
- Real symmetric matrices have real eigenvalues and orthogonal eigenvectors

**Orthogonal Matrix:**
- $\mathbf{A}\mathbf{A}^T = \mathbf{A}^T\mathbf{A} = \mathbf{I}$
- Columns form orthonormal basis
- Preserves lengths and angles: $\| \mathbf{A}\mathbf{x} \|_2 = \| \mathbf{x} \|_2$

**Positive Definite Matrix:**
- $\mathbf{x}^T \mathbf{A} \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$
- All eigenvalues are positive
- Symmetric positive definite matrices have Cholesky decomposition

**Sparse Matrix:**
- Majority of elements are zero
- Efficient storage and computation

## Matrix Decompositions

**Eigen Decomposition:**
- $\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}$ where $\mathbf{A} \mathbf{q}_i = \lambda_i \mathbf{q}_i$
- $\mathbf{Q}$ contains eigenvectors, $\mathbf{\Lambda}$ contains eigenvalues
- Applicable to diagonalizable matrices

**Singular Value Decomposition (SVD):**
- $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ where $\mathbf{A} \mathbf{v}_i = \sigma_i \mathbf{u}_i$
- $\mathbf{U}$, $\mathbf{V}$ orthogonal, $\mathbf{\Sigma}$ diagonal with singular values
- Always exists for any matrix
- Applications: PCA, low-rank approximation, pseudoinverse

**Other Important Decompositions:**
- **QR Decomposition**: $\mathbf{A} = \mathbf{Q}\mathbf{R}$ (orthogonal × upper triangular)
- **LU Decomposition**: $\mathbf{A} = \mathbf{L}\mathbf{U}$ (lower × upper triangular)
- **Cholesky Decomposition**: $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ for positive definite matrices