# Caculus

## Differentiation
**Derivative:** $\frac{d}{dx} f(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$

**Higher-Order Derivatives:** $\frac{d^n}{dx^n} f(x)$

**Taylor's Formula:** $f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n$

## Integration

**Definite Integral:** $\int_{a}^{b} f(x) \, dx = F(b) - F(a)$ where $F'(x) = f(x)$

**Indefinite Integral:** $\int f(x) \, dx = F(x) + C$ where $F'(x) = f(x)$

## Matrix Calculus

**scalar-by-vector Derivative:** $\frac{\partial y}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial y}{\partial x_1} & \frac{\partial y}{\partial x_2} & \ldots & \frac{\partial y}{\partial x_n} \end{bmatrix}$

**vector-by-scalar Derivative:** $\frac{\partial \mathbf{y}}{\partial x} = \begin{bmatrix} \frac{\partial y_1}{\partial x} \\ \frac{\partial y_2}{\partial x} \\ \vdots \\ \frac{\partial y_m}{\partial x} \end{bmatrix}$

**vector-by-vector Derivative (Jacobian):** $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \ldots & \frac{\partial y_1}{\partial x_n} \\ \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \ldots & \frac{\partial y_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \ldots & \frac{\partial y_m}{\partial x_n} \end{bmatrix}$

**scalar-by-vector Second Derivative (Hessian):** $\frac{\partial^2 y}{\partial \mathbf{x}^2} = \begin{bmatrix} \frac{\partial^2 y}{\partial x_1^2} & \frac{\partial^2 y}{\partial x_1 \partial x_2} & \ldots & \frac{\partial^2 y}{\partial x_1 \partial x_n} \\ \frac{\partial^2 y}{\partial x_2 \partial x_1} & \frac{\partial^2 y}{\partial x_2^2} & \ldots & \frac{\partial^2 y}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 y}{\partial x_n \partial x_1} & \frac{\partial^2 y}{\partial x_n \partial x_2} & \ldots & \frac{\partial^2 y}{\partial x_n^2} \end{bmatrix}$

**Chain Rule:** If $\mathbf{y} = f(\mathbf{u})$ and $\mathbf{u} = g(\mathbf{x})$, then $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{u}} \frac{\partial \mathbf{u}}{\partial \mathbf{x}}$

## Common Derivatives
- $\frac{d}{dx} (c) = 0$
- $\frac{d}{dx} (x^n) = n x^{n-1}$
- $\frac{d}{dx} (e^x) = e^x$
- $\frac{d}{dx} (\ln x) = \frac{1}{x}$
- $\frac{d}{dx} (\sin x) = \cos x$
- $\frac{d}{dx} (\cos x) = -\sin x$
- $\frac{d}{dx} (\tan x) = \sec^2 x$
- $\frac{d}{dx} (\arcsin x) = \frac{1}{\sqrt{1 - x^2}}$
- $\frac{d}{dx} (\arccos x) = -\frac{1}{\sqrt{1 - x^2}}$
- $\frac{d}{dx} (\arctan x) = \frac{1}{1 + x^2}$
- $\frac{d}{dx} (uv) = u'v + uv'$
- $\frac{d}{dx} \left( \frac{u}{v} \right) = \frac{u'v - uv'}{v^2}$
- $\frac{d}{dx} (f(g(x))) = f'(g(x)) \cdot g'(x)$
- $\frac{d}{dx} \left( \int_{a}^{x} f(t) \, dt \right) = f(x)$
- $\frac{d}{dx} \left( \int_{x}^{b} f(t) \, dt \right) = -f(x)$
- $\frac{d}{dx} \left( \int_{u(x)}^{v(x)} f(t) \, dt \right) = f(v(x)) v'(x) - f(u(x)) u'(x)$
- $\frac{d}{dx} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$
- $\frac{d}{dx} (\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T) \mathbf{x}$
- $\frac{d}{dx} (\| \mathbf{x} \|_2^2) = 2 \mathbf{x}$
- $\frac{d}{dx} (\| \mathbf{x} \|_1) = \text{sign}(\mathbf{x})$ where $\text{sign}(x_i) = \begin{cases} 1 & x_i > 0 \\ -1 & x_i < 0 \\ 0 & x_i = 0 \end{cases}$
- $\frac{d}{dx} (\det(\mathbf{A})) = \det(\mathbf{A}) \cdot \text{tr}(\mathbf{A}^{-1} \frac{d\mathbf{A}}{dx})$
- $\frac{d}{dx} (\text{tr}(\mathbf{A})) = \text{tr}(\frac{d\mathbf{A}}{dx})$
- $\frac{d}{dx} (\mathbf{A}^{-1}) = -\mathbf{A}^{-1} \frac{d\mathbf{A}}{dx} \mathbf{A}^{-1}$
- $\frac{d}{dx} (\mathbf{A} \mathbf{B}) = \frac{d\mathbf{A}}{dx} \mathbf{B} + \mathbf{A} \frac{d\mathbf{B}}{dx}$
- $\frac{d}{dx} (\mathbf{A} + \mathbf{B}) = \frac{d\mathbf{A}}{dx} + \frac{d\mathbf{B}}{dx}$
- $\frac{d}{dx} (\mathbf{A} \circ \mathbf{B}) = \frac{d\mathbf{A}}{dx} \circ \mathbf{B} + \mathbf{A} \circ \frac{d\mathbf{B}}{dx}$
- $\frac{d}{dx} (\mathbf{A} \otimes \mathbf{B}) = \frac{d\mathbf{A}}{dx} \otimes \mathbf{B} + \mathbf{A} \otimes \frac{d\mathbf{B}}{dx}$
- $\frac{d}{dx} (\| \mathbf{A} \|_F^2) = 2 \mathbf{A}$