# Mathematical Optimization

## Types of Optimization
### Discrete/Continuous Optimization
If the variables can only take specific discrete values, the optimization problem is considered discrete; otherwise, it is continuous.

For discrete optimization, common types include:
- **Combinatorial Optimization**: Optimization problems where the solution space is discrete and finite, often involving arrangements or selections of objects.
- **Integer Programming**: Optimization problems where some or all of the variables are restricted to be integers.

### Unconstrained/Constrained Optimization
If there are restrictions or constraints on the variables, the optimization problem is considered constrained; otherwise, it is unconstrained.

### Linear/Nonlinear Programming
If the objective function and constraints are linear, it is linear programming; otherwise, it is nonlinear programming.


## Optimization Algorithms
### Global/Local Optimization
If $x^*$ is the best solution in the entire solution space, it is a global optimum; if it is only the best in a small neighborhood, it is a local optimum.

- First-order necessary condition for local optimum: $\nabla f(x^*) = 0$
- Second-order necessary condition for local optimum: $\nabla^2 f(x^*)$ is positive semi-definite.
- Second-order sufficient condition for local optimum: $\nabla^2 f(x^*)$ is positive definite.

### Gradient Descent Method
For $\min f(x)$, the update rule is:
$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$
where $\alpha_k$ is the step size.

### Newton's Method
For $f(x)=0$, the update rule is:
$$x_{k+1} = x_k - [\nabla f(x_k)]^{-1} f(x_k)$$

For $\min f(x)$, the update rule is:
$$x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

### Lagrange Multipliers
For constrained optimization problems of the form:
$$\min f(x)$$
subject to
$$g_i(x) = 0, \quad i = 1, \ldots, m$$
the Lagrangian is defined as:
$$\mathcal{L}(x, \lambda) = f(x) + \sum_{i=1}^m \lambda_i g_i(x)$$
The necessary conditions for optimality are:
$$\nabla_x \mathcal{L}(x^*, \lambda^*) = 0$$
$$g_i(x^*) = 0, \quad i = 1, \ldots, m$$

### Karush-Kuhn-Tucker (KKT) Conditions
For optimization problems with inequality constraints:
$$\min f(x)$$
subject to
$$g_i(x) \leq 0, \quad i = 1, \ldots, m$$
$$h_j(x) = 0, \quad j = 1, \ldots, p$$
the KKT conditions are:
1. Stationarity:
   $$\nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \mu_j^* \nabla h_j(x^*) = 0$$
2. Primal feasibility:
   $$g_i(x^*) \leq 0, \quad i = 1, \ldots, m$$
   $$h_j(x^*) = 0, \quad j = 1, \ldots, p$$
3. Dual feasibility:
   $$\lambda_i^* \geq 0, \quad i = 1, \ldots, m$$
4. Complementary slackness:
   $$\lambda_i^* g_i(x^*) = 0, \quad i = 1, \ldots, m$$