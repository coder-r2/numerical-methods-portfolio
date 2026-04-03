# Chaotic Dynamics and Fractal Boundaries

This repository explores how complex, deterministic nonlinear systems give rise to chaotic behavior and fractal geometry. It includes vectorized Python implementations for visualizing standard fractals and analyzing the onset of mathematical chaos.

## Features & Visualizations

### 1. The Mandelbrot Set
The Mandelbrot set is generated using fixed-point iteration on the complex plane. For a given complex number $c$, we iterate the sequence:
$$z_{n+1} = z_{n}^{2} + c$$
starting with $z_{0} = 0$. If the sequence remains bounded (i.e., $|z_{n}| \le 2$), the point belongs to the set. The engine iterates over a $1000 \times 1000$ complex grid, mapping the escape velocity of points that diverge.

![Mandelbrot Set](fractal-dynamics/imgs/mandelbrot.png)

### 2. Newton's Fractal
Applying the Newton-Raphson method to find the roots of complex functions yields intricate fractal boundaries between basins of attraction. For the function $f(z) = z^3 + 1$, the iterative update is:
$$z_{n+1} = z_{n} - \frac{f(z_{n})}{f^{\prime}(z_{n})}$$
Depending on the starting value, the iteration converges to one of three complex roots ($\omega_{0}, \omega_{1}, \omega_{2}$). The visualization maps the complex plane based on the final converged root, revealing a self-similar fractal boundary.

![Newton's Fractal](fractal-dynamics/imgs/newton_fractal.png)

### 3. Logistic Map & Chaos
To observe the onset of chaos through period-doubling bifurcations, this module models the logistic map:
$$x_{n+1} = A x_{n}(1 - x_{n})$$
For small values of the control parameter $A$, the sequence settles to a stable fixed point. As $A$ increases, the system undergoes bifurcations leading to periodic oscillations and eventually deterministic chaos.

![Logistic Map](fractal-dynamics/imgs/bifurcation.png)

### 4. Lyapunov Exponents
A key quantity characterizing chaotic dynamics is the Lyapunov exponent ($\lambda$), which measures the average exponential rate at which nearby trajectories diverge. For a discrete map, the finite-time Lyapunov exponent is:
$$\lambda_{n} = \frac{1}{n}\sum_{k=0}^{n-1}\ln|f^{\prime}(x_{k})|$$
For $A=4$, the exponent evaluates to a positive constant ($\approx \ln 2$), proving extreme sensitivity to initial conditions and confirming the chaotic nature of the system.

## Setup & Usage

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install numpy matplotlib