import numpy as np

def generate_mandelbrot(x_min=-2.5, x_max=1.0, y_min=-1.5, y_max=1.5, width=1000, height=1000, max_iter=100):
    """
    Generates the Mandelbrot set using vectorized numpy operations.
    Returns a 2D array of iteration counts before divergence.
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
    z = np.zeros_like(c)
    
    iter_counts = np.full(c.shape, max_iter, dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask]**2 + c[mask]

        diverged_now = mask & (np.abs(z) > 2)
        iter_counts[diverged_now] = i
        
    return iter_counts

def generate_newton_fractal(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, width=1000, height=1000, tol=1e-6, max_iter=100):
    """
    Generates Newton's Fractal for the function f(z) = z^3 + 1.
    Returns a 2D array indicating which root each point converged to.
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y

    r0 = -1.0 + 0j
    r1 = 0.5 + 1j * (np.sqrt(3) / 2)
    r2 = 0.5 - 1j * (np.sqrt(3) / 2)
    
    for _ in range(max_iter):
        mask = np.abs(z) > 0 
        z[mask] = z[mask] - (z[mask]**3 + 1) / (3 * z[mask]**2)
        
    root_map = np.zeros(z.shape, dtype=int)
    root_map[np.abs(z - r0) < tol] = 0
    root_map[np.abs(z - r1) < tol] = 1
    root_map[np.abs(z - r2) < tol] = 2
    
    return root_map

def simulate_logistic_map(A_min=0.89, A_max=3.995, A_step=0.0125, x0=0.5, total_iter=200, discard_iter=15):
    """
    Simulates the logistic map x_{n+1} = A * x_n * (1 - x_n) for a range of A values.
    Returns arrays of A values and their corresponding steady-state x values for the bifurcation diagram.
    """
    A_values = np.arange(A_min, A_max, A_step)
    A_plot = []
    x_plot = []
    
    for A in A_values:
        x = x0
        for i in range(total_iter):
            x = A * x * (1 - x)
            if i >= discard_iter:
                A_plot.append(A)
                x_plot.append(x)
                
    return np.array(A_plot), np.array(x_plot)

def calculate_lyapunov_exponent(A=4.0, x0=0.4, n_iter=1000):
    """
    Calculates the finite-time Lyapunov exponent for the logistic map.
    Returns an array of the exponent values over the iterations.
    """
    x = x0
    lyapunov_vals = []
    sum_log_deriv = 0.0
    
    for n in range(1, n_iter + 1):
        # f'(x) = A * (1 - 2x)
        deriv = A * (1 - 2 * x)
        sum_log_deriv += np.log(np.abs(deriv))
        lyapunov_vals.append(sum_log_deriv / n)
        x = A * x * (1 - x)
        
    return np.array(lyapunov_vals)