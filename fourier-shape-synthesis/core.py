import numpy as np
import subprocess
import os

class FourierSynthesizer:
    def __init__(self, coordinates):
        """
        Initializes the synthesizer with a list or array of complex coordinates.
        """
        self.z = np.array(coordinates, dtype=complex)
        self.N = len(self.z)
        self.t = np.linspace(0, 1, self.N, endpoint=False)

    @classmethod
    def generate_and_load(cls, letter, executable_path='./letter', output_csv=None):
        """
        Runs the external executable to generate the CSV, then loads the coordinates.
        """
        output_csv = f"{letter}.csv"

        try:
            print(f"Setting execution permissions for {executable_path}...")
            subprocess.run(['chmod', '+x', executable_path], check=True)

            clean_env = os.environ.copy()
            if 'MPLBACKEND' in clean_env:
                del clean_env['MPLBACKEND']
            
            print(f"Running executable to generate outline for '{letter}'...")
            subprocess.run([executable_path, letter], check=True, env=clean_env)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Could not find '{executable_path}'. Make sure the file is in the same folder as this script.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"The executable failed to run. Error: {e}")

        return cls.from_csv(output_csv)

    @classmethod
    def from_csv(cls, filepath):
        """Helper method to load (x, y) coordinates from a CSV file."""
        x_j, y_j = [], []
        with open(filepath, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    x, y = line.strip().split(',')
                    x_j.append(float(x))
                    y_j.append(float(y))
                except ValueError:
                    continue
        
        z_j = np.array(x_j) + 1j * np.array(y_j)
        return cls(z_j)

    def compute_coefficients(self, M):
        """
        Computes Fourier coefficients c_k for frequencies k in [-M, M].
        Utilizes the Trapezoidal rule, which exhibits spectral convergence for periodic functions.
        """
        c_k = {}
        h = 1.0 / self.N
        
        for k in range(-M, M + 1):
            f = self.z * np.exp(-1j * 2 * np.pi * k * self.t)
            c_k[k] = np.sum(f) * h
            
        return c_k

    def reconstruct_shape(self, M, num_points=1000):
        """
        Reconstructs the shape using a truncated Fourier series up to frequency M.
        """
        c_k = self.compute_coefficients(M)
        t_eval = np.linspace(0, 1, num_points)
        z_reconstructed = np.zeros(num_points, dtype=complex)
        
        for k in range(-M, M + 1):
            z_reconstructed += c_k[k] * np.exp(1j * 2 * np.pi * k * t_eval)
            
        return z_reconstructed