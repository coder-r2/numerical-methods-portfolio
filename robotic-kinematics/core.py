import numpy as np
import math

class RoboticArmOptimizer:
    def __init__(self, link_lengths=[1.0, 1.0, 1.0], target_pos=[-0.8, 1.2]):
        """Initializes the robotic arm environment."""
        self.L = link_lengths
        self.target = np.array(target_pos)
        self.h = 1e-5

    def forward_kinematics(self, theta):
        """Calculates the (x, y) position of the end-hook given joint angles."""
        x = (self.L[0] * math.cos(theta[0]) + 
             self.L[1] * math.cos(theta[0] + theta[1]) + 
             self.L[2] * math.cos(theta[0] + theta[1] + theta[2]))
        
        y = (self.L[0] * math.sin(theta[0]) + 
             self.L[1] * math.sin(theta[0] + theta[1]) + 
             self.L[2] * math.sin(theta[0] + theta[1] + theta[2]))
        
        return np.array([x, y])

    def loss(self, theta):
        """Calculates the squared Euclidean distance to the target."""
        p = self.forward_kinematics(theta)
        return 0.5 * np.sum((p - self.target)**2)

    def get_gradient(self, theta):
        """Computes the numerical gradient using the central difference method."""
        grad = np.zeros(3)
        for i in range(3):
            theta_plus = np.copy(theta)
            theta_minus = np.copy(theta)
            
            theta_plus[i] += self.h
            theta_minus[i] -= self.h
            
            grad[i] = (self.loss(theta_plus) - self.loss(theta_minus)) / (2 * self.h)
        return grad

    def optimize_vanilla_gd(self, theta_0, alpha=0.01, stop_dist=0.01, max_iter=2000):
        """Performs Vanilla Gradient Descent to find target joint angles."""
        theta = np.copy(theta_0)
        p_history = [self.forward_kinematics(theta)]
        
        for _ in range(max_iter):
            p = self.forward_kinematics(theta)
            if np.linalg.norm(p - self.target) < stop_dist:
                break
                
            grad = self.get_gradient(theta)
            theta -= alpha * grad
            p_history.append(self.forward_kinematics(theta))
            
        return np.array(p_history), theta

    def optimize_momentum_gd(self, theta_0, alpha=0.01, beta=0.9, stop_dist=0.01, max_iter=2000):
        """Performs Gradient Descent with Momentum."""
        theta = np.copy(theta_0)
        v = np.zeros(3)
        p_history = [self.forward_kinematics(theta)]
        
        for _ in range(max_iter):
            p = self.forward_kinematics(theta)
            if np.linalg.norm(p - self.target) < stop_dist:
                break
                
            grad = self.get_gradient(theta)
            v = beta * v + alpha * grad
            theta -= v
            p_history.append(self.forward_kinematics(theta))
            
        return np.array(p_history), theta