import numpy as np
import matplotlib.pyplot as plt
from core import RoboticArmOptimizer

arm = RoboticArmOptimizer(link_lengths=[1.0, 1.0, 1.0], target_pos=[-0.8, 1.2])
theta_start = np.array([0.2, 0.1, -0.3])

print("Running Vanilla Gradient Descent...")
p_hist_gd, theta_gd = arm.optimize_vanilla_gd(theta_start)

print("Running Gradient Descent with Momentum...")
p_hist_mom, theta_mom = arm.optimize_momentum_gd(theta_start)

print(f"Vanilla GD Steps: {len(p_hist_gd) - 1}")
print(f"Momentum GD Steps: {len(p_hist_mom) - 1}")

plt.figure(figsize=(10, 8))
plt.style.use('seaborn-v0_8-darkgrid')

plt.plot(p_hist_gd[:, 0], p_hist_gd[:, 1], 'o-', label=f"Gradient Descent ({len(p_hist_gd)-1} steps)", markersize=3, alpha=0.7)
plt.plot(p_hist_mom[:, 0], p_hist_mom[:, 1], 'o-', label=f"GD with Momentum ({len(p_hist_mom)-1} steps)", markersize=3, alpha=0.7)

# Markers for start, target, and end positions
plt.plot(arm.target[0], arm.target[1], 'rx', markersize=15, markeredgewidth=3, label='Target')
plt.plot(p_hist_gd[0, 0], p_hist_gd[0, 1], 'ks', markersize=10, label='Start')
plt.plot(p_hist_gd[-1, 0], p_hist_gd[-1, 1], 'go', markersize=10, label='End (GD)')
plt.plot(p_hist_mom[-1, 0], p_hist_mom[-1, 1], 'bo', markersize=10, label='End (Momentum)', alpha=0.6)

plt.title('Path of Robotic Arm End-Hook: Inverse Kinematics Optimization')
plt.xlabel('x-position (m)')
plt.ylabel('y-position (m)')
plt.legend(fontsize=12)
plt.axis('equal')
plt.savefig('imgs/arm_optimization.png', dpi=300)
plt.show()