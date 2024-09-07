import mlpModule
import numpy as np
from matplotlib import pyplot as plt

# Initialize MLP and Backpropagation
myMLP = mlpModule.MLP(2, 8, 1)
myBack = mlpModule.Backpropagation(myMLP, 0.3, 0.001)

# Training the MLP
print("Training MLP...")
for i in range(5000):
    myBack.iterate(
        [[0,0],[0,1],[1,0],[1,1], [0.5, 0.5], [0.75, 0.5], [0.3, 0.5], [0.45, 0.2], [0.2, 0.7]],
        [[0],[1],[1],[0],[0],[1],[1],[0],[1]]
    )
print("Training completed.")

# Test the MLP
# Test the MLP with XOR inputs
print("Testing with XOR inputs:")
print(f"Input [0,0]: {myMLP.compute([0,0])}")
print(f"Input [0,1]: {myMLP.compute([0,1])}")
print(f"Input [1,0]: {myMLP.compute([1,0])}")
print(f"Input [1,1]: {myMLP.compute([1,1])}")

# Test the MLP with additional example inputs
print("Testing with additional example inputs:")
print(f"Input [0.5,0.5]: {myMLP.compute([0.5,0.5])}")
print(f"Input [0.75,0.5]: {myMLP.compute([0.75,0.5])}")
print(f"Input [0.3,0.5]: {myMLP.compute([0.3,0.5])}")
print(f"Input [0.45,0.2]: {myMLP.compute([0.45,0.2])}")
print(f"Input [0.2,0.7]: {myMLP.compute([0.2,0.7])}")
print("------------------------------")

# Plot decision boundary
x = np.arange(0, 1.0, 0.1)
y = np.arange(0, 1.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.array([[myMLP.compute2Arg(xi, yi) for xi in x] for yi in y])

plt.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='RdBu', alpha=0.6)

# Remove these lines to not show contour and colorbar
# plt.colorbar()

# plt.contour(X, Y, Z, levels=np.arange(0, 1, 0.2), linewidths=2, cmap='Set2')
# plt.clabel(contour_set, inline=True, fmt='%1.1f', fontsize=10)

plt.title('Decision Boundary')
plt.show()