import numpy as np
import matplotlib.pyplot as plt

# 1. Generate simple linear data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Flatten X and y for simplicity
X = X.flatten()
y = y.flatten()

# 2. Initialize parameters
m = 0     # slope
c = 0     # intercept
alpha = 0.1  # learning rate
epochs = 25

n = len(X)
cost_history = []

# 3. Gradient Descent Algorithm
for epoch in range(epochs):
    y_pred = m * X + c
    error = y_pred - y

    cost = (1/n) * np.sum(error**2)
    cost_history.append(cost)

    dm = (2/n) * np.dot(error, X)
    dc = (2/n) * np.sum(error)

    m -= alpha * dm
    c -= alpha * dc

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: m={m:.4f}, c={c:.4f}, cost={cost:.4f}")

print(f"\nFinal parameters: m = {m:.4f}, c = {c:.4f}")

# 4. Plot Regression Line
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, m * X + c, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit using Gradient Descent')
plt.legend()

# 5. Plot Cost Function
plt.subplot(1, 2, 2)
plt.plot(range(epochs), cost_history, color='green')
plt.xlabel('Epoch')
plt.ylabel('Cost (MSE)')
plt.title('Cost Reduction over Iterations')

plt.tight_layout()
plt.show()
