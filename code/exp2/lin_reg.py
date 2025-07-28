import torch

import torch.nn as nn
import matplotlib.pyplot as plt

# Generate synthetic data: y = 2x + 3 + noise
torch.manual_seed(42)
X = torch.linspace(0, 10, 100).unsqueeze(1)
y = 2 * X + 3 + torch.randn(X.size()) * 2

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 40 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot results
predicted = model(X).detach()
plt.scatter(X.numpy(), y.numpy(), label='Original Data')
plt.plot(X.numpy(), predicted.numpy(), color='red', label='Fitted Line')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression with PyTorch')
plt.show()