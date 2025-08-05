import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

# 1. Hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 3 * 32 * 32
HIDDEN_SIZES = [512, 256, 128]
NUM_CLASSES = 10

# 2. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Definition
class FeedForwardNet(nn.Module):
    def __init__(self, weight_init=None, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.fc3 = nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2])
        self.fc4 = nn.Linear(HIDDEN_SIZES[2], NUM_CLASSES)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        if weight_init is not None:
            self.apply(weight_init)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# 4. Initialization Functions
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 5. Training & Evaluation Functions
def train_model(model, train_loader, test_loader, epochs, lr, l2=0.0):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    stats = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate_model(model, test_loader)
        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['test_loss'].append(test_loss)
        stats['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{epochs}:",
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},",
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return stats

def evaluate_model(model, loader):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

# 6. Experiment Helper
def run_experiment(label, model_kwargs, l2=0.0):
    print(f"\n*** Running Experiment: {label} ***")
    model = FeedForwardNet(**model_kwargs)
    stats = train_model(model, train_loader, test_loader, EPOCHS, LR, l2)
    return stats

# 7. Run all experiments and collect results
results = {}

# (a) Baseline: Default initialization, no regularization
results['baseline'] = run_experiment('Baseline', {'weight_init': None, 'dropout': 0.0})

# (b) Xavier initialization
results['xavier'] = run_experiment('Xavier Initialization', {'weight_init': xavier_init, 'dropout': 0.0})

# (c) Kaiming initialization
results['kaiming'] = run_experiment('Kaiming Initialization', {'weight_init': kaiming_init, 'dropout': 0.0})

# (d) Dropout regularization (p=0.5)
results['dropout'] = run_experiment('Dropout (p=0.5)', {'weight_init': None, 'dropout': 0.5})

# (e) L2 Regularization (weight_decay in Adam)
results['l2'] = run_experiment('L2 Regularization (weight_decay=1e-4)', {'weight_init': None, 'dropout': 0.0}, l2=1e-4)

# 8. Plotting Results
def plot_results(results):
    plt.figure(figsize=(14, 6))
    for key, stat in results.items():
        plt.plot(stat['test_acc'], label=f'{key} Test Acc')
    plt.title('Test Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 6))
    for key, stat in results.items():
        plt.plot(stat['test_loss'], label=f'{key} Test Loss')
    plt.title('Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(results)

# 9. Analysis Boilerplate
def print_analysis(results):
    print("\n=== Summary Table (Final Epoch) ===")
    print(f"{'Setting':<15} {'Test Acc':<8} {'Test Loss':<10}")
    for key, stat in results.items():
        print(f"{key:<15} {stat['test_acc'][-1]:.4f}   {stat['test_loss'][-1]:.4f}")

    print("\n=== Analysis ===")
    print("a) Xavier initialization: Compare the curve and final accuracy/loss with baseline. Typically, Xavier helps convergence for tanh/linear activations, and can improve both speed and stability.")
    print("b) Kaiming initialization: Designed for ReLU, often aids convergence and prevents vanishing/exploding gradients, especially in deeper nets. Compare with baseline.")
    print("c) Dropout: Observe if test accuracy is higher and test/train gap is smaller, indicating less overfitting. Dropout often slows convergence but improves generalization.")
    print("d) L2 Regularization: Test if accuracy is higher and overfitting is reduced. L2 can help prevent weights from growing and force the model to generalize better.")

print_analysis(results)

# 10. Best Model Performance
def find_best_model(results):
    print("\n" + "="*60)
    print("BEST MODEL PERFORMANCE ANALYSIS")
    print("="*60)

    # Find best model based on final test accuracy
    best_accuracy = 0
    best_model = ""
    best_stats = None

    print("\nFinal Performance Comparison:")
    print("-" * 50)
    for key, stats in results.items():
        final_acc = stats['test_acc'][-1]
        final_loss = stats['test_loss'][-1]
        print(f"{key:<20}: Test Acc = {final_acc:.4f}, Test Loss = {final_loss:.4f}")

        if final_acc > best_accuracy:
            best_accuracy = final_acc
            best_model = key
            best_stats = stats

    print("\n" + "="*50)
    print(f"🏆 BEST PERFORMING MODEL: {best_model.upper()}")
    print("="*50)
    print(f"Final Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Final Test Loss: {best_stats['test_loss'][-1]:.4f}")
    print(f"Final Train Accuracy: {best_stats['train_acc'][-1]:.4f} ({best_stats['train_acc'][-1]*100:.2f}%)")
    print(f"Final Train Loss: {best_stats['train_loss'][-1]:.4f}")

    # Calculate overfitting metric (difference between train and test accuracy)
    overfitting = best_stats['train_acc'][-1] - best_stats['test_acc'][-1]
    print(f"Overfitting Gap: {overfitting:.4f} ({overfitting*100:.2f}%)")

    # Find best epoch performance
    best_epoch_acc = max(best_stats['test_acc'])
    best_epoch_idx = best_stats['test_acc'].index(best_epoch_acc)
    print(f"Best Test Accuracy: {best_epoch_acc:.4f} at Epoch {best_epoch_idx + 1}")

    print("\n" + "="*50)
    print("PERFORMANCE INSIGHTS:")
    print("="*50)

    if overfitting > 0.05:  # 5% gap
        print("⚠️  High overfitting detected - consider more regularization")
    elif overfitting < 0.02:  # 2% gap
        print("✅ Good generalization - low overfitting")
    else:
        print("✅ Moderate overfitting - acceptable performance")

    # Compare with baseline
    if best_model != 'baseline':
        baseline_acc = results['baseline']['test_acc'][-1]
        improvement = (best_accuracy - baseline_acc) * 100
        print(f"📈 Improvement over baseline: +{improvement:.2f}%")

    return best_model, best_stats

find_best_model(results)