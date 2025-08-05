from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Linear SVM classifier
svm_clf = SVC(kernel='linear', C=1.0, random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = svm_clf.predict(X_test_scaled)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# 7. Visualize the Confusion Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Reds',
            xticklabels=target_names, yticklabels=target_names, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("SVM Confusion Matrix (Linear Kernel)")

# 8. Visualize SVM Decision Boundary (using first 2 features)
plt.subplot(1, 2, 2)

# Use only first 2 features for 2D visualization
X_2d = X[:, :2]  # sepal length and sepal width
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)

# Scale the 2D data
scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

# Train SVM on 2D data
svm_2d = SVC(kernel='linear', C=1.0, random_state=42)
svm_2d.fit(X_train_2d_scaled, y_train_2d)

# Create a mesh for plotting decision boundary
import numpy as np
h = 0.02  # step size in the mesh
x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predictions on the mesh
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

# Plot the data points
colors = ['white', 'blue', 'orange']
for i, color in enumerate(colors):
    idx = np.where(y_train_2d == i)
    plt.scatter(X_train_2d_scaled[idx, 0], X_train_2d_scaled[idx, 1], 
                c=color, marker='o', label=f'{target_names[i]} (train)', s=50, alpha=0.8)

# Plot test points with different markers
for i, color in enumerate(colors):
    idx = np.where(y_test_2d == i)
    plt.scatter(X_test_2d_scaled[idx, 0], X_test_2d_scaled[idx, 1], 
                c=color, marker='s', label=f'{target_names[i]} (test)', s=60, alpha=1.0)

plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('SVM Decision Boundary (Linear Kernel)')
plt.legend()

plt.tight_layout()
plt.show()
