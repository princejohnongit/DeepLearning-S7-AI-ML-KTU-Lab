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
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Reds',
            xticklabels=target_names, yticklabels=target_names,color="Red")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("SVM Confusion Matrix (Linear Kernel)")
plt.show()

# 8. Plot SVM decision boundaries for the first two features

def plot_svm_decision_boundary(clf, X, y, title):
    # Only use the first two features for 2D visualization
    X = X[:, :2]
    y = y
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict for each point in mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(),
                          np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names)
    plt.show()

# Plot using training data (first two features)
plot_svm_decision_boundary(svm_clf, X_train_scaled, y_train, "SVM Decision Boundary (Train, first 2 features)")
# Plot using test data (first two features)
plot_svm_decision_boundary(svm_clf, X_test_scaled, y_test, "SVM Decision Boundary (Test, first 2 features)")
