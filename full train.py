# train_mnist_svm.py
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import joblib
import time

# Load MNIST data (this will download if you don't have it)
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Convert labels to integers
y = y.astype(np.uint8)

# Split data (MNIST is already pre-shuffled)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# Scale features to [0, 1] (better than StandardScaler for MNIST)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train SVM with optimal parameters for MNIST
print("Training SVM...")
start_time = time.time()

model = SVC(
    kernel='rbf',
    C=5,  # Regularization parameter
    gamma='scale',  # Kernel coefficient
    probability=True,  # Enable predict_proba
    random_state=42,
    verbose=True  # Show training progress
)

model.fit(X_train, y_train)

print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Save model and scaler
joblib.dump(model, 'mnist_svm_model.joblib')
print("Model saved to mnist_svm_model.joblib")

# Sample predictions
print("\nSample predictions:")
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)
    pred = model.predict(sample)[0]
    actual = y_test[idx]
    print(f"Predicted: {pred}, Actual: {actual}{' ✓' if pred == actual else ' ✗'}")