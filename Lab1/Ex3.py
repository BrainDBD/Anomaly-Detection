from pyod.utils.data import generate_data
import numpy as np
from sklearn.metrics import balanced_accuracy_score

cont_rate = 0.1
X_train, _, y_train, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=cont_rate, random_state=42)
x = X_train.ravel()
z_scores = np.abs((x - np.mean(x)) / np.std(x))
z_threshold = np.quantile(z_scores, 1 - cont_rate)
y_pred = (z_scores > z_threshold).astype(int)
print(f"Z-Score Threshold: {z_threshold:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_train, y_pred):.4f}")