import numpy as np
from sklearn.metrics import balanced_accuracy_score

rng = np.random.default_rng(42)
mu = np.array([2, 4])
Sigma = np.array([[1, 0.5], [0.5, 1]])
L = np.linalg.cholesky(Sigma)

n_inliers, n_outliers, n_features = 900, 100, 2
X_inliers = rng.standard_normal(size=(n_inliers, n_features))
Y_inliers = X_inliers @ L.T + mu

k = 9
L_out = np.sqrt(k) * L
X_outliers = rng.standard_normal(size=(n_outliers, n_features))
Y_outliers = X_outliers @ L_out.T + mu

Y_total = np.vstack((Y_inliers, Y_outliers))
y_true = np.hstack((np.zeros(n_inliers), np.ones(n_outliers)))

mu_hat = np.mean(Y_total, axis=0)
Sigma_hat = np.cov(Y_total, rowvar=False)
Y_centered = Y_total - mu_hat
z_scores = np.sqrt(np.sum((Y_centered @ np.linalg.inv(Sigma_hat)) * Y_centered, axis=1))
z_threshold = np.quantile(z_scores, 1 - n_outliers / (n_inliers + n_outliers))
y_pred = (z_scores > z_threshold).astype(int)

print(f"Estimated mean: {mu_hat}")
print(f"Estimated covariance:\n{Sigma_hat}")
print(f"Z-Score Threshold: {z_threshold:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")