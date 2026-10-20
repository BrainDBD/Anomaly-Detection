from pyod.utils.data import generate_data
import matplotlib.pyplot as plt

contamination_rate = 0.1
X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=contamination_rate, random_state=42)
inliers = X_train[y_train == 0]
outliers = X_train[y_train == 1]

plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', label='Nominal Points')
plt.scatter(outliers[:, 0], outliers[:, 1], marker='x', c='red', label='Outliers')
plt.title(f'Generated Training Dataset with Contamination = {contamination_rate}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(len(inliers), len(outliers))