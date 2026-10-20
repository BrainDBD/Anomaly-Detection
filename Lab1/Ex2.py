from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred, scores):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = TP / (TP + FN)
    tnr = TN / (TN + FP)
    balanced_accuracy = (tpr + tnr) / 2
    fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    return balanced_accuracy, (fpr_curve, tpr_curve), auc


X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=42)

contamination_rates = [0.01, 0.05, 0.1, 0.2]
fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
for ax, contamination_rate in zip(axes.flatten(), contamination_rates):
    model = KNN(contamination=contamination_rate)
    model.fit(X_train)

    train_scores = model.decision_scores_
    test_scores = model.decision_function(X_test)
    train_ba, (train_fpr, train_tpr), train_auc = compute_metrics(y_train, model.labels_, train_scores)
    test_ba, (test_fpr, test_tpr), test_auc = compute_metrics(y_test, model.predict(X_test), test_scores)

    print(f"KNN Contamination Rate: {contamination_rate}")
    print(f" Train Set: Balanced Accuracy = {train_ba:.3f} | AUC = {train_auc:.3f}")
    print(f"  Test Set: Balanced Accuracy = {test_ba:.3f} | AUC = {test_auc:.3f}")
    print()

    ax.plot(train_fpr, train_tpr, label=f"Train (AUC = {train_auc:.3f})")
    ax.plot(test_fpr, test_tpr, label=f"Test (AUC = {test_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"KNN Contamination = {contamination_rate}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

plt.show()