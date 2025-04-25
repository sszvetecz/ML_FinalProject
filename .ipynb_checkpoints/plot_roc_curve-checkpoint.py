from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch 

def plot_roc_curve(model, X_train, y_train, X_test, y_test):
    """
    Plot ROC curves and print AUC for training and test sets.

    Parameters:
    - model: trained PyTorch model
    - X_train, X_test: input data as NumPy arrays
    - y_train, y_test: binary labels (0 or 1)
    """
    model.eval()
    with torch.no_grad():
        train_probs = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
        test_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(train_fpr, train_tpr, label=f"Train AUC = {train_auc:.3f}")
    plt.plot(test_fpr, test_tpr, label=f"Test AUC = {test_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return train_auc, test_auc
