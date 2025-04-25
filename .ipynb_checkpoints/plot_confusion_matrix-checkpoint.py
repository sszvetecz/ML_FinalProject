import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Plot a confusion matrix using only matplotlib.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - class_names: list of class names (optional)
    - title: title of the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    # Add counts inside the boxes
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], va='center', ha='center', color='black', fontsize=12)

    plt.tight_layout()
    plt.show()
