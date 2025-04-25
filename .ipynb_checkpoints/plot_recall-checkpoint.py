import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_recall_by_dataset(tp_all, total_true_dict_per_model, title="Recall by Protein Group and Dataset", filename = None):

    datasets = list(tp_all.keys())
    models = sorted(set(m for ds in tp_all.values() for m in ds.keys()))
    groups = ['Strong', 'Weak', 'None']
    group_keys = ['strong', 'weak', 'none']

    # Full names for legend
    model_fullnames = {
        'IR': 'Isotonic Regression',
        '4PL': '4-Parameter Logistic',
        'FNN': 'Feedforward Neural Network'
    }
    colors = {
        'IR': '#66c2a5',
        '4PL': '#fc8d62',
        'FNN': '#8da0cb'
    }

   # fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 8), sharey=True)

    axes = axes.flatten()

    width = 0.2
    model_offsets = {'IR': -width, '4PL': 0, 'FNN': width}

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        ax.set_title(dataset, fontsize=14, pad=10)
        x = np.arange(len(groups))
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=12)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([f"{y:.0%}" for y in [0, 0.25, 0.5, 0.75, 1.0]], fontsize=11)
        if i in [0, 3]:
            ax.set_ylabel("Recall", fontsize=13)
        else:
            ax.set_yticklabels([])

        for model in models:
            if model not in tp_all[dataset]:
                continue
            recalls = []
            for gk in group_keys:
                tp = tp_all[dataset][model].get(gk, 0)
                total = total_true_dict_per_model[model].get(gk, 1)
                recall = tp / total if total > 0 else 0
                recalls.append(recall)

            bar_pos = x + model_offsets[model]
            bars = ax.bar(bar_pos, recalls, width=width, color=colors[model],
                          label=model_fullnames[model] if i == 0 else "")

            # Optional: annotate values on bars
            for bar, val in zip(bars, recalls):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                            ha='center', va='bottom', fontsize=10)


    # Create unified legend with full method names
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Method",
        loc='upper center',
        ncol=3,
        frameon=False,
        fontsize=12,
        title_fontsize=13,
        bbox_to_anchor=(0.5, 1.05)  # <--- Moves the legend down slightly
    )

    # Title pushed slightly higher
    fig.suptitle(title, fontsize=16, y=1.12)

    # Reserve more space for title + legend
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # keep this below everything else

   # fig.suptitle(title, fontsize=16, y=1.04)
    #plt.tight_layout(rect=[0, 0, 1, 0.98])

    if filename:
        fig.savefig(filename, format = "pdf")
        print(f"Figure saved to: {filename}")

    plt.show()
