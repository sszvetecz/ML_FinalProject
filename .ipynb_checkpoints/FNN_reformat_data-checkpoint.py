import numpy as np 
import pandas as pd 

def prepare_fnn_dataset(df, response_type="log_intensity", label_type="binary"):
    """
    Prepare a dataset for FNN training. Each replicate is treated as an individual sample.

    Parameters:
    - df: input DataFrame
    - response_type: "log_intensity" or "ratio"
    - label_type: "binary" or "multiclass"

    Returns:
    - X: ndarray of shape (n_samples, 9)
    - y: ndarray of shape (n_samples,)
    """
    dose_levels = sorted(df['dose'].unique())
    X, y = [], []

    for protein in df['protein'].unique():
        protein_df = df[df['protein'] == protein].copy()

        if response_type == "log_intensity":
            feature_col = "log_intensity"
        else:
            protein_df['intensity'] = 2**protein_df['log_intensity']
            dmso_mean = protein_df[protein_df['sample_group'] == 'DMSO']['intensity'].mean()
            protein_df['ratio'] = protein_df['intensity'] / dmso_mean 
            feature_col = "ratio"

        # Group by replicate
        for rep in protein_df['replicate'].unique():
            rep_df = protein_df[protein_df['replicate'] == rep]
            vec = rep_df.set_index("dose").loc[dose_levels, feature_col].values

            if len(vec) != 9 or np.isnan(vec).any():
                continue  # skip incomplete

            label = protein_df['protein_group'].iloc[0]
            if label_type == "binary":
                y_val = 1 if label in ['strong_interaction', 'weak_interaction'] else 0
            else:
                y_val = {'no_interaction': 0, 'weak_interaction': 1, 'strong_interaction': 2}[label]

            X.append(vec)
            y.append(y_val)

    return np.array(X), np.array(y)
