import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

def plot_isotonic_regression_fit(data, protein_name):
    """
    Perform isotonic regression, calculate predicted values, and plot results.
    
    Parameters:
    - data: pandas DataFrame containing the dataset
    - protein_name: string, name of the protein to filter
    
    Returns:
    - x_preds: np.ndarray, unique x values used for prediction
    - y_pred: np.ndarray, predicted y values
    """
    # Unlog intensities
    data['intensity'] = 2 ** data['log_intensity']
    
    # Filter to single protein
    protein_temp = data[data['protein'] == protein_name]
    
    # Calculate average DMSO intensity
    avg_dmso_protein_temp = (
        protein_temp[protein_temp['sample_group'] == 'DMSO']
        .groupby('sample_group')
        .agg(Mean=('intensity', 'mean'))
    )
    
    # Calculate relative intensity values
    protein_temp = protein_temp.copy()  # Avoid modifying the original DataFrame
    protein_temp['Relative_Intensity'] = (
        protein_temp['intensity'] / avg_dmso_protein_temp['Mean'].iloc[0]
    )
    
    # Prepare x and y for isotonic regression
    x = np.log10(protein_temp['dose'] + 1)
    y = protein_temp['Relative_Intensity']
    
    # Fit the isotonic regression model
    ir = IsotonicRegression(increasing=False)
    ir.fit(x, y)
    
    # Predict new values
    x_preds = np.unique(x)
    y_pred = ir.predict(x_preds)
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o', label="Observed Data", color='blue')  # Observed data points
    plt.plot(x_preds, y_pred, '-', label="Isotonic Regression Fit", color='orange')  # Fit line
    plt.ylim(bottom=0)
    
    # Customize the plot
    plt.xscale('linear')  # Keep x-axis linear since we log-transformed dose
    plt.xticks(x_preds, ['DMSO', '1nM', '3nM', '10nM','30nM','100nM', '300nM',
                        '1000nM','3000nM'])
    plt.xlabel('Drug Concentration (nM)', fontsize=12)
    plt.ylabel('Relative Intensity', fontsize=12)
    plt.title(f'Isotonic Regression Fit for {protein_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    # Return predicted values
    return x_preds, y_pred



