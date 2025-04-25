import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import mean_squared_error 

def isotonic_regression_fit_logscale(data, protein_name):
    """
    Perform goodness of fit test for Isotonic Regression compared to null model.
    
    Parameters:
    - data: pandas DataFrame containing the dataset
    - protein_name: string, name of the protein to filter
    
    Returns:
    - mse_ir: mean squared error for Isotonic Regression 
    - mse_ir_null: mean squared error for null model (i.e. average response) 
    - f_value: f-statistic comparing IR to null model 
    - p_value: f-stat model significane 
    
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

    mean_dmso = avg_dmso_protein_temp.loc['DMSO'].mean()
    # Calculate relative intensity values
    protein_temp = protein_temp.copy()  
    protein_temp['Relative_Intensity'] = (
            protein_temp['intensity'] / mean_dmso
        )
    
    # Prepare x and y for isotonic regression
    x = np.log10(protein_temp['dose'] + 1)
   # y = protein_temp['Relative_Intensity']
    y = protein_temp['log_intensity']

    # Fit the isotonic regression model
    ir = IsotonicRegression(increasing=False)
    ir.fit(x, y)
    
    # Predict new values
    #x_preds = np.unique(x)
    y_pred = ir.predict(x)
    
    
    # Fit the null regression model
    avg_y = np.repeat(np.mean(y),len(x))

    # calculate SSE 
    sse_null = np.sum((y - avg_y) ** 2)
    sse_full = np.sum((y - y_pred) ** 2)

    # Calculate the degrees of freedom
    df_full = len(y) - len(np.unique(x))
    df_null = len(y) - 1
    
    # calculate F-statistic 
    f_stat = ((sse_null - sse_full)/(df_null - df_full))/(sse_full/df_full)
    
    # Calculate the p-value
    p_value = stats.f.cdf(f_stat, df_null - df_full, df_null)
    
    res_stats = [sse_full, sse_null, f_stat, 1 - p_value]
    
    return(res_stats,y_pred)

    print("IR Model SSE:", sse_full)
    print("Null IR Model SSE:", sse_null)
    print("F-statistic:", f_stat)
    print("P-value:", 1 - p_value)

