import numpy as np
import pandas as pd
from lmfit import Model
from scipy.stats import f
from sklearn.metrics import mean_squared_error

def four_param_log_logistic(logx, a, b, c, d):
    return d + (a - d) / (1.0 + (10**(logx - c))**b)

def four_pl_fit_logscale(data, protein_name):
    """
    Perform goodness of fit test for 4-parameter logistic model compared to null model.
    
    Parameters:
    - data: pandas DataFrame containing the dataset
    - protein_name: string, name of the protein to filter
    
    Returns:
    - mse_4pl: mean squared error for 4PL fit
    - mse_null: mean squared error for null model (flat response)
    - f_value: f-statistic comparing 4PL to null model
    - p_value: p-value for the F-statistic
    - fit_result: lmfit result object for further inspection (optional)
    """
    
    # Unlog intensities
    data['intensity'] = 2 ** data['log_intensity']

    # Filter to single protein
    protein_temp = data[data['protein'] == protein_name].copy()

    # Calculate average DMSO intensity
    avg_dmso = (
        protein_temp[protein_temp['sample_group'] == 'DMSO']
        .groupby('sample_group')
        .agg(Mean=('intensity', 'mean'))
    ).loc['DMSO'].values[0]

    # Calculate relative intensity
    protein_temp['Relative_Intensity'] = protein_temp['intensity'] / avg_dmso

    # Prepare log10 dose and response
    x = np.log10(protein_temp['dose'].values + 1)
    y = protein_temp['Relative_Intensity'].values

    # Remove -inf (from log10(0) if present)
    valid_idx = np.isfinite(x)
    x = x[valid_idx]
    y = y[valid_idx]

    # Define and fit the 4PL model
    model = Model(four_param_log_logistic)
    params = model.make_params(a=1.0, b=1.0, c=0.0, d=0.0)
    params['a'].set(min=0, max=2)
    params['b'].set(min=0.1, max=10)
    params['c'].set(min=-2, max=3)
    params['d'].set(min=0, max=1)

    result = model.fit(y, params, logx=x)
    y_pred = result.best_fit

    # MSE for fitted model
    mse_4pl = mean_squared_error(y, y_pred)

    # Null model: horizontal line at mean of y
    y_null = np.full_like(y, np.mean(y))
    mse_null = mean_squared_error(y, y_null)

    # F-test
    rss_fit = np.sum((y - y_pred)**2)
    rss_null = np.sum((y - y_null)**2)
    df_fit = len(y) - result.nvarys
    df_null = len(y) - 1
    f_stat = ((rss_null - rss_fit) / (df_null - df_fit)) / (rss_fit / df_fit)
    p_value = 1 - f.cdf(f_stat, df_null - df_fit, df_fit)

    return mse_4pl, mse_null, f_stat, p_value, result
