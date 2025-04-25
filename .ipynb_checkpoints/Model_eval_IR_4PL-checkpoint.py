import numpy as np
import pandas as pd
from scipy import stats

from FourPL_model_fit import four_pl_fit_logscale
from IR_model_fit_logscale import isotonic_regression_fit_logscale

def evaluate_model_fit(data, method="IR"):
    """
    Evaluate model fit vs. null model for each protein using the specified method.

    input:
    - data: pandas DataFrame with columns ['protein', 'dose', 'log_intensity', 'sample_group']
    - method: "IR" for isotonic regression, or "Four_PL" for 4-parameter logistic model

    output:
    - model_res_all: DataFrame with p-values and adjusted p-values
    """
    
    protein_list = np.unique(data['protein'])
    pvalues = []

    for protein in protein_list:
        if method == "IR":
            fit = isotonic_regression_fit_logscale(data, protein)[0]
        elif method == "Four_PL":
            fit = four_pl_fit_logscale(data, protein)
        else:
            raise ValueError("Method must be either 'IR' or 'Four_PL'")
        pvalues.append(fit[3])  

    # Build result DataFrame
    model_res_all = pd.DataFrame({
        "protein": protein_list,
        "p-value": pvalues
    }).sort_values(by="p-value")

    # Adjust p-values using FDR
    #model_res_all["adj_pval"] = fdrcorrection(model_res_all["p-value"])[1]
    # FDR pvalue adjustment 
    model_res_all['adj_pval'] = stats.false_discovery_control(
        model_res_all['p-value'], method='bh')

    # Summary stats
    # Total number of sig proteins 
    p_sig_unadjust = len(model_res_all[model_res_all['p-value'] < .05])
    p_sig_fdr_adjust = len(model_res_all[model_res_all['adj_pval'] < .05])

    # count strong, weak , no interaction sig hits 
    n_strong_sig = len(model_res_all[
                       (model_res_all['adj_pval'] < 0.05) & 
                       (model_res_all['protein'].str.contains('strong'))])
    n_weak_sig = len(model_res_all[
                     (model_res_all['adj_pval'] < 0.05) & 
                     (model_res_all['protein'].str.contains('weak'))])
    n_no_sig = len(model_res_all[
                   (model_res_all['adj_pval'] < 0.05) & 
                   (model_res_all['protein'].str.contains('no'))])
    
    print("Number of sig proteins (p < .05):", p_sig_unadjust)
    print("Number of sig proteins with FDR adjustment (p < .05):", p_sig_fdr_adjust)
    print("Strong interaction hits:", n_strong_sig)
    print("Weak interaction hits:", n_weak_sig)
    print("No interaction hits:", n_no_sig)

    return model_res_all
