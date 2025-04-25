import matplotlib.pyplot as plt
import numpy as np
from FourPL_model_fit import four_pl_fit_logscale
from IR_model_fit_logscale import isotonic_regression_fit_logscale

def plot_simulation_curve(df, protein_name, plot_type="log_intensity", title="Dose-Response Curve",
                          fit_model=None, fit_color="blue"):
    """
    Plot a simulated dose-response curve for a single protein using matplotlib,
    with optional overlay of fitted model (FourPL or IR).

    Parameters:
    - df: pandas DataFrame
    - protein_name: str, name of the protein to plot
    - plot_type: str, 'log_intensity' or 'ratio'
    - title: str, plot title
    - fit_model: 'FourPL', 'IR', or None
    - fit_color: str, color for fitted curve
    """
    
    # Filter data
    data = df[df['protein'] == protein_name].copy()
    if data.empty:
        print(f"No data found for protein '{protein_name}'")
        return

    # Calculate response value
    if plot_type == "ratio":
        dmso_mean = data[data['sample_group'] == "DMSO"]['intensity'].mean()
        data['y'] = data['intensity'] / dmso_mean
        ylabel = "Relative Intensity (to DMSO)"
    elif plot_type == "log_intensity":
        data['y'] = data['log_intensity']
        ylabel = "Log2(Intensity)"
    else:
        raise ValueError("plot_type must be either 'log_intensity' or 'ratio'")

    # Add transformed dose axis (log10 + 1)
    data['dose_trans'] = np.log10(data['dose'] + 1)

    # Create tick labels
    dose_ticks = sorted(data['dose'].unique())
    tick_labels = ['DMSO' if d == 0 else f"{int(d)}nM" for d in dose_ticks]
    dose_tick_positions = np.log10(np.array(dose_ticks) + 1)

    # Plot mean line
    grouped = data.groupby('dose_trans')['y']
    mean_x = grouped.mean().index.values
    mean_y = grouped.mean().values

    plt.figure(figsize=(8, 5))
    plt.plot(mean_x, mean_y, '-o', label='Mean Response', color='black')

    # Plot individual replicates with outlier highlight
    for _, row in data.iterrows():
        color = 'orange' if row['outlier'] else 'gray'
        plt.scatter(row['dose_trans'], row['y'], color=color, s=30, alpha=0.8)

    # Overlay model fit
    if fit_model == "FourPL":
        try:
            fit = four_pl_fit_logscale(df, protein_name)
            model = fit[-1]  # lmfit result
            x_fit = np.linspace(min(data['dose']), max(data['dose']), 200)
            x_trans = np.log10(x_fit + 1)
            y_pred = model.eval(logx=np.log10(x_fit))
            plt.plot(np.log10(x_fit + 1), y_pred, color=fit_color, linestyle='--', label='FourPL Fit')
        except Exception as e:
            print(f"FourPL fit failed: {e}")
    elif fit_model == "IR":
        try:
            x = np.log10(data['dose'] + 1)
            y = data['y']
            #order = np.argsort(x)
            iso_fit = isotonic_regression_fit_logscale(df, protein_name)
            plt.plot(x, iso_fit[1], color=fit_color, linestyle='--', label='Isotonic Fit')
        except Exception as e:
            print(f"IR fit failed: {e}")

    plt.xticks(ticks=dose_tick_positions, labels=tick_labels)
    plt.xlabel("Dose (log10 + 1 transformed")
    plt.ylabel(ylabel)
    plt.ylim(0)
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
