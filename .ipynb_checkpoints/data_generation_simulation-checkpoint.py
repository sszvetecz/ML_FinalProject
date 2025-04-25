import numpy as np
import pandas as pd

def simulate_chemo_proteinlevel_nonparametric(
    N_proteins=100,
    TP=0.1,
    TW=0.2,
    TN=0.7,
    concentrations=[0, 1, 3, 10, 30, 100, 300, 1000, 3000],
    rep=3,
    seed=3,
    var_tech=0.275,
    template=None,
    outlier_prob=0.05
):
    # Validate concentrations
    valid_concentrations = [0, 1, 3, 10, 30, 100, 300, 1000, 3000]
    if not all(c in valid_concentrations for c in concentrations):
        raise ValueError("`concentrations` must be a subset of the valid list.")

    # Subset template data
    def subset_template(df, concentrations):
        return df[df['dose'].isin(concentrations)].copy()

    if template is None:
        template = {
            "strong_interaction": pd.DataFrame(columns=["dose", "LogIntensities"]),
            "weak_interaction": pd.DataFrame(columns=["dose", "LogIntensities"]),
            "no_interaction": pd.DataFrame(columns=["dose", "LogIntensities"])
        }

    template = {k: subset_template(v, concentrations) for k, v in template.items()}

    np.random.seed(seed)

    # Calculate numbers
    tp = int(np.ceil(N_proteins * TP))
    tw = int(np.ceil(N_proteins * TW))
    tn = int(np.ceil(N_proteins * TN))

    def simulate_proteins(n, interaction_type, var_tech, template_data, rep):
        if n == 0 or template_data.empty:
            return pd.DataFrame()
        results = []
        for i in range(1, n + 1):
            df = pd.DataFrame({
                "dose": np.repeat(template_data["dose"].values, rep),
                "replicate": np.tile(np.arange(1, rep + 1), len(template_data))
            })
            base_logint = np.repeat(template_data["LogIntensities"].values, rep)
            noise = np.random.normal(loc=0, scale=var_tech, size=len(base_logint))
            df["log_intensity"] = base_logint + noise
            df["protein"] = f"p_{interaction_type}_{i}"
            df["protein_group"] = interaction_type
            results.append(df)
        return pd.concat(results, ignore_index=True)

    sim_tp = simulate_proteins(tp, "strong_interaction", var_tech, template["strong_interaction"], rep)
    sim_tw = simulate_proteins(tw, "weak_interaction", var_tech, template["weak_interaction"], rep)
    sim_tn = simulate_proteins(tn, "no_interaction", var_tech, template["no_interaction"], rep)

    data = pd.concat([sim_tp, sim_tw, sim_tn], ignore_index=True)
    data["sample_group"] = np.where(data["dose"] == 0, "DMSO", "treated")

    def add_outliers(df, outlier_prob):
        n = len(df)
        n_outliers = int(np.ceil(n * outlier_prob))
        outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)

        # Ensure at least one outlier in strong/weak groups
        for group in ["strong_interaction", "weak_interaction"]:
            group_idx = df[df["protein_group"] == group].index
            if len(group_idx) > 0:
                outlier_indices = np.unique(np.append(outlier_indices, np.random.choice(group_idx, 1)))

        df["outlier"] = False
        offsets = np.random.lognormal(mean=0, sigma=1, size=len(outlier_indices))
        signs = np.random.choice([-1, 1], size=len(outlier_indices))
        df.loc[outlier_indices, "log_intensity"] += offsets * signs
        df.loc[outlier_indices, "outlier"] = True
        return df

    data = add_outliers(data, outlier_prob)
    return data
