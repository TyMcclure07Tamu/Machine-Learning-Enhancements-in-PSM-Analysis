import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import t

def placebo_results(df, bcs, trim_value, iv_flag):
    for w in range(1, 13):
        varname = f'lead2_pscore_wk{w}'
        group_var = f'{varname}_group'

        # Calculating min and max for treated and untreated
        min_treated = max(df.loc[df[f'lead2_win_wk{w}'] == 1 & iv_flag, varname].min(), 0.05)
        max_treated = min(df.loc[df[f'lead2_win_wk{w}'] == 0 & iv_flag, varname].max(), 0.95)

        # Filtering the DataFrame
        df_filtered = df[iv_flag & (df[varname] >= min_treated) & (df[varname] <= max_treated)]

        # Calculating centiles
        centiles = np.percentile(df_filtered[varname], np.arange(10, 100, 8))

        # Assigning groups based on centiles
        df[group_var] = pd.cut(df_filtered[varname], bins=np.insert(centiles, [0, len(centiles)], [0, 1]), labels=False, right=False)

        # Handling edge cases
        df.loc[df[varname] < centiles[0], group_var] = 1
        df.loc[(df[varname] >= centiles[-1]) & (df[varname].notna()), group_var] = 12

        # Assuming df is your DataFrame
    for w in range(1, 13):
        # Generate interaction terms
        df[f'int_pscore_win_w{w}'] = df[f'lead2_pscore_wk{w}_group'] * df[f'lead2_win_wk{w}']
        df[f'int_pscore_pscore_w{w}'] = df[f'lead2_pscore_wk{w}_group'] * df[f'lead2_pscore_wk{w}']

        # Running the regression (adjust your formula as needed)
        formula = f'lead2_exp_wins_wk{w} ~ int_pscore_win_w{w} + int_pscore_pscore_w{w} + win_wk{w} + lag_win_wk{w}'
        y, X = dmatrices(formula, df, return_type='dataframe')
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['school_id']})

        # Creating frequency tables
        freq_table = df[df['_Ilea'].notnull()].groupby(f'lead2_pscore_wk{w}_group').size().reset_index(name='count')
        freq_table['val'] = freq_table[f'lead2_pscore_wk{w}_group']

        # Calculate frequencies
        freq_dict = {j: 0 for j in range(1, 13)}
        for j in range(1, 13):
            row = freq_table[freq_table['val'] == j]
            if not row.empty:
                freq_dict[j] = row['count'].iloc[0] if row['count'].iloc[0] > 3 else 0
        # Dropping the columns 'val1' and 'freq1' if they exist
    if 'val1' in df.columns:
        df.drop('val1', axis=1, inplace=True)
    if 'freq1' in df.columns:
        df.drop('freq1', axis=1, inplace=True)

    for w in range(1, 13):
        # Dropping and creating matching_fs_coeff_lhs_w{w}
        col_name = f'matching_fs_coeff_lhs_w{w}'
        if col_name in df.columns:
            df.drop(col_name, axis=1, inplace=True)
        df[col_name] = np.nan

        # Creating 'group' and 'groupsq' for the 1st 12 rows
        df.loc[:11, 'group'] = np.arange(1, 13)
        df.loc[:11, 'groupsq'] = df.loc[:11, 'group'] ** 2

        # Assigning coefficients
        for j in range(1, 13):
            # Extracting the coefficient
            coeff_name = f'int_pscore_win_w{w}[T.{j}]'  # Adjust the coefficient name based on your model
            if coeff_name in model.params:
                df.at[j-1, col_name] = model.params[coeff_name]
    for w in range(1, 13):
        # Define column names
        lhs_col = f'matching_fs_coeff_lhs_w{w}'
        group_col = 'group'
        groupsq_col = 'groupsq'

        # Ensure the columns exist in the DataFrame
        if lhs_col not in df.columns or group_col not in df.columns or groupsq_col not in df.columns:
            continue  # Skip this iteration if any column is missing

        # Running the regression
        y = df[lhs_col]  # Dependent variable
        X = df[[group_col, groupsq_col]]  # Independent variables
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        model = sm.OLS(y, X).fit()

        # Predictions and their standard errors
        predictions = model.predict(X)
        pred_se, lower, upper = wls_prediction_std(model, exog=X, alpha=0.05)  # 95% confidence interval

        # Storing predictions and standard errors
        for j in range(1, 13):
            coeff_col = f'matching_fs_coeff_w{w}_{j}'
            se_col = f'matching_fs_se_w{w}_{j}'

            if coeff_col in df.columns:
                df.drop(coeff_col, axis=1, inplace=True)
            if se_col in df.columns:
                df.drop(se_col, axis=1, inplace=True)

            df[coeff_col] = np.nan
            df[se_col] = np.nan

            if j - 1 < len(predictions):
                df.at[j - 1, coeff_col] = predictions[j - 1]
                df.at[j - 1, se_col] = pred_se[j - 1]

        # Dropping intermediate variables
        df.drop(['group', 'groupsq'], axis=1, inplace=True)

        for w in range(1, 13):
            # Dropping prediction variables if they exist
            pred_col = f'matching_fs_coeff_w{w}_pred'
            pred_se_col = f'matching_fs_coeff_w{w}_pred_se'
            if pred_col in df.columns:
                df.drop(pred_col, axis=1, inplace=True)
            if pred_se_col in df.columns:
                df.drop(pred_se_col, axis=1, inplace=True)

            # Conditional update for w = 12
            if w == 12:
                for j in range(1, 13):
                    coeff_col = f'matching_fs_coeff_w{w}_{j}'
                    se_col = f'matching_fs_se_w{w}_{j}'

                    # Set values to 0 in the first 30 rows
                    df.loc[:29, coeff_col] = 0
                    df.loc[:29, se_col] = 0

    ### Run Reduced form for Matching Regressions

        # Assuming df is your DataFrame
    variables_list = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 
                    'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 
                    'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']

    for w in range(1, 13):
        # Generate interaction terms
        df[f'int_pscore_win_w{w}'] = df[f'lead2_pscore_wk{w}_group'] * df[f'lead2_win_wk{w}']
        df[f'int_pscore_pscore_w{w}'] = df[f'lead2_pscore_wk{w}_group'] * df[f'lead2_pscore_wk{w}']

        for varname in variables_list:
            # Running the regression
            formula = f'{varname} ~ int_pscore_win_w{w} + int_pscore_pscore_w{w}'
            y, X = dmatrices(formula, df, return_type='dataframe')
            model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['school_id']})

            # Creating frequency tables
            freq_table = df.groupby(f'lead2_pscore_wk{w}_group').size().reset_index(name='count')

            # Calculate frequencies
            freq_dict = {}
            for j in range(1, 13):
                # Filtering the relevant group
                group_data = df[df[f'lead2_pscore_wk{w}_group'] == j]

                # Calculate frequency based on conditions similar to Stata code
                if len(group_data) == 1:
                    mean_value = group_data['freq1'].mean()
                    freq_dict[j] = mean_value if mean_value > 3 else 0
                else:
                    freq_dict[j] = 0
        if 'val1' in df.columns:
            df.drop('val1', axis=1, inplace=True)
        if 'freq1' in df.columns:
            df.drop('freq1', axis=1, inplace=True)

    # Initialize a dictionary to store weights for each group
    weight_dict = {}

    for j in range(1, 13):
        # Counting wins
        wincount = len(df[(df[f'lead2_win_wk{w}'] == 1) & 
                        (df[f'lead2_pscore_wk{w}_group'] == j)])

        # Counting losses
        losscount = len(df[(df[f'lead2_win_wk{w}'] == 0) & 
                        (df[f'lead2_pscore_wk{w}_group'] == j)])

        # Setting weight based on counts
        weight_dict[j] = 1 if min(wincount, losscount) >= 2 else 0

    # Calculate total weighted frequency (totalestobs)
    totalestobs = sum(freq_dict[j] * weight_dict[j] for j in range(1, 13))

    # Assuming that model.params contains the coefficients
    coefficients = [model.params.get(f'_IleaXlead_{j}', 0) for j in range(1, 13)]
    weights = np.array([freq_dict[j] * weight_dict[j] for j in range(1, 13)])

    # Linear combination of coefficients
    linear_combination = sum(coeff * weight for coeff, weight in zip(coefficients, weights))

    # Calculate C (the linear combination divided by total weighted frequency)
    C = linear_combination / totalestobs

    # Extracting relevant parts of the covariance matrix
    indices = [f'_IleaXlead_{j}' for j in range(1, 13)]
    cov_matrix = model.cov_params().loc[indices, indices]

    # Calculate the standard error of the linear combination
    SE = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T))) / totalestobs

    # Update the DataFrame for each j
    for j in range(1, 13):
        df.at[j - 1, f'matching_rf_resid_coeff_{w}'] = C
        df.at[j - 1, f'matching_rf_resid_se_{w}'] = SE
        df.at[j - 1, f'matching_resid_N_{w}'] = len(df)  # Or however you calculate N

        # Update fs_j and fs_se_j values
        for j in range(1, 13):
            fs_j = 1 + df.at[j, f'matching_fs_coeff_{w}_{j}']
            fs_se_j = df.at[j, f'matching_fs_se_{w}_{j}']
            # If you need to store these values somewhere, include that code here
    # Calculation of C (the linear combination)
    C = sum(model.params.get(f'_IleaXlead_{j}', 0) * freq_dict[j] * weight_dict[j] / fs_dict[j] for j in range(1, 13)) / totalestobs

        # Calculation of SE (standard error)
    SE_components = []
    for j in range(1, 13):
        # Example calculation for fs and fs_se values
        fs = calculate_fs(j)  # Replace with actual calculation
        fs_se = calculate_fs_se(j)  # Replace with actual calculation

        coefficient_se = model.bse.get(f'_IleaXlead_{j}', 0) / fs
        additional_term = fs_se * model.params.get(f'_IleaXlead_{j}', 0) / fs**2
        weighted_se = (coefficient_se**2 + additional_term**2) * (freq_dict[j] * weight_dict[j] / totalestobs)**2
        SE_components.append(weighted_se)

        SE = np.sqrt(sum(SE_components))

        # Update the DataFrame
    df.at[i, f'matching_resid_coeff_{w}'] = C
    df.at[i, f'matching_resid_se_{w}'] = SE

    i += 1

    # Drop columns if they exist
    columns_to_drop = ['total_resid_N', 'matching_resid_coeff', 'matching_resid_se', 
                    'matching_resid_N', 'matching_rf_resid_coeff', 'matching_rf_resid_se']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Generate total_resid_N
    total_resid_cols = [f'matching_resid_N_{i}' for i in range(1, 13)]
    df['total_resid_N'] = df[total_resid_cols].sum(axis=1)

    # Generate matching_rf_resid_coeff and matching_resid_coeff
    for i in range(1, 13):
        df[f'matching_rf_resid_coeff_{i}_weighted'] = df[f'matching_rf_resid_coeff_{i}'] * (df[f'matching_resid_N_{i}'] / df['total_resid_N'])
        df[f'matching_resid_coeff_{i}_weighted'] = df[f'matching_resid_coeff_{i}'] * (df[f'matching_resid_N_{i}'] / df['total_resid_N'])

    df['matching_rf_resid_coeff'] = df[[f'matching_rf_resid_coeff_{i}_weighted' for i in range(1, 13)]].sum(axis=1)
    df['matching_resid_coeff'] = df[[f'matching_resid_coeff_{i}_weighted' for i in range(1, 13)]].sum(axis=1)

    # Generate matching_rf_resid_se and matching_resid_se
    for i in range(1, 13):
        df[f'matching_rf_resid_se_{i}_weighted'] = (df[f'matching_rf_resid_se_{i}']**2) * ((df[f'matching_resid_N_{i}'] / df['total_resid_N'])**2)
        df[f'matching_resid_se_{i}_weighted'] = (df[f'matching_resid_se_{i}']**2) * ((df[f'matching_resid_N_{i}'] / df['total_resid_N'])**2)

    df['matching_rf_resid_se'] = np.sqrt(df[[f'matching_rf_resid_se_{i}_weighted' for i in range(1, 13)]].sum(axis=1))
    df['matching_resid_se'] = np.sqrt(df[[f'matching_resid_se_{i}_weighted' for i in range(1, 13)]].sum(axis=1))

    # Generate matching_resid_N
    df['matching_resid_N'] = df[total_resid_cols].max(axis=1)

    # Dropping columns if they exist and generating new columns
    columns_to_generate = ['ols_result_seq', 'ols_result_wgt_seq', 'ols_pval_seq', 'ols_N_seq', 
                        'ldv_result_seq', 'ldv_result_wgt_seq', 'ldv_pval_seq', 'ldv_N_seq', 'ldv_se_wgt_seq']
    df.drop(columns=[col for col in columns_to_generate if col in df.columns], inplace=True)
    for col in columns_to_generate:
        df[col] = np.nan

    # Looping and updating values
    obscounter = 0
    varcounter = 0
    while obscounter < 20:
        secounter = obscounter + 1
        df.at[obscounter, 'ldv_N'] = df.at[varcounter, 'matching_resid_N']
        
        if 3 == 0:  # Condition always false, adjust if this is not intended
            df.at[obscounter, 'ldv_result'] = df.at[varcounter, 'matching_rf_resid_coeff']
            df.at[secounter, 'ldv_result'] = df.at[varcounter, 'matching_rf_resid_se']
            df.at[obscounter, 'ldv_pval'] = 2 * t.sf(np.abs(df.at[varcounter, 'matching_rf_resid_coeff'] / df.at[varcounter, 'matching_rf_resid_se']), 105)
        else:
            df.at[obscounter, 'ldv_result'] = df.at[varcounter, 'matching_resid_coeff']
            df.at[secounter, 'ldv_result'] = df.at[varcounter, 'matching_resid_se']
            df.at[obscounter, 'ldv_pval'] = 2 * t.sf(np.abs(df.at[varcounter, 'matching_resid_coeff'] / df.at[varcounter, 'matching_resid_se']), 105)
        
        varcounter += 1
        obscounter += 2

    # Replace xi i.year with equivalent pandas operation if needed

    counter = 0  # This is initialized but not used in the given snippet



        # freq_dict now contains the frequencies for each group

    return print(df)

def main():
    # Load the CSV file
    df = pd.read_csv('Formatted_data.csv')

    # List of variables to calculate statistics for
    variables = [
        'lag_seasonwins', 'lag_seasongames', 'lag_exp_wins',
        'alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving',
        'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants',
        'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']
    
    print(placebo_results)


    
if __name__ == "__main__":
    main()
    
