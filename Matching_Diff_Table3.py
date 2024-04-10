import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy.stats import t



    ##Moving to the Method call out that this function makes starts at 252 in stata


def main_results(df, bcs, trim_value, iv_flag, cluster_school):
    """
    Function to perform grouping based on percentile calculation.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data.
    bcs (str): Condition to be applied for bcs.
    trim_value (float): Trim value for filtering.
    iv_flag (bool): Indicator variable flag.
    cluster_school (str): Cluster school column name.
    
    Returns:
    pd.DataFrame: DataFrame with new group assignments.
    """

    # Apply the BCS condition
    df = df.query('bcs == 1')

    # Iterate through the weeks
    for w in range(1, 13):
        varname = f'lag_pscore_wk{w}'
        group_varname = f'{varname}_group'

        # Calculate minimum and maximum treated values
        min_treated = max(df[varname][(df[f'lag_win_wk{w}'] == 1) & iv_flag].min(), 0.05)
        max_treated = min(df[varname][(df[f'lag_win_wk{w}'] == 0) & iv_flag].max(), 0.95)

        # Calculate percentiles
        percentiles = np.percentile(df[varname][(df[varname] >= min_treated) & (df[varname] <= max_treated)], np.arange(10, 100, 8))

        # Create a new column for groups
        df[group_varname] = pd.cut(df[varname], bins=[-np.inf] + percentiles.tolist() + [np.inf], labels=False, right=False)
        df[group_varname] += 1  # Shift labels to start from 1 instead of 0

        # Adjust group numbers based on conditions
        df.loc[(df[varname] < percentiles[0]) & iv_flag & (df[varname] >= min_treated), group_varname] = 1
        df.loc[(df[varname] >= percentiles[-1]) & iv_flag & (df[varname] <= max_treated), group_varname] = 12

        # List of variables to manage
        variables = ['variable_name', 'ols_result', 'ols_pval', 'ols_N', 'ldv_result', 'ldv_pval', 'ldv_N']

        # Drop the variables if they exist
        for var in variables:
            if var in df.columns:
                df.drop(columns=[var], inplace=True)

        # Add the variables back with initial values
        df['variable_name'] = ''
        df['ols_result'] = np.nan
        df['ols_pval'] = np.nan
        df['ols_N'] = np.nan
        df['ldv_result'] = np.nan
        df['ldv_pval'] = np.nan
        df['ldv_N'] = np.nan

        # Now 'df' has the required columns reset
        # Drop the variable if it exists and then add it back
    if 'matching_dep_var' in df.columns:
        df.drop(columns=['matching_dep_var'], inplace=True)
    df['matching_dep_var'] = ''

    # Run first stage for matching regressions
    for w in range(1, 13):
        # Generate dummies and interaction terms
        dummies = pd.get_dummies(df[f'lag_pscore_wk{w}_group'], prefix=f'lag_pscore_wk{w}_group')
        interaction1 = dummies.mul(df[f'lag_win_wk{w}'], axis=0)
        interaction2 = dummies.mul(df[f'lag_pscore_wk{w}'], axis=0)
        df = pd.concat([df, interaction1, interaction2], axis=1)

        # Regression
        y, X = dmatrices(f'lag_exp_wins_wk{w} ~ 0 + _Ilag*', data=df, return_type='dataframe')
        model = sm.OLS(y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['school_id']})

        # Frequency and value matrices
        freq_val = df[f'lag_pscore_wk{w}_group'][results.model.data.row_labels].value_counts().reset_index()
        freq_val.columns = ['val', 'freq']
        
        # Ensure all groups are represented
        for j in range(1, 13):
            if j not in freq_val['val']:
                freq_val = freq_val.append({'val': j, 'freq': 0}, ignore_index=True)
        
        freq_val.sort_values(by='val', inplace=True)
        # Additional operations can be performed on freq_val as needed

    # Note: Some aspects like 'noconstant' in regression are handled differently in statsmodels
    # and might need adjustment based on the specifics of your analysis.

    # Run the process for each week
    for w in range(1, 13):
        # Drop if exists and generate new variables
        for j in range(1, 13):
            df.drop(columns=[f'matching_fs_coeff_{w}_{j}', f'matching_fs_se_{w}_{j}'], errors='ignore', inplace=True)

        # Create variables for regression
        df[f'matching_fs_coeff_lhs_{w}'] = np.nan
        df['group'] = np.arange(1, 13)
        df['groupsq'] = df['group'] ** 2

        # Extracting coefficients and assigning them to the created variable
        for j in range(1, 13):
            df.loc[j - 1, f'matching_fs_coeff_lhs_{w}'] = results.params.get(f'_IlagXlag__{j}', np.nan)

        # Regression
        formula = f'matching_fs_coeff_lhs_{w} ~ group + groupsq'
        result = smf.ols(formula, data=df.loc[:11, :]).fit()
        
        # Predictions
        df[f'matching_fs_coeff_{w}_pred'], df[f'matching_fs_coeff_{w}_pred_se'] = result.predict(return_std=True)

        # Creating additional columns
        for j in range(1, 13):
            df[f'matching_fs_coeff_{w}_{j}'] = np.nan
            df[f'matching_fs_se_{w}_{j}'] = np.nan

        # Fill in the generated columns with predictions and standard errors
        for j in range(1, 13):
            df.loc[j - 1, f'matching_fs_coeff_{w}_{j}'] = df.loc[j - 1, f'matching_fs_coeff_{w}_pred']
            df.loc[j - 1, f'matching_fs_se_{w}_{j}'] = df.loc[j - 1, f'matching_fs_coeff_{w}_pred_se']

        # Clean up
        df.drop(columns=['matching_fs_coeff_' + str(w) + '_pred', 'matching_fs_coeff_' + str(w) + '_pred_se', 'group', 'groupsq'], inplace=True)

        # Special case for w==12
        if w == 12:
            df.loc[:29, [f'matching_fs_coeff_{w}_{j}' for j in range(1, 13)]] = 0
            df.loc[:29, [f'matching_fs_se_{w}_{j}' for j in range(1, 13)]] = 0

    # Note: Adjust the index ranges (like :11 or :29) based on your DataFrame's structure

    # List of variables for the regression
    varlist = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']

    # Run reduced form for matching regressions
    for w in range(1, 13):
        # Drop and generate new variables
        for var in ['matching_coeff', 'matching_se', 'matching_rf_coeff', 'matching_rf_se', 'matching_N']:
            df.drop(columns=[f'{var}_{w}'], errors='ignore', inplace=True)
            df[f'{var}_{w}'] = np.nan

        # Generate dummies and interaction terms
        dummies = pd.get_dummies(df[f'lag_pscore_wk{w}_group'], prefix=f'lag_pscore_wk{w}_group')
        interaction1 = dummies.mul(df[f'lag_win_wk{w}'], axis=0)
        interaction2 = dummies.mul(df[f'lag_pscore_wk{w}'], axis=0)
        df = pd.concat([df, interaction1, interaction2], axis=1)

        i = 1
        for varname in varlist:
            # Running the regression
            y, X = dmatrices(f'{varname} ~ 0 + _Ilag*', data=df, return_type='dataframe')
            model = sm.OLS(y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': df['school_id']})

            # Storing regression results
            df.at[i - 1, f'matching_coeff_{w}'] = results.params.values[0]  # Assuming the first coefficient is of interest
            df.at[i - 1, f'matching_se_{w}'] = results.bse.values[0]        # Standard error of the first coefficient
            df.at[i - 1, f'matching_N_{w}'] = results.nobs                 # Number of observations

            # Assuming 'weight_variable' is the column with weights
            weight_variable = 'your_weight_column_name'

            for w in range(1, 13):
                # Assuming the group variable is something like 'lag_pscore_wkX_group'
                group_var = f'lag_pscore_wk{w}_group'
                
                # Calculate frequency distribution
                freq_distribution = df[group_var].value_counts()

                # Apply weights: multiply frequencies by the weight of each group
                # Assuming weights are in another column and need to be aggregated in some way
                weighted_freq = freq_distribution.mul(df.groupby(group_var)[weight_variable].mean())

                # Sum of weighted frequencies
                total_weighted_freq = weighted_freq.sum()

                # The variable 'total_weighted_freq' now holds the sum of weighted frequencies for week 'w'
                # You can store or use this variable as needed in your analysis

            i += 1

        # Drop if exists and generate new variables
        for j in range(1, 13):
            df.drop(columns=[f'matching_fs_coeff_{w}_{j}', f'matching_fs_se_{w}_{j}'], errors='ignore', inplace=True)
            df[f'matching_fs_coeff_{w}_{j}'] = np.nan
            df[f'matching_fs_se_{w}_{j}'] = np.nan

        # Create variables for regression
        df[f'matching_fs_coeff_lhs_{w}'] = np.nan
        df['group'] = np.arange(1, 13)
        df['groupsq'] = df['group'] ** 2

        # Regression
        formula = f'matching_fs_coeff_lhs_{w} ~ group + groupsq'
        result = smf.ols(formula, data=df.loc[:11, :]).fit()

        # Predictions
        df['matching_fs_coeff_pred'], df['matching_fs_coeff_pred_se'] = result.predict(df.loc[:11, ['group', 'groupsq']], return_std=True)

        # Storing predictions and standard errors in the designated columns
        for j in range(1, 13):
            df.loc[j - 1, f'matching_fs_coeff_{w}_{j}'] = df.loc[j - 1, 'matching_fs_coeff_pred']
            df.loc[j - 1, f'matching_fs_se_{w}_{j}'] = df.loc[j - 1, 'matching_fs_coeff_pred_se']

        # Clean up
        df.drop(columns=['matching_fs_coeff_pred', 'matching_fs_coeff_pred_se', 'group', 'groupsq'], inplace=True)

        # Special case for w==12
        if w == 12:
            df.loc[:29, [f'matching_fs_coeff_{w}_{j}' for j in range(1, 13)]] = 0
            df.loc[:29, [f'matching_fs_se_{w}_{j}' for j in range(1, 13)]] = 0

    # Adjust index ranges (:11 or :29) based on your DataFrame's structure

    # Assuming 'df' is your pandas DataFrame
    varlist = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']

    for w in range(1, 13):
        # Drop and create new variables
        for var in ['matching_coeff', 'matching_se', 'matching_rf_coeff', 'matching_rf_se', 'matching_N']:
            df.drop(columns=[f'{var}_{w}'], errors='ignore', inplace=True)
            df[f'{var}_{w}'] = np.nan

        # Create interaction terms
        for varname in varlist:
            dummies = pd.get_dummies(df[f'lag_pscore_wk{w}_group'], prefix=f'lag_pscore_wk{w}_group')
            interaction1 = dummies.mul(df[f'lag_win_wk{w}'], axis=0)
            interaction2 = dummies.mul(df[f'lag_pscore_wk{w}'], axis=0)
            df = pd.concat([df, interaction1, interaction2], axis=1)

            # Run regression
            y, X = dmatrices(f'{varname} ~ 0 + _Ilag*', data=df, return_type='dataframe')
            model = sm.OLS(y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': df['school_id']})

            # Storing regression results
            df.at[i - 1, f'matching_coeff_{w}'] = results.params.values[0]  # Assuming the first coefficient is of interest
            df.at[i - 1, f'matching_se_{w}'] = results.bse.values[0]        # Standard error of the first coefficient
            df.at[i - 1, f'matching_N_{w}'] = results.nobs                 # Number of observations

            # Calculate frequencies
            freq_dist = df[f'lag_pscore_wk{w}_group'][results.model.data.row_labels].value_counts()
            for j in range(1, 13):
                df.at[j - 1, f'freq_{w}_{j}'] = freq_dist.get(j, 0)

            i += 1

    # Note: Adjust the index ranges and variable names to align with your specific dataset and requirements.

    for w in range(1, 13):
        # Initialize a column to store weights for each group
        df[f'weight_{w}'] = 1  # Assume all weights are 1 initially

        # Iterate through each of the 12 groups
        for j in range(1, 13):
            # Calculate the count of wins and losses in each group
            win_count = df[(df[f'lag_pscore_wk{w}_group'] == j) & (df[f'lag_win_wk{w}'] == 1)].shape[0]
            loss_count = df[(df[f'lag_pscore_wk{w}_group'] == j) & (df[f'lag_win_wk{w}'] == 0)].shape[0]

            # Enforce overlap condition
            if min(win_count, loss_count) < 2:
                df.loc[df[f'lag_pscore_wk{w}_group'] == j, f'weight_{w}'] = 0


    for w in range(1, 13):
        # Initialize variables to store the total coefficient and standard error
        total_coeff = 0
        total_se_sq = 0
        total_weighted_freq = 0

        for j in range(1, 13):
            # Assume 'coeff_j' and 'se_j' are the regression coefficient and standard error for group j
            coeff = df.loc[df[f'lag_pscore_wk{w}_group'] == j, 'coeff_j'].values[0]
            se = df.loc[df[f'lag_pscore_wk{w}_group'] == j, 'se_j'].values[0]

            # Frequency and weight for group j
            freq = df.loc[df[f'lag_pscore_wk{w}_group'] == j, 'frequency'].values[0]
            weight = df.loc[df[f'lag_pscore_wk{w}_group'] == j, 'weight'].values[0]

            # Weighted coefficient and its variance
            weighted_coeff = coeff * freq * weight
            weighted_var = (se ** 2) * (freq ** 2) * (weight ** 2)

            # Add to totals
            total_coeff += weighted_coeff
            total_se_sq += weighted_var
            total_weighted_freq += freq * weight

        # Final weighted average and standard error
        weighted_avg_coeff = total_coeff / total_weighted_freq
        weighted_avg_se = np.sqrt(total_se_sq) / total_weighted_freq

        # Store these values in the DataFrame
        df[f'matching_coeff_{w}'] = weighted_avg_coeff
        df[f'matching_se_{w}'] = weighted_avg_se

        # Increment to next set of groups

        # Assuming 'df' is your pandas DataFrame
    # Add up the total number of observations across all groups
    df['total_N'] = df[[f'matching_N_{i}' for i in range(1, 13)]].sum(axis=1)

    # Calculate weighted averages for coefficients and standard errors
    df['matching_rf_coeff'] = sum(df[f'matching_rf_coeff_{i}'] * (df[f'matching_N_{i}'] / df['total_N']) for i in range(1, 13))
    df['matching_rf_se'] = np.sqrt(sum((df[f'matching_rf_se_{i}'] ** 2) * ((df[f'matching_N_{i}'] / df['total_N']) ** 2) for i in range(1, 13)))

    df['matching_coeff'] = sum(df[f'matching_coeff_{i}'] * (df[f'matching_N_{i}'] / df['total_N']) for i in range(1, 13))
    df['matching_se'] = np.sqrt(sum((df[f'matching_se_{i}'] ** 2) * ((df[f'matching_N_{i}'] / df['total_N']) ** 2) for i in range(1, 13)))

    # The maximum number of observations among all groups
    df['matching_N'] = df[[f'matching_N_{i}' for i in range(1, 13)]].max(axis=1)

    # Assuming 'df' is your pandas DataFrame
    df['variable_name'] = pd.NA  # Initialize the column
    df['ols_N'] = pd.NA
    df['ols_result'] = pd.NA
    df['ols_pval'] = pd.NA

    obscounter = 0
    varcounter = 0

    while obscounter < 20:
        secounter = obscounter + 1
        
        # Updating the DataFrame
        df.at[obscounter, 'variable_name'] = df.at[varcounter, 'matching_dep_var']
        df.at[obscounter, 'ols_N'] = df.at[varcounter, 'matching_N']
        
        if 3 == 0:  # Replace 3 with your actual condition
            df.at[obscounter, 'ols_result'] = df.at[varcounter, 'matching_rf_coeff']
            df.at[secounter, 'ols_result'] = df.at[varcounter, 'matching_rf_se']
            df.at[obscounter, 'ols_pval'] = 2 * t.sf(abs(df.at[varcounter, 'matching_rf_coeff'] / df.at[varcounter, 'matching_rf_se']), 105)
        else:
            df.at[obscounter, 'ols_result'] = df.at[varcounter, 'matching_coeff']
            df.at[secounter, 'ols_result'] = df.at[varcounter, 'matching_se']
            df.at[obscounter, 'ols_pval'] = 2 * t.sf(abs(df.at[varcounter, 'matching_coeff'] / df.at[varcounter, 'matching_se']), 105)
        
        varcounter += 1
        obscounter += 2

    # Note: Replace the 105 in the stats.t.sf function with the appropriate degrees of freedom for your t-test

    # List of variables to process
    varlist = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']

    # Generate year and school_id dummies
    year_dummies = pd.get_dummies(df['year'], prefix='year')
    school_id_dummies = pd.get_dummies(df['school_id'], prefix='school_id')
    df = pd.concat([df, year_dummies, school_id_dummies], axis=1)

    # Create residualized dependent variables
    for varname in varlist:
        # Creating a lagged variable
        df[f'lag2_{varname}'] = df[varname].shift(2)  # Adjust the lag as necessary

        # Difference between variable and its lag
        df[f'r{varname}_temp'] = df[varname] - df[f'lag2_{varname}']

        # Regression
        formula = f'r{varname}_temp ~ {" + ".join(year_dummies.columns)} + {" + ".join(school_id_dummies.columns)} - 1' # No intercept
        y, X = dmatrices(formula, data=df, return_type='dataframe')
        model = sm.OLS(y, X)
        results = model.fit()

        # Predicting residuals and storing them
        df[f'r{varname}'] = results.resid

        # Clean up: drop temporary variables
        df.drop(columns=[f'r{varname}_temp', f'lag2_{varname}'], inplace=True)

    # Note: This code assumes that 'year' and 'school_id' are present in your DataFrame.
    # Also, ensure that the 'year' and 'school_id' columns are correctly formatted for dummy variable creation.

    # Assuming 'df' is your pandas DataFrame
    varlist = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']

    # Creating the new columns
    for var in ['matching_resid_coeff', 'matching_resid_se', 'matching_resid_N', 'matching_rf_resid_coeff', 'matching_rf_resid_se']:
        df[var] = pd.NA  # Initialize with NA values

    for w in range(1, 13):
        for varname in varlist:
            # Interaction term generation
            dummies = pd.get_dummies(df[f'lag_pscore_wk{w}_group'], prefix=f'lag_pscore_wk{w}_group')
            interaction1 = dummies.mul(df[f'lag_win_wk{w}'], axis=0)
            interaction2 = dummies.mul(df[f'lag_pscore_wk{w}'], axis=0)
            df = pd.concat([df, interaction1, interaction2], axis=1)

            # Regression
            formula = f'r{varname} ~ 0 + _Ilag*'  # Replace '0 + _Ilag*' with actual interaction terms
            y, X = dmatrices(formula, data=df, return_type='dataframe')
            model = sm.OLS(y, X)
            results = model.fit()

            # Store regression statistics
            df.at[i, 'matching_resid_coeff'] = results.params[0]
            df.at[i, 'matching_resid_se'] = results.bse[0]
            df.at[i, 'matching_resid_N'] = len(results.model.endog)

            # Frequency distribution
            # Similar logic as in previous translations
            freq_dist = df[f'lag_pscore_wk{w}_group'][results.model.data.row_labels].value_counts()
            # Process and use the freq_dist as required

            group_var = f'lag_pscore_wk{w}_group'

            # Calculate frequency for each group
            freq = df[group_var].value_counts().get(j, 0)
            
            # Optional: Apply weights if necessary# Assuming 'df' is your pandas DataFrame
            for w in range(1, 13):
                # This is where the frequency distribution logic will go for each week

                # Initialize a dictionary to store frequencies for each group
                freq_dict = {}

                # Iterate through each group and calculate frequencies
                for j in range(1, 13):
                    group_var = f'lag_pscore_wk{w}_group'

                    # Calculate frequency for each group
                    freq = df[group_var].value_counts().get(j, 0)
                    
                    # Optional: Apply weights if necessary
                    # Assuming weight is a column like 'weight_j'. Adjust as per your data structure.
                    weight = df[f'weight_{j}'].iloc[0] if f'weight_{j}' in df.columns else 1

                    # Storing weighted frequency
                    freq_dict[j] = freq * weight
                    # Assuming weight is a column like 'weight_j'. Adjust as per your data structure.
                    weight = df[f'weight_{j}'].iloc[0] if f'weight_{j}' in df.columns else 1

                    # Storing weighted frequency
                    freq_dict[j] = freq * weight
    # Drop interaction terms to prepare for next iteration
            df.drop(interaction1.columns.tolist() + interaction2.columns.tolist(), axis=1, inplace=True)

            i += 1# Assuming 'df' is your pandas DataFrame

    for w in range(1, 13):
        # Add a column for weights for each week
        df[f'weight_{w}'] = 1  # Initialize all weights as 1

        for j in range(1, 13):
            # Count wins and losses for each group
            win_count = df[(df[f'lag_pscore_wk{w}_group'] == j) & (df[f'lag_win_wk{w}'] == 1)].shape[0]
            loss_count = df[(df[f'lag_pscore_wk{w}_group'] == j) & (df[f'lag_win_wk{w}'] == 0)].shape[0]

            # Enforce overlap condition
            if min(win_count, loss_count) < 2:
                df.loc[df[f'lag_pscore_wk{w}_group'] == j, f'weight_{w}'] = 0

    for w in range(1, 13):
        # Calculate total weighted frequency
        totalestobs = sum(df[f'freq{j}'] * df[f'weight{j}'] for j in range(1, 13))

        # Calculate weighted average of coefficients
        C = sum(df[f'coefficients_{j}'] * df[f'freq{j}'] * df[f'weight{j}'] for j in range(1, 13)) / totalestobs
        SE = sum(df[f'standard_errors_{j}'] * df[f'freq{j}'] * df[f'weight{j}'] for j in range(1, 13)) / totalestobs

        # Store these values
        df.at[i, f'matching_rf_resid_coeff_{w}'] = C
        df.at[i, f'matching_rf_resid_se_{w}'] = SE
        df.at[i, f'matching_resid_N_{w}'] = totalestobs  # Assuming totalestobs represents the number of observations

        # Adjust the coefficients and standard errors
        for j in range(1, 13):
            fs_j = 1 + df.at[i, f'matching_fs_coeff_{w}_{j}']
            fs_se_j = df.at[i, f'matching_fs_se_{w}_{j}']

            # Adjust coefficients and SE here as needed

        i += 1  # Increment i for the next iteration

        for w in range(1, 13):
        # Calculate total estimated observations (totalestobs)
            totalestobs = sum(df[f'freq{j}'] * df[f'weight{j}'] for j in range(1, 13))

            # Calculating the weighted sum of coefficients (C)
            C = sum(df[f'_b_IlagXlag__{j}'] * df[f'freq{j}'] * df[f'weight{j}'] / df[f'fs{j}'] for j in range(1, 13)) / totalestobs

            # Calculating the composite standard error (SE)
            SE_squared_sum = sum(((df[f'_se_IlagXlag__{j}'] / df[f'fs{j}']) ** 2 + (df[f'fs_se{j}'] * df[f'_b_IlagXlag__{j}'] / df[f'fs{j}'] ** 2) ** 2) * (df[f'freq{j}'] * df[f'weight{j}'] / totalestobs) ** 2 for j in range(1, 13))
            SE = np.sqrt(SE_squared_sum)

            # Store the calculated C and SE
            i = ...  # Set 'i' to the appropriate index
            df.at[i, f'matching_resid_coeff_{w}'] = C
            df.at[i, f'matching_resid_se_{w}'] = SE

            i += 1  # Increment i for the next iteration

    # Note: Replace '_b_IlagXlag__{j}', '_se_IlagXlag__{j}', 'fs{j}', and 'fs_se{j}' with the actual names of your DataFrame columns
    # that contain these values. Ensure that 'fs{j}' and 'fs_se{j}' are already calculated and available in your DataFrame.

    # Calculating total number of observations across all groups
    df['total_resid_N'] = df[[f'matching_resid_N_{i}' for i in range(1, 13)]].sum(axis=1)

    # Calculating weighted averages for coefficients and standard errors
    df['matching_rf_resid_coeff'] = sum(df[f'matching_rf_resid_coeff_{i}'] * (df[f'matching_resid_N_{i}'] / df['total_resid_N']) for i in range(1, 13))
    df['matching_rf_resid_se'] = np.sqrt(sum((df[f'matching_rf_resid_se_{i}'] ** 2) * ((df[f'matching_resid_N_{i}'] / df['total_resid_N']) ** 2) for i in range(1, 13)))

    df['matching_resid_coeff'] = sum(df[f'matching_resid_coeff_{i}'] * (df[f'matching_resid_N_{i}'] / df['total_resid_N']) for i in range(1, 13))
    df['matching_resid_se'] = np.sqrt(sum((df[f'matching_resid_se_{i}'] ** 2) * ((df[f'matching_resid_N_{i}'] / df['total_resid_N']) ** 2) for i in range(1, 13)))

    # The maximum number of observations among all groups
    df['matching_resid_N'] = df[[f'matching_resid_N_{i}' for i in range(1, 13)]].max(axis=1)

    # Initialize new variables
    for var in ['ols_result_seq', 'ols_result_wgt_seq', 'ols_pval_seq', 'ols_N_seq', 'ldv_result_seq', 'ldv_result_wgt_seq', 'ldv_pval_seq', 'ldv_N_seq', 'ldv_se_wgt_seq']:
        df[var] = np.nan

    obscounter = 0
    varcounter = 0

    while obscounter < 20:
        secounter = obscounter + 1

        # Updating the DataFrame
        df.at[obscounter, 'ldv_N'] = df.at[varcounter, 'matching_resid_N']
        
        if 3 == 0:  # Replace 3 with your actual condition
            df.at[obscounter, 'ldv_result'] = df.at[varcounter, 'matching_rf_resid_coeff']
            df.at[secounter, 'ldv_result'] = df.at[varcounter, 'matching_rf_resid_se']
            df.at[obscounter, 'ldv_pval'] = 2 * t.sf(abs(df.at[varcounter, 'matching_rf_resid_coeff'] / df.at[varcounter, 'matching_rf_resid_se']), 105)
        else:
            df.at[obscounter, 'ldv_result'] = df.at[varcounter, 'matching_resid_coeff']
            df.at[secounter, 'ldv_result'] = df.at[varcounter, 'matching_resid_se']
            df.at[obscounter, 'ldv_pval'] = 2 * t.sf(abs(df.at[varcounter, 'matching_resid_coeff'] / df.at[varcounter, 'matching_resid_se']), 105)
        
        varcounter += 1
        obscounter += 2

    # Drop unused interaction terms if they exist in your DataFrame
    # df.drop(columns=[col for col in df.columns if col.startswith('_I')], errors='ignore', inplace=True)

    # Note: Replace the 105 in the stats.t.sf function with the appropriate degrees of freedom for your t-test
    # Adjust the indices, variable names, and conditional logic as per your specific requirements.

    # Assuming 'df' is your pandas DataFrame
    df['rseasonwins'] = df['lag_seasonwins'] - df['lag3_seasonwins']
    df['rseasongames'] = df['lag_seasongames'] - df['lag3_seasongames']

    varlist = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']
    counter = 0

    for varname in varlist:
        # Creating difference variables
        df[f'r{varname}'] = df[varname] - df[f'lag2_{varname}']

        # Regression with clustered standard errors
        formula = f'r{varname} ~ rseasonwins + lag3_seasonwins + lag_seasongames + lag3_seasongames + C(year)' # Adjust the formula as necessary
        model = smf.ols(formula, data=df, weights=df['lag_ipw_weight']) # Adjust weights and subset conditions as necessary
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['school_id']})

        # Updating variables based on regression results
        df.at[counter, 'ldv_result_seq'] = results.params['rseasonwins']
        df.at[counter, 'ldv_pval_seq'] = 2 * t.sf(np.abs(results.params['rseasonwins'] / results.bse['rseasonwins']), results.df_resid) # df_resid is degrees of freedom
        df.at[counter, 'ldv_N_seq'] = results.nobs
        counter += 1

        df.at[counter, 'ldv_result_seq'] = results.bse['rseasonwins']
        counter += 1

    # Conditional operation
    if 3 == 0:  # Replace 3 with the actual condition
        df.rename(columns={'ldv_pval': 'ldv_pval_rf'}, inplace=True)



# Load your dataset
df = pd.read_csv('Formatted_data.csv')

# Filter parameters
bcs = df['bcs'] <= 1
trim_value = df['lag_ipw_weight'].quantile(0.9)
iv_flag = 1
cluster = 'school_id'

# Call to Main_results function
# You should define this function based on your specific analysis needs
main_results(df, bcs, trim_value, iv_flag, cluster)

# Assuming main_results modifies df in-place or returns a modified DataFrame
# If it returns a DataFrame, uncomment the following line:
df = main_results(df, bcs, trim_value, iv_flag, cluster)

# Define the list of variables for the regression
var_list = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']

# Loop for regression, prediction, and adjustments
for varname in var_list:
    # Run regression
    formula = f'{varname} ~ lag3_seasonwins + lag_seasongames + lag3_seasongames + C(year)'
    model = smf.ols(formula, data=df.query('lag_ipw_weight < @trim_value and @bcs and lag_seasonwins != "."'))
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df[cluster]})

    # Store results (example: residuals)
    df[f'r{varname}_resid'] = result.resid + result.params['Intercept']

# Define the list of variables
var_list = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25']

# Loop through the variables and apply transformations
for varname in var_list:
    resid_varname = f'r{varname}_resid'

    # Adjustments based on variable names
    if varname in ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving']:
        df[resid_varname] = df[resid_varname] / 1000000
        df[resid_varname].rename("Millions of dollars", inplace=True)

    elif varname == 'vse_alum_giving_rate':
        df[resid_varname].rename("Share", inplace=True)

    elif varname in ['usnews_academic_rep_new', 'sat_25']:
        df[resid_varname].rename("Points", inplace=True)

    elif varname == 'acceptance_rate':
        df[resid_varname] = df[resid_varname] / 100
        df[resid_varname].rename("Share", inplace=True)

    elif varname in ['applicants', 'firsttime_outofstate', 'first_time_instate']:
        df[resid_varname].rename("Students", inplace=True)

labels = {
    'alumni_ops_athletics': "Athletic Operating Donations",
    'alum_non_athl_ops': "Nonathletic Operating Donations",
    'alumni_total_giving': "Total Alumni Donations",
    'vse_alum_giving_rate': "Alumni Giving Rate",
    'usnews_academic_rep_new': "Academic Reputation",
    'applicants': "Applicants",
    'acceptance_rate': "Acceptance Rate",
    'firsttime_outofstate': "First-Time Out-of-State Enrollment",
    'first_time_instate': "First-Time In-State Enrollment",
    'sat_25': "25th Percentile SAT Score"
}

var_list = list(labels.keys())

import seaborn as sns
import matplotlib.pyplot as plt

counter = 1
for varname in var_list:
    plt.figure(figsize=(10, 6))
    sns.regplot(x=f'r{varname}_resid', y='rseasonwins_resid', data=df, lowess=True, 
                scatter_kws={'s': 5}, line_kws={'color': 'grey', 'lw': 1})

    plt.title(f"Panel {counter}: Effect of Wins on {labels[varname]}", size='large')
    plt.xlabel("Change in Wins", size='medium')
    plt.ylabel(labels[varname], size='medium')
    plt.xticks(rotation=-6, fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.legend().set_visible(False)

    plt.savefig(f'gph_{counter}.png', bbox_inches='tight')
    plt.close()
    counter += 1
