import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.sandbox.regression.predstd import wls_prediction_std

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

    # weight_dict now contains the weights for each group
        ## Made it to this line in stata should be 845
		# local totalestobs = (`freq1'*`weight1' + `freq2'*`weight2' + `freq3'*`weight3' + `freq4'*`weight4' + `freq5'*`weight5' + `freq6'*`weight6' + `freq7'*`weight7' + `freq8'*`weight8' + `freq9'*`weight9' + `freq10'*`weight10' + `freq11'*`weight11' + `freq12'*`weight12')
		# lincom (_b[_IleaXlead_1]*`freq1'*`weight1' + _b[_IleaXlead_2]*`freq2'*`weight2' + _b[_IleaXlead_3]*`freq3'*`weight3' + _b[_IleaXlead_4]*`freq4'*`weight4' + _b[_IleaXlead_5]*`freq5'*`weight5' + _b[_IleaXlead_6]*`freq6'*`weight6' + _b[_IleaXlead_7]*`freq7'*`weight7' + _b[_IleaXlead_8]*`freq8'*`weight8' + _b[_IleaXlead_9]*`freq9'*`weight9' + _b[_IleaXlead_10]*`freq10'*`weight10' + _b[_IleaXlead_11]*`freq11'*`weight11' + _b[_IleaXlead_12]*`freq12'*`weight12')/`totalestobs'
		# local C = r(estimate)
		# local SE = r(se)
		# replace matching_rf_resid_coeff_`w' = `C' in `i'
		# replace matching_rf_resid_se_`w' = `SE' in `i'
		# replace matching_resid_N_`w' = e(N) in `i'		
		# forvalues j = 1(1)12 {
		# 	local fs`j' = 1 + matching_fs_coeff_`w'_`j'[`i']
		# 	local fs_se`j' = matching_fs_se_`w'_`j'[`i']
		# }






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
    
