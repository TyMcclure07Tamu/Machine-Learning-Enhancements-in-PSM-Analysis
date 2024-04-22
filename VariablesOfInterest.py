import pandas as pd
import statsmodels.api as sm

# Assuming df is your DataFrame and it's already loaded with your data
# Replace 'Formatted_data.csv' with the path to your actual data file
df = pd.read_csv('Formatted_data.csv')

# Sorting the DataFrame by 'teamname' and 'year'
df_sorted = df.sort_values(by=['teamname', 'year'])

# If you want to sort in place without creating a new DataFrame:
# df.sort_values(by=['teamname', 'year'], inplace=True)


# Define the list of variable names you want to create lag and lead variables for
varlist = ['seasonwins', 'pct_win', 'exp_wins']  # Add all the variables in your list here

# Loop over each variable name
for varname in varlist:
    # Sort the DataFrame by teamname and year to ensure the shifts are applied correctly
    df.sort_values(by=['teamname', 'year'], inplace=True)

    # Group by 'teamname' to ensure the lag/lead calculations are done within each team
    grouped = df.groupby('teamname')

    # Generate lagged and lead variables
    df[f'lag_{varname}'] = grouped[varname].shift(1)
    df[f'lag2_{varname}'] = grouped[varname].shift(2)
    df[f'lag3_{varname}'] = grouped[varname].shift(3)
    df[f'lag4_{varname}'] = grouped[varname].shift(4)
    df[f'lead_{varname}'] = grouped[varname].shift(-1)
    df[f'lead2_{varname}'] = grouped[varname].shift(-2)

    # For year-based conditions, you might need to explicitly check year differences
    # This part is a bit tricky because shift doesn't account for year differences
    # You'll need to create a temporary column for 'year' to compare the year differences
    df['temp_year_shift_1'] = grouped['year'].shift(1)
    df['temp_year_shift_2'] = grouped['year'].shift(2)
    # Then use these temporary columns to filter rows where year differences don't match the conditions
    # You would need to create temporary columns to hold shifted 'year' values and then nullify the generated lag/lead variables where the year difference conditions are not met.
    # For example:
    # df.loc[df['year'] - df['temp_year_shift_1'] != 1, f'lag_{varname}'] = None

# Remember to remove the temporary columns after use
df.drop(['temp_year_shift_1', 'temp_year_shift_2'], axis=1, inplace=True)

# Use factorize() to generate numeric identifiers for each unique 'teamname'
df['school_id'] = pd.factorize(df['teamname'])[0]

# Use get_dummies() to create dummy variables for each unique year
year_dummies = pd.get_dummies(df['year'], prefix='year')

# Concatenate the dummy variables with the original DataFrame
df = pd.concat([df, year_dummies], axis=1)

# Make sure your data is sorted by panel variable and time
df.sort_values(by=['school_id', 'year'], inplace=True)

# Example: Running a random effects model using MixedLM in statsmodels
# Here, we assume 'outcome_variable' is your dependent variable and 'independent_variable' is your independent variable
# You should replace these with your actual variable names
md = sm.MixedLM.from_formula("outcome_variable ~ independent_variable", groups=df["school_id"], data=df)
mdf = md.fit()
print(mdf.summary())

# Deal with special reporting dates for SAT scores and applicants

# Sort the DataFrame by 'teamname' and 'year'
df_sorted = df.sort_values(by=['teamname', 'year'])

# If you want to sort in place without creating a new DataFrame:
# df.sort_values(by=['teamname', 'year'], inplace=True)

# List of variables to iterate over
varlist = ['satmt25', 'satmt75', 'satvr25', 'satvr75']

for varname in varlist:
    # Step 1: Create a temporary column
    temp_varname = f'{varname}_temp'
    df[temp_varname] = df[varname]
    
    # Step 2: Replace SAT scores in the temp column where satdate == 1
    df.loc[df['satdate'] == 1, temp_varname] = pd.NA

    # Ensure DataFrame is sorted by 'teamname' and 'year' for the next operation
    df.sort_values(by=['teamname', 'year'], inplace=True)

    # Step 3: Replace SAT scores in the temp column based on the given conditions
    for i in df.index[:-1]:  # Exclude the last row to avoid index out of bounds
        if pd.isna(df.at[i, temp_varname]) and df.at[i+1, 'satdate'] == 1 and \
           df.at[i+1, 'teamname'] == df.at[i, 'teamname'] and \
           df.at[i+1, 'year'] == df.at[i, 'year'] + 1:
            df.at[i, temp_varname] = df.at[i+1, varname]
    
    # Step 4: Drop the original SAT score column
    df.drop(columns=[varname], inplace=True)
    
    # Step 5: Rename the temporary column back to the original column name
    df.rename(columns={temp_varname: varname}, inplace=True)

# List of variables to iterate over
varlist = ['applicants_male', 'applicants_female', 'enrolled_male', 'enrolled_female']

for varname in varlist:
    # Create a temporary column
    temp_varname = f'{varname}_temp'
    df[temp_varname] = df[varname]
    
    # Replace values in the temp column where appdate == 1
    df.loc[df['appdate'] == 1, temp_varname] = pd.NA

    # Ensure DataFrame is sorted by 'teamname' and 'year' for the next operation
    df.sort_values(by=['teamname', 'year'], inplace=True)

    # Using iterrows() for row-wise operation. Note: iterrows() is generally slower on large datasets
    for i, row in df[:-1].iterrows():  # Exclude the last row to avoid index out of bounds
        next_row = df.iloc[i + 1]
        if pd.isna(row[temp_varname]) and next_row['appdate'] == 1 and \
           next_row['teamname'] == row['teamname'] and \
           next_row['year'] == row['year'] + 1:
            df.at[i, temp_varname] = next_row[varname]

    # Drop the original variable column
    df.drop(columns=[varname], inplace=True)
    
    # Rename the temporary column back to the original variable name
    df.rename(columns={temp_varname: varname}, inplace=True)

    # Sort the DataFrame by 'teamname' and 'year'
df_sorted = df.sort_values(by=['teamname', 'year'])

# If you want to sort in place without creating a new DataFrame:
# df.sort_values(by=['teamname', 'year'], inplace=True)

# Define the list of variable names to process
varlist = ['satmt25', 'satmt75', 'satvr25', 'satvr75']

# Ensure DataFrame is sorted by 'teamname' and 'year' for accurate processing
df.sort_values(by=['teamname', 'year'], inplace=True)

# Iterate over each variable in the varlist
for varname in varlist:
    # Create a temporary column by copying the values from the original column
    temp_varname = f'{varname}_temp'
    df[temp_varname] = df[varname]

    # Iterate through the DataFrame row by row
    for i in range(len(df)-1):  # Exclude the last row to avoid index out of bounds error
        # Check if the next row is the same team and the subsequent year
        if df.iloc[i+1]['teamname'] == df.iloc[i]['teamname'] and df.iloc[i+1]['year'] == df.iloc[i]['year'] + 1:
            # Update the temporary variable with the value from the next row
            df.at[i, temp_varname] = df.at[i+1, varname]

    # Drop the original column
    df.drop(columns=[varname], inplace=True)

    # Rename the temporary column to the original column name
    df.rename(columns={temp_varname: varname}, inplace=True)

# Define the list of variable names to iterate over
varlist = ['applicants_male', 'applicants_female', 'enrolled_male', 'enrolled_female']

# Ensure DataFrame is sorted by 'teamname' and 'year' for accurate processing
df.sort_values(by=['teamname', 'year'], inplace=True)

# Iterate over each variable in the varlist
for varname in varlist:
    # Step 1: Create a temporary column with NaN values
    temp_varname = f'{varname}_temp'
    df[temp_varname] = pd.NA  # Using pd.NA for consistency with pandas missing values

    # Step 2: Iterate through the DataFrame row by row
    for i in range(len(df)-1):  # Exclude the last row to avoid index out of bounds error
        # Check if the next row is the same team and the subsequent year
        if df.iloc[i+1]['teamname'] == df.iloc[i]['teamname'] and df.iloc[i+1]['year'] == df.iloc[i]['year'] + 1:
            # Update the temporary variable with the value from the next row
            df.at[i, temp_varname] = df.at[i+1, varname]

    # Step 3: Drop the original column
    df.drop(columns=[varname], inplace=True)

    # Step 4: Rename the temporary column to the original column name
    df.rename(columns={temp_varname: varname}, inplace=True)


# Generate 'athletics_share' based on the condition
# Use numpy.where to vectorize the conditional operation for efficiency
import numpy as np

# Calculate the ratio
ratio = df['alumni_ops_athletics'] / df['alumni_ops_total']

# Apply the condition and assign to 'athletics_share'
df['athletics_share'] = np.where((ratio > 0.05) & (ratio < 0.8), ratio, np.nan)

# Generate 'alum_non_athl_ops' by subtracting 'alumni_ops_athletics' from 'alumni_ops_total'
df['alum_non_athl_ops'] = df['alumni_ops_total'] - df['alumni_ops_athletics']

# Generate 'sat_75' by adding 'satmt75' and 'satvr75'
df['sat_75'] = df['satmt75'] + df['satvr75']

# Generate 'sat_25' by adding 'satmt25' and 'satvr25'
df['sat_25'] = df['satmt25'] + df['satvr25']

# Generate 'applicants' by adding 'applicants_male' and 'applicants_female'
df['applicants'] = df['applicants_male'] + df['applicants_female']

# Drop 'appdate' and 'satdate' columns from the DataFrame
df.drop(columns=['appdate', 'satdate'], inplace=True)

# Rename 'ops_athletics_total_grand' column to 'ops_athl_grndtotal'
df.rename(columns={'ops_athletics_total_grand': 'ops_athl_grndtotal'}, inplace=True)

# Rename 'first_time_students_total' column to 'firsttime_total'
df.rename(columns={'first_time_students_total': 'firsttime_total'}, inplace=True)

# Rename 'first_time_outofstate' column to 'firsttime_outofstate'
df.rename(columns={'first_time_outofstate': 'firsttime_outofstate'}, inplace=True)

# Sort the DataFrame by 'teamname' and 'year'
df.sort_values(by=['teamname', 'year'], inplace=True)

# List of variables to create lagged versions for
varlist = [
    'alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving', 
    'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants', 
    'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 
    'sat_25', 'sat_75'
]

# Ensure DataFrame is sorted by 'teamname' and 'year' for accurate lagging
df.sort_values(by=['teamname', 'year'], inplace=True)

# Generate lagged variables
for varname in varlist:
    # Group by 'teamname' and shift the variable for lagged values
    df[f'lag_{varname}'] = df.groupby('teamname')[varname].shift(1)
    df[f'lag2_{varname}'] = df.groupby('teamname')[varname].shift(2)
    df[f'lag3_{varname}'] = df.groupby('teamname')[varname].shift(3)
    df[f'lag4_{varname}'] = df.groupby('teamname')[varname].shift(4)

    # Additional logic to ensure that lagged values are only taken from consecutive years
    for i in range(1, 5):  # For lag_ to lag4_
        df.loc[df['year'] - df.groupby('teamname')['year'].shift(i) != i, f'lag{i}_{varname}'] = pd.NA
