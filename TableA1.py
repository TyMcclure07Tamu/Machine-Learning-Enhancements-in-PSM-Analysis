# ***************************************************
# * Table A1: Unexpected Losses vs. Unexpected Wins *
# ***************************************************

import pandas as pd

# Replace 'path/to/college_data.dta' with the actual path to your data file
# Since pandas cannot directly read .dta files without the full path,
# ensure the file path is correct and accessible.

# Load data
college_data = pd.read_csv("College_data_outbound.csv")

# Sort data
college_data.sort_values(by=["teamname", "year"], inplace=True)

# Conditional setup based on unexpected wins or losses
# This is a placeholder for where this condition might be used later in analysis
unexp_wins = 0

# Define the columns you want to keep
columns_to_keep = ['teamname', 'year', 'athletics_total', 'alumni_ops_athletics',
                   'alumni_ops_total', 'ops_athletics_total_grand', 'usnews_academic_rep_new',
                   'acceptance_rate', 'appdate', 'satdate', 'satmt75', 'satvr75',
                   'satmt25', 'satvr25', 'applicants_male', 'applicants_female',
                   'enrolled_male', 'enrolled_female', 'vse_alum_giving_rate',
                   'first_time_students_total', 'first_time_outofstate', 'first_time_instate',
                   'total_giving_pre_2004', 'alumni_total_giving', 'asian', 'hispanic',
                   'black', 'control']

# Filter the dataframe to keep only the specified columns
college_data_filtered = college_data[columns_to_keep]

# Save the filtered dataframe to a new file
# You may choose to save in a format that's convenient for reading back into pandas; here, I'm using CSV for compatibility
# Replace 'temp.csv' with the desired file name and path
college_data_filtered.to_csv('temp.csv', index=False)

# Load a new dataset
# Replace 'path/to/covers_data.dta' with the actual path to your covers_data.dta file
covers_data = pd.read_csv("Covers_data_outbound.csv")

import pandas as pd

# Assuming 'covers_data.csv' is the CSV version of 'covers_data.dta'
covers_data = pd.read_csv('Covers_data_outbound.csv')

# Conditions for rows to be dropped
condition1 = (covers_data['team'] == 68) & (covers_data['date'] == 14568) & (covers_data['line'].isna())
condition2 = (covers_data['team'] == 136) & (covers_data['date'] == 14568) & (covers_data['line'].isna())

# Drop rows where either condition is True
covers_data = covers_data[~(condition1 | condition2)]

# Optional: Save the cleaned data back to a new CSV file, if needed
covers_data.to_csv('covers_data_cleaned.csv', index=False)

import pandas as pd

# Assuming 'covers_data.csv' is already loaded as covers_data DataFrame
# covers_data = pd.read_csv('path/to/covers_data.csv')

# Ensure the data is sorted by team, season, and date as required
covers_data.sort_values(by=['team', 'season', 'date'], inplace=True)

# Generate a 'week' variable that increments for each game within a given team and season
# The count starts at 0 by default with cumcount, so add 1 to start counting weeks at 1
covers_data['week'] = covers_data.groupby(['team', 'season']).cumcount() + 1

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

import pandas as pd
import numpy as np

covers_data = pd.read_csv('covers_data_cleaned.csv')

# Assuming 'month' is a column in your DataFrame. If not, you'll need to derive it from a date column.
# Also assumes that 'week' and 'month' columns already exist.
covers_data = covers_data[(covers_data['week'] <= 12) & ~(covers_data['month'].isin([12, 1]))]

# Sort data by teamname, season, and week
covers_data.sort_values(by=['teamname', 'season', 'week'], inplace=True)

# Generate a count of observations for each group
covers_data['total_obs'] = covers_data.groupby(['teamname', 'season'])['line'].transform('count')

# Keep rows where total_obs >= 8
covers_data = covers_data[covers_data['total_obs'] >= 8]

# Replace missing values in 'line' column with the mean of non-missing values
covers_data['line'] = covers_data['line'].fillna(covers_data['line'].mean())

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

covers_data = pd.read_csv('covers_data_cleaned.csv')

# Replace missing values in 'win' column with the mean of non-missing values
covers_data['win'] = covers_data['win'].fillna(covers_data['win'].mean())

print(covers_data['win'].unique())

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit

covers_data = pd.read_csv('covers_data_cleaned.csv')

# Assuming 'covers_data' DataFrame is already loaded and contains 'line' and 'win' columns
for i in range(2, 6):
    covers_data[f'line{i}'] = covers_data['line'] ** i

import statsmodels.api as sm

# Prepare the independent variables by adding a constant (intercept) to the DataFrame
X = sm.add_constant(covers_data[['line', 'line2', 'line3', 'line4', 'line5']])
y = covers_data['win']

# Fit the logistic regression model
model = Logit(y, X).fit()

# Display the model summary to get an overview of the model performance and coefficients
print(model.summary())

# Predict the propensity scores based on the logistic regression model
covers_data['pscore'] = model.predict(X)

# Summarize the predicted propensity scores
print(covers_data['pscore'].describe())

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

import pandas as pd

covers_data = pd.read_csv('covers_data_cleaned.csv')

# Assuming 'covers_data' is your DataFrame and it already contains 'realspread' and 'line' columns
covers_data['outperform'] = covers_data['realspread'] + covers_data['line']

covers_data.sort_values(by=['teamname', 'season', 'week'], inplace=True)

# Initialize columns for the calculations to avoid key errors
for i in range(1, 12):
    covers_data[f'outperform_wk{i}'] = 0
    covers_data[f'outperformwk{i}_2'] = 0
    covers_data[f'outperformwk{i}_3'] = 0

# Iterate over each group of 'teamname' and 'season'
for (teamname, season), group_df in covers_data.groupby(['teamname', 'season']):
    for i in range(1, 12):
        # Filter for the specific week
        week_data = group_df[group_df['week'] == i]
        
        # Calculate mean outperformance for the week, if any rows match
        if not week_data.empty:
            mean_outperform = week_data['outperform'].mean()
        else:
            mean_outperform = 0  # Treat as neither under nor overperforming
        
        # Assign the calculated mean outperformance and its powers to the original DataFrame
        covers_data.loc[(covers_data['teamname'] == teamname) & (covers_data['season'] == season), f'outperform_wk{i}'] = mean_outperform
        covers_data.loc[(covers_data['teamname'] == teamname) & (covers_data['season'] == season), f'outperformwk{i}_2'] = mean_outperform ** 2
        covers_data.loc[(covers_data['teamname'] == teamname) & (covers_data['season'] == season), f'outperformwk{i}_3'] = mean_outperform ** 3

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Pre-define the line_clean column with NaNs
covers_data['line_clean'] = np.nan

# Handle week 1 separately as per Stata code logic
covers_data.loc[covers_data['week'] == 1, 'line_clean'] = covers_data['line']

# Iterate through weeks 2 to 12 for regression and prediction
for i in range(2, 13):
    # Create a DataFrame for the current and all previous weeks
    current_and_previous_weeks = covers_data[covers_data['week'] < i]
    
    # Create independent variables (X): cubic terms of outperform from all previous weeks
    for j in range(1, i):
        current_and_previous_weeks[f'outperform_wk{j}_cubic'] = current_and_previous_weeks[f'outperform_wk{j}'] ** 3
    
    # Filter rows for the regression
    regression_data = current_and_previous_weeks[current_and_previous_weeks['week'] == i]
    
    if not regression_data.empty:
        X = regression_data[[col for col in regression_data.columns if 'outperform_wk' in col and 'cubic' in col]]
        X = sm.add_constant(X)  # Add constant term for the intercept
        y = regression_data['line']
        
        # Perform the regression
        model = sm.OLS(y, X, missing='drop').fit()  # OLS regression
        
        # Predict and calculate residuals
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Update the line_clean column with residuals for the current week
        covers_data.loc[(covers_data['week'] == i), 'line_clean'] = residuals + model.params['const']

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

covers_data = pd.read_csv('covers_data_cleaned.csv')

# Assuming 'covers_data' DataFrame already contains 'line_clean' column
for i in range(2, 6):
    covers_data[f'line_clean_p{i}'] = covers_data['line_clean'] ** i

# Add a constant term for the intercept
X = sm.add_constant(covers_data[['line_clean', 'line_clean_p2', 'line_clean_p3', 'line_clean_p4', 'line_clean_p5']])
y = covers_data['win'].astype('float')  # Ensure that 'win' is of a correct type for logistic regression

# Fit the logistic regression model
model = Logit(y, X, missing='drop').fit()  # 'missing='drop'' to ignore rows with NaNs

# Print the summary of the regression model to check its performance
print(model.summary())

# Predict the propensity scores using the fitted model
covers_data['pscore_clean_line'] = model.predict(X)

# Summary statistics for the propensity scores
print(covers_data['pscore_clean_line'].describe())

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

covers_data = pd.read_csv('covers_data_cleaned.csv')

covers_data.sort_values(by=['teamname', 'season', 'week'], inplace=True)

# Assuming 'covers_data' DataFrame is correctly loaded and contains the necessary columns

# Generate aggregate measures
covers_data['seasonwins'] = covers_data.groupby(['teamname', 'season'])['win'].transform('sum')
covers_data['seasongames'] = covers_data.groupby(['teamname', 'season'])['win'].transform('count')
covers_data['seasonspread'] = covers_data.groupby(['teamname', 'season'])['realspread'].transform('sum')
covers_data['seasonline'] = covers_data.groupby(['teamname', 'season'])['line'].transform('sum')
covers_data['seasonoutperform'] = covers_data.groupby(['teamname', 'season'])['outperform'].transform('sum')

# Calculate winning percentage
covers_data['pct_win'] = covers_data['seasonwins'] / covers_data['seasongames']

# Assert that pct_win is within the valid range [0, 1]
assert covers_data['pct_win'].between(0, 1).all(), "pct_win is out of bounds"

# Display the first few rows to verify the calculations
print(covers_data[['teamname', 'season', 'week', 'win', 'seasonwins', 'seasongames', 'pct_win']].head())

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

covers_data = pd.read_csv('covers_data_cleaned.csv')

# Calculate total expected wins (naive and with clean line)
covers_data['exp_wins_naive'] = covers_data.groupby(['teamname', 'season'])['pscore'].transform('sum')
covers_data['exp_wins'] = covers_data.groupby(['teamname', 'season'])['pscore_clean_line'].transform('sum')

# Calculate expected win percentage
covers_data['exp_win_pct'] = covers_data['exp_wins'] / covers_data['seasongames']

# Initialize expected wins by week columns to zero
for w in range(1, 13):
    covers_data[f'exp_wins_wk{w}'] = 0

# Calculate expected wins by week for weeks 1 through 11
for w in range(1, 12):
    # Filter rows for games in later weeks
    future_games = covers_data[covers_data['week'] > w]
    
    # Sum propensity scores for future games by teamname and season
    future_wins = future_games.groupby(['teamname', 'season'])['pscore'].transform('sum')
    
    # Assign summed future wins back to original dataframe
    covers_data.loc[covers_data['week'] == w, f'exp_wins_wk{w}'] = future_wins

# Expected wins for week 12 are set to 0 by initialization

# Save the updated DataFrame to covers_data_cleaned.csv
covers_data.to_csv('covers_data_cleaned.csv', index=False)

# Display the first few rows to verify the calculations
print(covers_data[['teamname', 'season', 'week', 'exp_wins_naive', 'exp_wins', 'exp_win_pct', 'exp_wins_wk1', 'exp_wins_wk11']].head())