import pandas as pd

def calculate_statistics(df, varname, bcs_status):
    """
    Calculate summary statistics for a given variable and BCS status.
    """
    data_filtered = df[df['bcs'] == bcs_status]
    data_var = data_filtered[varname].dropna()

    mean_val = data_var.mean()
    std_dev = data_var.std()
    sample_size = data_var.count()
    first_year = df['year'][data_var.index].min() if sample_size > 0 else None
    last_year = df['year'][data_var.index].max() if sample_size > 0 else None
    teams = data_filtered['teamname'][data_var.index].nunique() if sample_size > 0 else None

    return {
        'Variable': varname,
        'Mean': mean_val,
        'Std Dev': std_dev,
        'N': sample_size,
        'Teams': teams,
        'First Year': first_year,
        'Last Year': last_year
    }

def main():
    # Load the CSV file
    df = pd.read_csv('Formatted_data.csv')

    # List of variables to calculate statistics for
    variables = [
        'lag_seasonwins', 'lag_seasongames', 'lag_exp_wins',
        'alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving',
        'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants',
        'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25'
    ]

    # Generate summary statistics
    summary_stats = []
    for varname in variables:
        summary_stats.append(calculate_statistics(df, varname, 1))
        summary_stats.append(calculate_statistics(df, varname, 0))

    # Convert the list of dictionaries to a DataFrame
    summary_df = pd.DataFrame(summary_stats)

    # Display the summary statistics DataFrame
    print(summary_df)
    summary_df.to_csv('Table1.csv', index = False)

    
if __name__ == "__main__":
    main()