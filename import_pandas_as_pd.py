import pandas as pd

# Loading Data from the File
df = pd.read_csv('College_data_outbound.csv')

# Display the first 5 rows of the dataframe
print(df.head())

# Load the dataset but return a StataReader object instead of a DataFrame
with pd.read_csv('Covers_data_outbound.csv', iterator=True) as reader:
    # Extract variable labels
    variable_names = df.columns.tolist()

# Print the variable names
print(variable_names)

