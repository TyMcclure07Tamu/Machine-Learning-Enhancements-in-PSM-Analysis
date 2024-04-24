import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load and prepare data
df = pd.read_csv('Formatted_data.csv')
needed_columns = ['teamname', 'year', 'seasonwins', 'alumni_ops_athletics', 'alum_non_athl_ops',
                  'alumni_total_giving', 'vse_alum_giving_rate', 'usnews_academic_rep_new',
                  'applicants', 'acceptance_rate', 'firsttime_outofstate', 'first_time_instate',
                  'sat_25', 'seasongames']
df = df[needed_columns]

# Create lagged variables
for lag in range(1, 11):
    df[f'lag_seasonwins{lag}'] = df.groupby('teamname')['seasonwins'].shift(lag)

# Define the treatment variable
df['treated'] = (df['seasonwins'] > df['seasonwins'].median()).astype(int)

# Covariates for the model
covariates = ['alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving',
              'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants',
              'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25'] + \
              [f'lag_seasonwins{i}' for i in range(1, 11)]

# Scaling and imputation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[covariates].astype(float))
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df_scaled)
df_imputed = pd.DataFrame(df_imputed, columns=covariates)

# Random Forest for propensity score calculation
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(df_imputed, df['treated'])
df['propensity_score_rf'] = rf_classifier.predict_proba(df_imputed)[:, 1]

# Use KNN for matching based on Random Forest propensity scores
control = df[df['treated'] == 0]
treated = df[df['treated'] == 1]
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['propensity_score_rf']])
distances, indices = nn.kneighbors(treated[['propensity_score_rf']])
matched_control = control.iloc[indices.flatten()]
matched_data = pd.concat([treated, matched_control])

# Logistic Regression for Propensity Score Calculation
logit_model = LogisticRegression(max_iter=1000, solver='liblinear')
logit_model.fit(df_imputed, df['treated'])
df['propensity_score_logit'] = logit_model.predict_proba(df_imputed)[:, 1]

# Matching using Random Forest propensity scores
nn_rf = NearestNeighbors(n_neighbors=1, metric='euclidean')
control_rf = df[df['treated'] == 0]
treated_rf = df[df['treated'] == 1]
nn_rf.fit(control_rf[['propensity_score_rf']])
distances_rf, indices_rf = nn_rf.kneighbors(treated_rf[['propensity_score_rf']])
matched_control_rf = control_rf.iloc[indices_rf.flatten()]
matched_data_rf = pd.concat([treated_rf, matched_control_rf])

# Matching using Logistic Regression propensity scores
nn_logit = NearestNeighbors(n_neighbors=1, metric='euclidean')
control_logit = df[df['treated'] == 0]
treated_logit = df[df['treated'] == 1]
nn_logit.fit(control_logit[['propensity_score_logit']])
distances_logit, indices_logit = nn_logit.kneighbors(treated_logit[['propensity_score_logit']])
matched_control_logit = control_logit.iloc[indices_logit.flatten()]
matched_data_logit = pd.concat([treated_logit, matched_control_logit])

# Evaluate the logistic regression model
print("Logistic Regression R^2 Score:", logit_model.score(df_imputed, df['treated']))  # Accuracy, not R²
print("Confusion Matrix:\n", confusion_matrix(df['treated'], logit_model.predict(df_imputed)))
print("Classification Report:\n", classification_report(df['treated'], logit_model.predict(df_imputed)))

# Elastic Net Model
elastic_net = ElasticNetCV(cv=5, random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
elastic_net.fit(df_imputed, df['seasonwins'])  # Assume df['seasonwins'] is available correctly
print(f"Best alpha: {elastic_net.alpha_}")
print(f"Best l1 ratio: {elastic_net.l1_ratio_}")
print(f"Elastic Net Coefficients: {elastic_net.coef_}")
print(f"Elastic Net Intercept: {elastic_net.intercept_}")

# Perform nearest neighbor matching
control = df[df['treated'] == 0]
treated = df[df['treated'] == 1]
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['propensity_score_rf']])
distances, indices = nn.kneighbors(treated[['propensity_score_rf']])
matched_control = control.iloc[indices.flatten()]

# Create a matched dataset
matched_data = pd.concat([treated, matched_control])

# Proceed with your analysis on matched_data
X_matched = matched_data[covariates + ['propensity_score_rf']]
y_matched = matched_data['seasonwins']

# Scale data before imputation
scaler = StandardScaler()
X_matched_scaled = scaler.fit_transform(X_matched)
X_matched_scaled = pd.DataFrame(X_matched_scaled, columns=covariates + ['propensity_score_rf'])

# Apply KNN imputation
imputer = KNNImputer(n_neighbors=5)
X_matched_imputed = imputer.fit_transform(X_matched_scaled)
X_matched_imputed = pd.DataFrame(X_matched_imputed, columns=covariates + ['propensity_score_rf'])

# Replace old values with imputed values in matched data
matched_data[covariates + ['propensity_score_rf']] = X_matched_imputed

# Visualizing propensity score distributions
sns.histplot(data=df, x='propensity_score_rf', hue='treated', element='step', stat='density', common_norm=False)
plt.title('Propensity Score Distribution by Treatment Status')
plt.savefig('propensity_score_rf_distribution.png')

# Random Forest analysis including propensity score in predictors
covariates.append('propensity_score_rf')
X = df[covariates]
y = df['seasonwins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Random Forest Mean Squared Error:", mse)

# Visualize feature importances
feature_importance = rf.feature_importances_
important_features = pd.DataFrame({'Feature': covariates, 'Importance': feature_importance})
important_features = important_features.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=important_features)
plt.title('Feature Importance Including Propensity Score')
plt.savefig('Feature_Importance.png')

print(df)

varlist = [
    'alumni_ops_athletics', 'alum_non_athl_ops', 'alumni_total_giving',
    'vse_alum_giving_rate', 'usnews_academic_rep_new', 'applicants',
    'acceptance_rate', 'firsttime_outofstate', 'first_time_instate', 'sat_25' , 'seasongames', 'seasonwins'
]

# Create lagged variables for each needed column including 'seasonwins' and 'seasongames'
lags = [1, 2, 3]
for col in needed_columns:
    if col not in ['teamname', 'year']:
        for lag in lags:
            df[f'lag{lag}_{col}'] = df.groupby('teamname')[col].shift(lag)

df['weight'] = np.where(df['treated'] == 1, 
                        1 / df['propensity_score_rf'], 
                        1 / (1 - df['propensity_score_rf']))


for var in varlist:
    df[f'r{var}'] = df[var] - df[f'lag2_{var}']

df.to_csv('ML_lag_variable_check.csv')
# Proceed with your regression analysis as previously coded
# Initialize DataFrame to store results, etc.
results_df = pd.DataFrame()

counter = 1

for varname in varlist:
    formula_wls = f"r{varname} ~ rseasonwins + lag_seasonwins3 + lag1_seasongames + lag3_seasongames + year + propensity_score_rf"
    try:
        wls_model = smf.wls(formula=formula_wls, data=df, weights=df['weight']).fit()
        print(f"WLS Model Summary for {varname}:")
        print(wls_model.summary())
        
    except Exception as e:
        print(f"Failed to fit WLS model for {varname}:", str(e))

