import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

##############################################
# PART 1: TRAINING PHASE (Using Dataset with Actual AI Impact)
##############################################

# Load training dataset 
train_data = pd.read_csv(r'C:\Users\User\ai predictor test\My_Data.csv')
print("Training Data - First 5 rows:")
print(train_data.head())
print("\nTraining Data Info:")
print(train_data.info())

# Clean the target column "AI Impact":
# Remove the '%' sign and convert to numeric.
train_data['AI Impact'] = train_data['AI Impact'].str.replace('%', '', regex=True)
train_data['AI Impact'] = pd.to_numeric(train_data['AI Impact'], errors='coerce')
train_data.dropna(subset=['AI Impact'], inplace=True)

# Set up features and target.
X_train_full = train_data.drop(columns=['AI Impact'])
y_train = train_data['AI Impact']

# Create a new "Job" column from "Job titiles" by removing the prefix.
X_train_full['Job'] = X_train_full['Job titiles'].str.replace(r'^Job titiles_', '', regex=True)
# Drop the original "Job titiles" column.
X_train_full = X_train_full.drop(columns=['Job titiles'])

# Process features: convert non-numeric columns to string, then one-hot encode.
for col in X_train_full.columns:
    if not pd.api.types.is_numeric_dtype(X_train_full[col]):
        X_train_full[col] = X_train_full[col].astype(str)
X_train_processed = pd.get_dummies(X_train_full, drop_first=True)
X_train_processed = X_train_processed.replace([np.inf, -np.inf], np.nan)
X_train_processed = X_train_processed.apply(pd.to_numeric, errors='coerce')
X_train_processed = X_train_processed.fillna(X_train_processed.mean())

# Standardize training features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)

# Train a RandomForestRegressor to predict AI Impact.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate training performance.
y_train_pred = model.predict(X_train_scaled)
print("Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Training R^2:", r2_score(y_train, y_train_pred))

##############################################
# PART 2: PREDICTION PHASE (Using Dataset without AI Impact)
##############################################

# Load prediction dataset - this file does NOT include the "AI Impact" column.
pred_data = pd.read_csv(r'C:\Users\User\ai predictor test\My_Prediction_Data.csv')
print("Prediction Data - First 5 rows:")
print(pred_data.head())
print("\nPrediction Data Info:")
print(pred_data.info())

# Process prediction data similarly: create "Job" from "Job titiles" and drop original.
pred_data['Job'] = pred_data['Job titiles'].str.replace(r'^Job titiles_', '', regex=True)
pred_data = pred_data.drop(columns=['Job titiles'])

# Convert non-numeric columns to string and one-hot encode.
for col in pred_data.columns:
    if not pd.api.types.is_numeric_dtype(pred_data[col]):
        pred_data[col] = pred_data[col].astype(str)
pred_processed = pd.get_dummies(pred_data, drop_first=True)
pred_processed = pred_processed.replace([np.inf, -np.inf], np.nan)
pred_processed = pred_processed.apply(pd.to_numeric, errors='coerce')
pred_processed = pred_processed.fillna(pred_processed.mean())

# Align the prediction data columns to the training features.
pred_processed = pred_processed.reindex(columns=X_train_processed.columns, fill_value=0)

# Standardize prediction features using the same scaler.
X_pred_scaled = scaler.transform(pred_processed)

# Predict AI Impact for the new jobs.
predictions = model.predict(X_pred_scaled)

# Add predictions to the prediction DataFrame.
df_pred_plot = pred_data.copy()
df_pred_plot['Predicted_AI_Impact'] = predictions

# For visualization, sort the predicted data by predicted AI Impact and create a numeric y-index.
df_pred_plot.sort_values(by='Predicted_AI_Impact', inplace=True)
df_pred_plot.reset_index(drop=True, inplace=True)
df_pred_plot['y_index'] = df_pred_plot.index * 5  # Adjust multiplier for vertical spacing.

##############################################
# PART 3: VISUALIZATION: Compare Actual vs. Predicted Data Side by Side
##############################################

# Prepare actual data for plotting from training file.
# ---------- PART 3: VISUALIZATION: Compare Actual vs. Predicted Data Side by Side -----------
# Process actual data for plotting from training file.
df_actual_plot = train_data[['Job titiles', 'AI Impact']].copy()
df_actual_plot['Job'] = df_actual_plot['Job titiles'].str.replace(r'^Job titiles_', '', regex=True)
df_actual_plot.rename(columns={'AI Impact': 'AI_Impact'}, inplace=True)
# Force conversion to string, remove '%', then convert to numeric.
df_actual_plot['AI_Impact'] = df_actual_plot['AI_Impact'].astype(str).str.replace('%', '', regex=True)
df_actual_plot['AI_Impact'] = pd.to_numeric(df_actual_plot['AI_Impact'], errors='coerce')
df_actual_plot.dropna(subset=['AI_Impact'], inplace=True)
df_actual_plot.sort_values(by='AI_Impact', inplace=True)
df_actual_plot.reset_index(drop=True, inplace=True)
df_actual_plot['y_index'] = df_actual_plot.index * 5

# Create subplots to display the two scatter plots side by side.
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Actual AI Impact", "Predicted AI Impact"),
    horizontal_spacing=0.1
)

# Add scatter plot for Actual Data.
fig.add_trace(
    go.Scatter(
        x=df_actual_plot['AI_Impact'],
        y=df_actual_plot['y_index'],
        mode='markers',
        marker=dict(color='blue', size=4),
        hovertemplate="Job: %{customdata}<br>AI Impact: %{x:.2f}%<extra></extra>",
        customdata=df_actual_plot[['Job']].to_numpy()
    ),
    row=1, col=1
)

# Add scatter plot for Predicted Data.
fig.add_trace(
    go.Scatter(
        x=df_pred_plot['Predicted_AI_Impact'],
        y=df_pred_plot['y_index'],
        mode='markers',
        marker=dict(color='blue', size=4),
        hovertemplate="Job: %{customdata}<br>Predicted AI Impact: %{x:.2f}%<extra></extra>",
        customdata=df_pred_plot[['Job']].to_numpy()
    ),
    row=1, col=2
)

# Configure x-axes for both subplots (set range from 0 to 100).
fig.update_xaxes(title_text="AI Impact (%)", range=[0, 100], row=1, col=1)
fig.update_xaxes(title_text="Predicted AI Impact (%)", range=[0, 100], row=1, col=2)

# For a clean look, hide y-axis tick labels but label the y-axis as "Jobs".
fig.update_yaxes(title_text="Jobs", showticklabels=False, row=1, col=1)
fig.update_yaxes(title_text="Jobs", showticklabels=False, row=1, col=2)

# Update overall layout.
fig.update_layout(
    title_text="Comparison of Actual vs. Predicted Job Vulnerability to AI Automation",
    title_x=0.5,
    plot_bgcolor="white",
    width=1200,
    height=800,
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    xaxis2=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis2=dict(showgrid=True, gridcolor='lightgrey')
)

fig.show()

