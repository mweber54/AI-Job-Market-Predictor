import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Filepath to the CSV file
file_path = '/workspaces/AI-Job-Market-Predictor/job_market_dataset.csv'

# Define column names based on the CSV structure
column_names = [
    "Job Title", "Industry", "Company Size", "Location", "Experience Level",
    "Education Level", "Skill", "Salary", "Remote Work", "Job Outlook"
]

# Read the CSV file
df = pd.read_csv(file_path, names=column_names, header=None)

# Display the first few rows of the dataframe
print("First 5 rows of the dataset:")
print(df.head())

# Explore dataset summary statistics
print("\nDataset summary:")
print(df.describe())

# Check for missing values and data types
print("\nDataset info:")
print(df.info())

# Preprocess the data
# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features and target variable
X = df.drop('Salary', axis=1)
y = df['Salary']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Print the predictions in a DataFrame
predictions_df = pd.DataFrame({'Actual Salary': y_test, 'Predicted Salary': y_pred})
print("\nPredictions:")
print(predictions_df.head(10))  # Display the first 10 predictions

# Plot the actual vs predicted salaries
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salaries')
plt.ylabel('Predicted Salaries')
plt.title('Actual vs Predicted Salaries')
plt.show()

# Example: Create a bar chart of average salaries by job title
avg_salaries = df.groupby('Job Title')['Salary'].mean()
avg_salaries.plot(kind='bar', figsize=(10, 6))

# Add title and labels
plt.title('Average Salaries by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Average Salary')

# Show the plot
plt.show()