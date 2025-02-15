import pandas as pd
import matplotlib.pyplot as plt

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

# Example: Create a bar chart of average salaries by job title
avg_salaries = df.groupby('Job Title')['Salary'].mean()
avg_salaries.plot(kind='bar', figsize=(10, 6))

# Add title and labels
plt.title('Average Salaries by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Average Salary')

# Show the plot
plt.show()