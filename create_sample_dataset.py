import pandas as pd

# Create a sample dataset
data = {
    'Job_Title': ['Data Scientist', 'Machine Learning Engineer', 'AI Researcher', 'Data Analyst'],
    'Experience_Years': [2, 5, 3, 1],
    'Salary': [70000, 120000, 95000, 60000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('/workspaces/AI-Job-Market-Predictor/job_market_dataset.csv', index=False)

print("Sample dataset created and saved as 'job_market_dataset.csv'.")
