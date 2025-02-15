import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # as an example model
from sklearn.metrics import mean_squared_error, r2_score

# Check if the dataset file exists
dataset_path = '/workspaces/AI-Job-Market-Predictor/job_market_dataset.csv'
if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}. Creating a sample dataset...")
    import create_sample_dataset

# Load the dataset
data = pd.read_csv(dataset_path)

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Explore dataset summary statistics
print("\nDataset summary:")
print(data.describe())

# Check for missing values and data types
print("\nDataset info:")
print(data.info())
