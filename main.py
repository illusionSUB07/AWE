import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('/Users/daniyalkhan/Downloads/dataset.csv')

# Check the first few rows
print("First few rows of the dataset:")
print(df.head())

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Concise summary
print("\nDataframe summary:")
print(df.info())

# Value counts for 'is_fraud'
print("\nValue counts for 'is_fraud':")
print(df['is_fraud'].value_counts())


# Plot the distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['amt'], bins=50, color='blue', kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()

# Bar plot of fraudulent vs. non-fraudulent transactions
plt.figure(figsize=(10, 6))
sns.countplot(x='is_fraud', data=df)
plt.title('Fraudulent vs. Non-Fraudulent Transactions')
plt.show()

# Select only numeric columns for the correlation matrix
numeric_cols = df.select_dtypes(include=[np.number])

# Create a correlation matrix
corr = numeric_cols.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Scatter plot of merchant latitude and longitude
plt.figure(figsize=(10, 6))
sns.scatterplot(x='merch_lat', y='merch_long', hue='is_fraud', data=df)
plt.title('Scatter Plot of Merchant Locations with Fraud Indication')
plt.show()

# Examine the average transaction amount for each class of 'is_fraud'
print("\nAverage transaction amount grouped by 'is_fraud':")
print(df.groupby('is_fraud')['amt'].mean())

# Count the number of fraud and non-fraud transactions
fraud_counts = df['is_fraud'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values)
plt.title('Number of Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
plt.show()

