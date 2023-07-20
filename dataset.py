import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/Users/daniyalkhan/Downloads/dataset.csv')

# Examine the average transaction amount for each class of 'is_fraud'
print("\nAverage transaction amount grouped by 'is_fraud':")
print(df.groupby('is_fraud')['amt'].mean())


# Convert 'trans_date_trans_time' to datetime object and extract the hour
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour

# Plot the number of fraudulent transactions by hour
plt.figure(figsize=(10, 6))
sns.histplot(df[df['is_fraud'] == 1]['hour'], bins=24, color='blue', kde=False)
plt.title('Number of Fraudulent Transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Fraudulent Transactions')
plt.show()

# Print the number of unique merchants and categories
print("\nNumber of unique merchants:", df['merchant'].nunique())
print("Number of unique categories:", df['category'].nunique())
