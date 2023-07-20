import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('/Users/daniyalkhan/Downloads/dataset.csv')

# Preprocess the data (You might need to do more preprocessing depending on your data)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour

# Select your features and target variable
features = df.drop(columns=['is_fraud', 'trans_date_trans_time', 'merchant', 'category', 'city', 'state', 'job', 'dob', 'trans_num'])
# Remove unnecessary or hard to process columns
target = df['is_fraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Accuracy: ", accuracy_score(y_test, y_pred))
