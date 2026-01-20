import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("../data/cloud_logs.csv")

# Convert text columns to numeric for training the dataaaaaaa

encoder = LabelEncoder()
data['user'] = encoder.fit_transform(data['user'])
data['ip'] = encoder.fit_transform(data['ip'])

# Define features and label to understand malicious activity
X = data[['api_calls', 'data_mb', 'login_failures']]
y = data['label']

# Split dataset it will easy to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest model for classification
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict the score on test data
y_pred = model.predict(X_test)

# Accuracy for the given data
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
