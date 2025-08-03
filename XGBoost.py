import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'dataset.csv' with the actual filename)
data = pd.read_csv('/Users/christinageorge/Desktop/logon.csv')

# Create target variable indicating attacks (1 for Logon, 0 for Logoff)
data['is_attack'] = data['activity'].apply(lambda x: 1 if x == 'Logon' else 0)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode 'id', 'user', and 'activity' columns
data['id'] = label_encoder.fit_transform(data['id'])
data['user'] = label_encoder.fit_transform(data['user'])
data['pc'] = label_encoder.fit_transform(data['pc'])
data['activity'] = label_encoder.fit_transform(data['activity'])

# Separate features and target variable
X = data.drop(['id','date', 'user', 'pc'], axis=1)  # Features
y = data['is_attack']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into DMatrix format, which is required for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Train the model
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
y_pred = xgb_model.predict(dtest)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Print performance metrics
print("\nPerformance Metrics for XGBoost:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

# Generate a bar plot for performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
plt.bar(metrics, values)
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for XGBoost')
plt.show()

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for XGBoost')
plt.show()
