import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'dataset.csv' with the actual filename)
data = pd.read_csv('/Users/christinageorge/Desktop/logon.csv')

# Create target variable indicating attacks (1) and normal behavior (0)
data['is_attack'] = data['activity'].apply(lambda x: 1 if 'Logon' in x else 0)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode 'id', 'user', and 'activity' columns
data['id'] = label_encoder.fit_transform(data['id'])
data['user'] = label_encoder.fit_transform(data['user'])
data['activity'] = label_encoder.fit_transform(data['activity'])

# Separate features and target variable
X = data.drop(['date', 'user', 'pc'], axis=1)  # Features
y = data['is_attack']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set feature names explicitly for both training and test data
X_train.columns = [str(col) for col in X_train.columns]
X_test.columns = [str(col) for col in X_test.columns]

# Train the Isolation Forest classifier
classifier = IsolationForest(contamination=0.1)  # Adjust contamination if needed
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Convert predictions to positive (inlier) and negative (outlier) classes
y_pred[y_pred == 1] = 0  # Inlier
y_pred[y_pred == -1] = 1  # Outlier

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print performance metrics
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
plt.title('Performance Metrics')
plt.show()

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
