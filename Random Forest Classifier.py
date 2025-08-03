import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'dataset.csv' with the actual filename)
data = pd.read_csv('/Users/christinageorge/Desktop/logon.csv')

# Create target variable indicating attacks (1 for Logon, 0 for Logoff)
data['is_attack'] = data['activity'].apply(lambda x: 1 if x == 'Logon' else 0)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode 'id', 'user', 'pc', and 'activity' columns
data['id'] = label_encoder.fit_transform(data['id'])
data['user'] = label_encoder.fit_transform(data['user'])
data['pc'] = label_encoder.fit_transform(data['pc'])
data['activity'] = label_encoder.fit_transform(data['activity'])

# Separate features and target variable
X = data.drop(['id', 'date', 'user', 'pc'], axis=1)  # Features
y = data['is_attack']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print performance metrics
print("\nPerformance Metrics for Random Forest:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

# Save output to a file
with open('output.txt', 'w') as f:
    f.write("Accuracy: {}\n".format(accuracy))
    f.write("Precision: {}\n".format(precision))
    f.write("Recall: {}\n".format(recall))
    f.write("F1 Score: {}\n".format(f1))
    f.write("Confusion Matrix:\n{}\n".format(conf_matrix))

# Generate a bar plot for performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
plt.bar(metrics, values)
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for Random Forest')
plt.show()

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Random Forest')
plt.show()
