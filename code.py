       #Customer Churn Prediction


import pandas as pd

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Check for missing values and handle them if needed
data.dropna(inplace=True)

# Split the data into features and target variable (churn)
X = data.drop('Churn', axis=1)
y = data['Churn']
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
encoder = LabelEncoder()
X['Gender'] = encoder.fit_transform(X['Gender'])
X['Contract'] = encoder.fit_transform(X['Contract'])

# Scale numerical features
scaler = StandardScaler()
X['TotalCharges'] = scaler.fit_transform(X['TotalCharges'].values.reshape(-1, 1))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import xgboost as xgb

# Initialize the XGBoost model
model = xgb.XGBClassifier()

# Train the model on the training data
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
